"""MCP tool definitions for OpenFOAM."""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import sys
import threading
import time as _time
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp_types import ToolAnnotations
from pydantic import Field

from . import __version__, dictparse, fields, logs, mesh, plotting, postproc
from .config import Settings
from .errors import FoamError
from .foam import FoamEnv, fatal_error, strip_banner
from .jobs import Job, JobManager
from .paths import case_file, display, from_windows, resolve_case, within
from .render import ParaView

INSTRUCTIONS = """\
OpenFOAM CFD server. Typical workflow:
1. foam_info -> what OpenFOAM version/flavour is installed (openfoam.org uses `foamRun` with a `solver`
   entry in controlDict; openfoam.com/ESI uses per-solver apps such as simpleFoam).
2. list_tutorials (query/contains) -> pick the closest tutorial; create_case clones it into the workspace.
   Prefer adapting a tutorial over writing a case from scratch.
3. case_summary -> solver, mesh, patches, boundary/initial conditions, numerics, function objects.
4. Edit with set_dict (entries like `endTime`, `boundaryField/inlet/value`); write_file for whole new files;
   import_file to bring in geometry (STL/OBJ) or files from $FOAM_TUTORIALS/resources.
   foam_reference lists valid boundary conditions, models, function objects and app options.
5. Mesh: run blockMesh / snappyHexMesh (see the case's Allrun for the exact commands), then check_mesh.
6. Solve: run(app="foamRun" or the solver, np=N) returns a job id when it outlasts the wait;
   poll solver_progress(case) / job_status(job_id). stop_job(mode="write") ends a run cleanly.
7. Inspect: render (field images), field_stats, post_process (function objects, e.g. forceCoeffs, yPlus),
   list_postprocessing / read_postprocessing (tables, optional plots).
Case arguments are names inside the workspace (or absolute paths inside allowed roots).
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

CaseArg = Annotated[str, Field(description="Case name in the workspace (e.g. 'pitzDaily') or an absolute path inside an allowed root.")]


class App:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs = JobManager(settings.state_dir)
        self.logs = logs.LogCache()
        self._foam: FoamEnv | None = None
        self._foam_error: str | None = None
        self._lock = threading.Lock()
        self._paraview: ParaView | None = None
        self._tutorials: list[dict] | None = None

    @property
    def foam(self) -> FoamEnv:
        with self._lock:
            if self._foam is None:
                if self._foam_error:
                    raise FoamError(self._foam_error)
                try:
                    s = self.settings
                    self._foam = FoamEnv.detect(
                        bashrc=s.bashrc,
                        launcher=s.launcher,
                        container=s.container,
                        image=s.container_image,
                        container_cli=s.container_cli,
                        mounts=s.roots,
                    )
                except FoamError as e:
                    self._foam_error = str(e)
                    raise
            return self._foam

    @property
    def paraview(self) -> ParaView:
        if self._paraview is None:
            self._paraview = ParaView(self.settings.paraview_python, self.foam.host_env)
        return self._paraview

    def case(self, case: str, must_exist: bool = True) -> Path:
        return resolve_case(self.settings, case, must_exist)

    def name(self, case_dir: Path) -> str:
        return display(case_dir, self.settings)


APP: App | None = None


def _app() -> App:
    global APP
    if APP is None:
        APP = App(Settings.from_env())
    return APP


def configure(settings: Settings) -> App:
    global APP
    APP = App(settings)
    return APP


mcp = MCPServer(
    name="openfoam",
    title="OpenFOAM",
    version=__version__,
    instructions=INSTRUCTIONS,
    website_url="https://github.com/0xFFD/openfoam-mcp",
)


def _encode(result: Any) -> Any:
    """One compact JSON text block per result (images pass through), instead of the SDK default of
    pretty-printed text plus a duplicate structuredContent copy, which doubles token usage."""
    if isinstance(result, list) and any(isinstance(x, Image) for x in result):
        return [x if isinstance(x, Image) else _encode(x) for x in result]
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def tool(annotations: ToolAnnotations):
    """Register a tool; the undecorated function stays callable from Python with native return values."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return _encode(fn(*args, **kwargs))

        mcp.add_tool(wrapper, annotations=annotations, structured_output=False)
        return fn

    return deco


# ============================================================================ helpers


def _truncate(value: Any, n: int = 160) -> Any:
    if isinstance(value, str) and len(value) > n:
        return value[: n - 3] + "..."
    return value


def _compact(d: Any, depth: int = 3, max_items: int = 40) -> Any:
    """Shrink a parsed dictionary for display: limited depth, truncated strings."""
    if not isinstance(d, dict):
        return _truncate(d)
    if depth <= 0:
        return f"{{...{len(d)} entries}}"
    out = {}
    for i, (k, v) in enumerate(d.items()):
        if k == "FoamFile":
            continue
        if i >= max_items:
            out["..."] = f"{len(d) - max_items} more"
            break
        out[k] = _compact(v, depth - 1, max_items)
    return out


def _read_dict(path: Path) -> dict:
    try:
        return dictparse.read(path)
    except (OSError, UnicodeError, RecursionError, ValueError):
        return {}


def _control(case_dir: Path) -> dict:
    return _read_dict(case_dir / "system" / "controlDict")


def _end_time(case_dir: Path) -> float | None:
    c = _control(case_dir)
    try:
        if dictparse.first_word(c.get("stopAt")) in (None, "endTime"):
            return float(dictparse.first_word(c.get("endTime")) or "nan")
    except ValueError:
        pass
    return None


def _solver_of(control: dict) -> str | None:
    return dictparse.first_word(control.get("solver")) or dictparse.first_word(control.get("application"))


def _mtime_skew(case_dir: Path) -> float:
    """fileModificationSkew in effect for running solvers (printed in every log header; default 10 s)."""
    for log in sorted(case_dir.glob("log.*"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        with open(log, errors="replace") as f:
            head = f.read(4000)
        if m := re.search(r"fileModificationSkew (\d+(?:\.\d+)?)", head):
            return float(m.group(1))
    return 10.0


def _edit_live(case_dir: Path, path: Path, old_mtime: float | None) -> bool:
    """Make an edit visible to running solvers.

    With timeStamp-based modification checking OpenFOAM ignores a change unless the new mtime is more
    than fileModificationSkew seconds newer than the previous one, so rapid successive edits are
    silently dropped. Push the mtime past that window when a job is running on the case.
    """
    a = _app()
    if not a.jobs.running_for(a.name(case_dir)) or old_mtime is None:
        return False
    target = max(_time.time(), old_mtime + _mtime_skew(case_dir) + 1)
    os.utime(path, (target, target))
    return True


def _rotate_log(log: Path) -> None:
    if log.exists():
        os.replace(log, log.with_name(log.name + ".1"))


def _wait(job: Job, seconds: float) -> Job:
    a = _app()
    deadline = _time.time() + max(0.0, seconds)
    job = a.jobs.refresh(job)
    while job.running and _time.time() < deadline:
        _time.sleep(0.4)
        job = a.jobs.refresh(job)
    return _after_finish(job)


def _after_finish(job: Job) -> Job:
    """Restore controlDict entries changed by a graceful stop once the job has ended."""
    if job.running or not job.restore:
        return job
    a = _app()
    try:
        case_dir = a.case(job.case)
        for entry, value in job.restore.items():
            a.foam.run_app("foamDictionary", ["system/controlDict", "-entry", entry, "-set", value], cwd=case_dir)
    except FoamError:
        pass
    job.restore = {}
    a.jobs.save(job)
    return job


def _job_report(job: Job, tail_lines: int = 20) -> dict:
    a = _app()
    rep = job.public()
    log = Path(job.log)
    if log.is_file():
        data = a.logs.get(log)
        if data.residuals:
            case_dir = Path(job.log).parent
            rep["progress"] = logs.summarize(data, _end_time(case_dir))
        if not job.running or not data.residuals:
            err = fatal_error(logs.tail(log, 200)) if not job.running and job.returncode else None
            if err and "progress" not in rep:
                rep["error"] = err
            rep["log_tail"] = logs.tail(log, 6 if err else tail_lines)
        if not job.running and job.returncode == 0:
            # Some utilities are kept as stubs that only print a notice and exit successfully
            # (e.g. reconstructParMesh in openfoam.org 12+), so a zero exit code is not proof of work done.
            notice = re.search(r"^.*(?:has been superseded|has been replaced|is deprecated|no longer supported).*$",
                               logs.tail(log, 40), re.M | re.I)
            if notice:
                rep["warning"] = f"The application reported: '{notice.group(0).strip()}' - it may not have done anything. See log_tail."
                rep["log_tail"] = logs.tail(log, 15)
    if job.running:
        rep["hint"] = f"Still running. Poll with job_status('{job.id}') or solver_progress('{job.case}')."
    return rep


_ENV_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _start(case_dir: Path, app_name: str, args: list[str], np: int = 1, log_name: str | None = None) -> Job:
    a = _app()
    foam = a.foam
    prefix = foam.resolve_app(app_name, case_dir, a.settings.allow_scripts)
    is_script = prefix[0] == "bash"
    # Expand $VARS (e.g. $FOAM_TUTORIALS) like a shell would, without involving a shell.
    args = [_ENV_VAR.sub(lambda m: foam.env.get(m.group(1) or m.group(2), m.group(0)), x) for x in args]
    argv = [*prefix, *args]
    if np > 1 and not is_script:
        if not foam.mpirun:
            raise FoamError("mpirun not found; install OpenMPI to run in parallel.")
        mpi = [foam.mpirun, "-np", str(np)]
        if np > (os.cpu_count() or 1):
            mpi.append("--oversubscribe")
        argv = [*mpi, *prefix, *args]
        if "-parallel" not in args:
            argv.append("-parallel")
    log = case_dir / (log_name or f"log.{Path(app_name).name.lstrip('./')}")
    _rotate_log(log)
    return a.jobs.start(case_dir, a.name(case_dir), app_name, argv, log, foam, np=np)


def _ensure_decomposed(case_dir: Path, np: int, method: str | None, copy_zero: bool = False) -> str | None:
    """Make sure processor* directories exist for `np` subdomains. Returns a note if work was done.

    `copy_zero` copies the initial fields instead of decomposing them - needed before parallel meshing,
    when 0/ refers to patches that only exist once the mesher has run.
    """
    procs = fields.processor_dirs(case_dir)
    if procs:
        if len(procs) != np:
            raise FoamError(
                f"Case is decomposed into {len(procs)} processors but np={np}. Use np={len(procs)}, or "
                "reconstruct (run reconstructPar) and clean_case(keep_processor=False) before re-decomposing."
            )
        return None
    a = _app()
    note = None
    dict_path = case_dir / "system" / "decomposeParDict"
    if not dict_path.exists():
        key = _decomposer_key({})
        dict_path.write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
            f"numberOfSubdomains {np};\n\n{key:<16}{method or 'scotch'};\n"
        )
    else:
        # Keep the case's own method (it may carry coefficients) unless one was requested explicitly.
        key = _decomposer_key(_read_dict(dict_path))
        a.foam.run_app("foamDictionary", ["system/decomposeParDict", "-entry", "numberOfSubdomains", "-set", str(np)], case_dir)
        if not method and (mismatch := _split_mismatch(_read_dict(dict_path), np)):
            method = "scotch"
            note = mismatch
        if method:
            a.foam.run_app("foamDictionary", ["system/decomposeParDict", "-entry", key, "-set", method], case_dir)
    used = dictparse.first_word(_read_dict(dict_path).get(key)) or method
    if note:
        used = f"{used}; {note}"
    res = a.foam.run_app("decomposePar", ["-force", *(["-copyZero"] if copy_zero else [])], case_dir, timeout=3600)
    (case_dir / "log.decomposePar").write_text(res.output)
    if res.returncode != 0:
        raise FoamError("decomposePar failed:\n" + (fatal_error(res.output) or logs_tail_text(res.output)))
    return f"decomposed into {np} subdomains with method '{used}' (log.decomposePar)"


def _decomposer_key(d: dict) -> str:
    """openfoam.org >= 10 calls the entry `decomposer`; older versions and openfoam.com use `method`."""
    if "decomposer" in d:
        return "decomposer"
    if "method" in d:
        return "method"
    f = _app().foam
    return "decomposer" if f.flavor == "org" and (f.major or 0) >= 10 else "method"


def _split_mismatch(d: dict, np: int) -> str | None:
    """For simple/hierarchical decomposition, report when the `n (a b c)` split does not give np domains."""
    method = dictparse.first_word(d.get(_decomposer_key(d)))
    if method not in ("simple", "hierarchical"):
        return None
    for scope in (d, d.get(method), d.get(f"{method}Coeffs"), d.get("coeffs")):
        if isinstance(scope, dict) and isinstance(scope.get("n"), str):
            nums = re.findall(r"\d+", scope["n"])
            if len(nums) == 3:
                product = int(nums[0]) * int(nums[1]) * int(nums[2])
                if product != np:
                    return f"switched from {method} (n {scope['n']} = {product} domains) to scotch for np={np}"
    return None


def logs_tail_text(text: str, n: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


# ============================================================================ environment


@tool(READ_ONLY)
def foam_info() -> dict:
    """OpenFOAM installation, server configuration and resources (version, flavour, paths, cores, ParaView)."""
    a = _app()
    f = a.foam
    info = {
        "openfoam": {
            "version": f.version,
            "flavor": "openfoam.org (Foundation)" if f.flavor == "org" else "openfoam.com (ESI)",
            "project_dir": str(f.project_dir),
            "source": f.origin,
            **f.runner.describe(),
            "tutorials": str(f.tutorials),
            "solver_style": "foamRun + `solver` in controlDict" if f.uses_foam_run else "one application per solver",
            "post_process_app": f.post_process_app,
            "applications": len(f.apps),
        },
        "server": {
            "version": __version__,
            "workspace": str(a.settings.workspace),
            "extra_roots": [str(r) for r in a.settings.extra_roots],
            "scripts_allowed": a.settings.allow_scripts,
            "default_wait_seconds": a.settings.wait_seconds,
        },
        "machine": {"platform": sys.platform, "cpus": os.cpu_count(), "mpirun": f.mpirun},
        "running_jobs": [j.public() for j in a.jobs.all() if j.running],
    }
    try:
        info["paraview"] = {"python": a.paraview.resolve()[0], "version": a.paraview.resolve()[2]}
    except FoamError as e:
        info["paraview"] = {"error": str(e)}
    return info


@tool(READ_ONLY)
def foam_reference(
    topic: Annotated[
        str,
        Field(
            description="One of: solvers, apps, scalarBCs, vectorBCs, functionObjects, functions (configured "
            "post-processing templates), fvModels, fvConstraints, tables, 'table:<name>' (e.g. table:RAScompressibleMomentumTransportModel), "
            "'search:<name>' (which tables contain a type), 'help:<app>' (an application's options)."
        ),
    ],
    filter: Annotated[str | None, Field(description="Case-insensitive substring to filter output lines.")] = None,
) -> str:
    """Look up what OpenFOAM supports: valid boundary-condition types, models, function objects, solvers, app options."""
    f = _app().foam
    t = topic.strip()
    cwd = _app().settings.workspace
    if t == "apps":
        text = "\n".join(sorted(f.apps))
    elif t == "solvers" and not f.uses_foam_run:
        text = "\n".join(sorted(a for a in f.apps if a.endswith("Foam")))
    elif t.startswith("help:"):
        res = f.run_app(t[5:].strip(), ["-help"], cwd)
        text = res.output
    elif t == "functions":
        res = f.run_app(f.post_process_app, ["-list"], cwd)
        text = strip_banner(res.output)
    else:
        if f.flavor != "org" or "foamToC" not in f.apps:
            raise FoamError("This topic needs foamToC (openfoam.org). Try 'apps', 'functions' or 'help:<app>'.")
        flags = {
            "solvers": ["-solvers"],
            "scalarBCs": ["-scalarBCs"],
            "vectorBCs": ["-vectorBCs"],
            "functionObjects": ["-functionObjects"],
            "fvModels": ["-fvModels"],
            "fvConstraints": ["-fvConstraints"],
            "tables": ["-tables"],
        }
        if t in flags:
            argv = flags[t]
        elif t.startswith("table:"):
            argv = ["-table", t[6:].strip()]
        elif t.startswith("search:"):
            argv = ["-search", t[7:].strip()]
        else:
            raise FoamError(f"Unknown topic '{topic}'.")
        res = f.run_app("foamToC", argv, cwd, timeout=180)
        text = strip_banner(res.output)
    lines = text.splitlines()
    if filter:
        lines = [ln for ln in lines if filter.lower() in ln.lower()]
    if len(lines) > 400:
        lines = lines[:400] + [f"... {len(lines) - 400} more lines; use filter"]
    return "\n".join(lines) or "(no matches)"


# ============================================================================ cases


_MESHERS = (
    "blockMesh", "snappyHexMesh", "extrudeMesh", "refineMesh", "mirrorMesh", "fluentMeshToFoam",
    "gmshToFoam", "ideasUnvToFoam", "foamyHexMesh", "createBaffles", "splitMeshRegions",
)


def _tutorial_index() -> list[dict]:
    a = _app()
    if a._tutorials is not None:
        return a._tutorials
    root = a.foam.tutorials
    index = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if (d / "system" / "controlDict").is_file():
            dirnames[:] = []
            control = _read_dict(d / "system" / "controlDict")
            sysfiles = {p.name for p in (d / "system").iterdir()}
            allrun = (d / "Allrun").read_text(errors="replace") if (d / "Allrun").is_file() else ""
            meshing = [m for m in _MESHERS if re.search(rf"\b{m}\b", allrun) or f"{m}Dict" in sysfiles]
            if not meshing and (d / "constant" / "polyMesh").is_dir():
                meshing = ["polyMesh (supplied)"]
            index.append(
                {
                    "tutorial": str(d.relative_to(root)),
                    "solver": _solver_of(control),
                    "allrun": bool(allrun),
                    "mesh": meshing,
                    "parallel": "decomposeParDict" in sysfiles,
                }
            )
        dirnames[:] = sorted(n for n in dirnames if not n.startswith((".", "processor")))
    a._tutorials = index
    return index


@tool(READ_ONLY)
def list_tutorials(
    query: Annotated[str | None, Field(description="Words that must all appear in the tutorial path or solver, e.g. 'pitzDaily' or 'VoF dam'.")] = None,
    contains: Annotated[str | None, Field(description="Regex searched inside the tutorial files, e.g. 'kOmegaSST' or 'forceCoeffs' - find working examples of a feature.")] = None,
    limit: int = 40,
) -> dict:
    """Find OpenFOAM tutorial cases to start from (and working examples of any keyword or feature)."""
    a = _app()
    items = _tutorial_index()
    if query:
        words = query.lower().split()
        items = [t for t in items if all(w in f"{t['tutorial']} {t['solver']}".lower() for w in words)]
    results = []
    if contains:
        rx = re.compile(contains)
        root = a.foam.tutorials
        for t in items:
            d = root / t["tutorial"]
            hits = []
            for p in sorted(d.rglob("*")):
                if not p.is_file() or p.stat().st_size > 1_000_000 or "polyMesh" in p.parts:
                    continue
                try:
                    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{p.relative_to(d)}:{i}: {line.strip()[:120]}")
                            break
                except OSError:
                    continue
                if len(hits) >= 3:
                    break
            if hits:
                results.append({**t, "matches": hits})
            if len(results) >= limit:
                break
    else:
        results = items[:limit]
    return {"tutorials_dir": str(a.foam.tutorials), "count": len(results), "total_indexed": len(_tutorial_index()), "results": results}


_RESULT_DIR = re.compile(r"^(processor\d+|postProcessing|images|VTK|dynamicCode)$")


@tool(WRITES)
def create_case(
    name: Annotated[str, Field(description="New case name (directory) in the workspace.")],
    source: Annotated[str, Field(description="Tutorial path from list_tutorials (e.g. 'incompressibleFluid/pitzDaily') or an existing case name/path to copy.")],
    overwrite: Annotated[bool, Field(description="Replace an existing case with this name.")] = False,
    include_results: Annotated[bool, Field(description="When copying a case, also copy time directories, logs and postProcessing.")] = False,
) -> dict:
    """Create a case by cloning a tutorial or copying an existing case (results are left behind unless include_results)."""
    a = _app()
    dest = a.case(name, must_exist=False)
    tut = a.foam.tutorials / source.strip("/")
    if (tut / "system").is_dir() and within(tut.resolve(), a.foam.tutorials.resolve()):
        src, kind = tut, "tutorial"
    else:
        src, kind = a.case(source), "case"
    if dest.exists():
        if not overwrite:
            raise FoamError(f"Case '{name}' already exists. Pass overwrite=true or choose another name.")
        if a.jobs.running_for(a.name(dest)):
            raise FoamError(f"Case '{name}' has running jobs; stop them first.")
        shutil.rmtree(dest)

    times = fields.time_dirs(src)
    first = times[0] if times else None

    def ignore(d: str, names: list[str]) -> set[str]:
        if include_results or Path(d) != src:
            return set()
        skip = {n for n in names if _RESULT_DIR.match(n) or n.startswith("log.") or n.endswith(".foam")}
        skip |= {n for n in names if n in times and n != first}
        return skip

    shutil.copytree(src, dest, symlinks=True, ignore=ignore)
    out = {"case": a.name(dest), "path": str(dest), "from": f"{kind}:{source}"}
    allrun = dest / "Allrun"
    if allrun.is_file():
        out["allrun"] = "\n".join(
            ln for ln in allrun.read_text(errors="replace").splitlines() if ln.strip() and not ln.startswith("#")
        )
        out["note"] = "Allrun shows the intended steps; run them individually with run(), or run(app='Allrun')."
    out["summary"] = case_summary(a.name(dest))
    return out


@tool(READ_ONLY)
def list_cases() -> list[dict]:
    """List cases in the workspace with solver, mesh/result state and running jobs."""
    a = _app()
    ws = a.settings.workspace
    running = {}
    for j in a.jobs.all():
        if j.running:
            running.setdefault(j.case, []).append(j.id)
    out = []
    for control in sorted(ws.glob("*/system/controlDict")) + sorted(ws.glob("*/*/system/controlDict")):
        d = control.parent.parent
        if any(p.startswith(".") for p in d.relative_to(ws).parts):
            continue
        times = fields.time_dirs(d)
        name = a.name(d)
        out.append(
            {
                "case": name,
                "solver": _solver_of(_read_dict(control)),
                "mesh": (d / "constant" / "polyMesh" / "faces").exists() or (d / "constant" / "polyMesh" / "faces.gz").exists(),
                "times": len(times),
                "latest_time": times[-1] if times else None,
                "processors": len(fields.processor_dirs(d)),
                "running_jobs": running.get(name, []),
            }
        )
    return out


def _field_bcs(path: Path) -> dict:
    d = _read_dict(path)
    if not d:
        return {"error": "could not parse"}
    out: dict = {}
    if "dimensions" in d:
        out["dimensions"] = d["dimensions"]
    out["internalField"] = _truncate(d.get("internalField"), 80)
    bcs = {}
    for patch, spec in (d.get("boundaryField") or {}).items():
        if isinstance(spec, dict):
            parts = [str(spec.get("type", "?"))]
            for k, v in spec.items():
                if k != "type":
                    parts.append(f"{k}={_truncate(v if isinstance(v, str) else '{...}', 60)}")
            bcs[patch] = "; ".join(parts)
        elif spec is None:
            bcs[patch] = "(directive)"
    out["boundaryField"] = bcs
    return out


@tool(READ_ONLY)
def case_summary(
    case: CaseArg,
    include_fields: Annotated[bool, Field(description="Include initial/boundary conditions of every field.")] = True,
) -> dict:
    """One-call overview of a case: solver, run control, mesh & patches, initial/boundary conditions, physics, numerics, results."""
    a = _app()
    d = a.case(case)
    control = _control(d)
    out: dict = {"case": a.name(d), "path": str(d), "solver": _solver_of(control)}
    keys = (
        "startFrom", "startTime", "stopAt", "endTime", "deltaT", "writeControl", "writeInterval",
        "adjustTimeStep", "maxCo", "maxAlphaCo", "writeFormat", "purgeWrite",
    )
    out["controlDict"] = {k: control[k] for k in keys if k in control}
    funcs = control.get("functions")
    if isinstance(funcs, dict):
        out["function_objects"] = {
            k: (dictparse.first_word(v.get("type")) if isinstance(v, dict) else "#include") for k, v in funcs.items()
        }
    fpath = d / "system" / "functions"
    if fpath.is_file():
        out["system/functions"] = _compact(_read_dict(fpath), depth=1)

    size = mesh.mesh_size(d)
    patches = mesh.read_boundary(d)
    regions = sorted(p.parent.name for p in (d / "constant").glob("*/polyMesh") if p.is_dir())
    out["mesh"] = {
        "exists": patches is not None,
        "size": size,
        "patches": patches,
        "regions": regions or None,
    } if (patches is not None or regions) else {"exists": False, "note": "no mesh yet - run the mesh generator (see Allrun)"}

    if include_fields:
        start = next((s for s in ("0", "0.orig") if (d / s).is_dir()), None)
        if start:
            fdir = d / start
            out["initial_conditions"] = {"dir": start}
            for f in sorted(fdir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    out["initial_conditions"][f.name] = _field_bcs(f)

    phys = {}
    for p in sorted((d / "constant").glob("*")):
        if p.is_file() and p.stat().st_size < 200_000 and not p.name.endswith((".stl", ".obj", ".gz", ".eMesh")):
            parsed = _read_dict(p)
            if parsed:
                phys[p.name] = _compact(parsed, depth=2, max_items=15)
    out["constant"] = phys
    sol = _read_dict(d / "system" / "fvSolution")
    out["fvSolution"] = {k: _compact(v, 2) for k, v in sol.items() if k not in ("FoamFile", "solvers")}
    out["fvSolution"]["solvers"] = sorted((sol.get("solvers") or {}).keys())
    sch = _read_dict(d / "system" / "fvSchemes")
    out["fvSchemes"] = {k: _compact(v, 1, 12) for k, v in sch.items() if k != "FoamFile"}

    times = fields.time_dirs(d)
    out["results"] = {
        "time_dirs": len(times),
        "first": times[0] if times else None,
        "latest": times[-1] if times else None,
        "processor_dirs": len(fields.processor_dirs(d)),
        "logs": sorted(p.name for p in d.glob("log.*")),
        "postProcessing": sorted(p.name for p in (d / "postProcessing").glob("*")) if (d / "postProcessing").is_dir() else [],
    }
    scripts = sorted(p.name for p in d.glob("All*") if p.is_file())
    if scripts:
        out["scripts"] = scripts
    jobs = a.jobs.running_for(a.name(d))
    if jobs:
        out["running_jobs"] = [j.public() for j in jobs]
    return out


def _unreconstructed(d: Path, proc0: Path) -> str | None:
    """Describe data in a processor directory that has no reconstructed counterpart, if any."""

    def faces(mesh_dir: Path) -> Path | None:
        return next((p for p in (mesh_dir / "faces", mesh_dir / "faces.gz") if p.is_file()), None)

    pmesh = faces(proc0 / "constant" / "polyMesh")
    cmesh = faces(d / "constant" / "polyMesh")
    if pmesh and (cmesh is None or cmesh.stat().st_mtime < pmesh.stat().st_mtime - 1):
        return "a newer mesh (e.g. from parallel snappyHexMesh)"
    missing = sorted(set(fields.time_dirs(proc0)[1:]) - set(fields.time_dirs(d)), key=float)
    if missing:
        return f"results for times {missing[:5]}{'...' if len(missing) > 5 else ''}"
    return None


@tool(DESTRUCTIVE)
def clean_case(
    case: CaseArg,
    keep_mesh: Annotated[bool, Field(description="Keep constant/polyMesh. Mesh removal is refused when the case cannot regenerate it.")] = True,
    keep_logs: bool = False,
    keep_processor: Annotated[bool, Field(description="Keep processor* (decomposed) directories.")] = False,
    discard_processor_data: Annotated[bool, Field(description="Allow deleting processor* dirs even when they hold a mesh or results that were never reconstructed.")] = False,
) -> dict:
    """Remove results: time directories after the first, postProcessing, logs, processor dirs and optionally the mesh."""
    a = _app()
    d = a.case(case)
    if a.jobs.running_for(a.name(d)):
        raise FoamError("Case has running jobs; stop them first.")
    procs = fields.processor_dirs(d)
    if procs and not keep_processor and not discard_processor_data:
        if problem := _unreconstructed(d, procs[0]):
            raise FoamError(
                f"processor0 holds {problem} that only exists in the decomposed case; deleting it would lose work. "
                "Run reconstructPar first (with -constant for a mesh made in parallel), keep_processor=true, "
                "or pass discard_processor_data=true to delete anyway."
            )
    removed = []
    times = fields.time_dirs(d)
    for t in times[1:]:
        shutil.rmtree(d / t)
        removed.append(t)
    targets = [d / "postProcessing", d / "VTK", d / "dynamicCode"]
    if not keep_processor:
        targets += fields.processor_dirs(d)
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            removed.append(t.name)
    if not keep_logs:
        for p in d.glob("log.*"):
            p.unlink()
            removed.append(p.name)
    if not keep_mesh:
        regenerable = any((d / "system" / n).exists() for n in ("blockMeshDict", "snappyHexMeshDict")) or (
            (d / "Allrun").is_file() and re.search(r"blockMesh|snappyHexMesh|fluentMeshToFoam|gmshToFoam", (d / "Allrun").read_text(errors="replace"))
        )
        if not regenerable:
            raise FoamError("Refusing to delete constant/polyMesh: this case has no mesh generator to rebuild it.")
        pm = d / "constant" / "polyMesh"
        if pm.exists():
            shutil.rmtree(pm)
            removed.append("constant/polyMesh")
    return {"case": a.name(d), "removed": removed}


@tool(DESTRUCTIVE)
def delete_case(
    case: CaseArg,
    confirm: Annotated[bool, Field(description="Must be true to actually delete.")] = False,
) -> dict:
    """Permanently delete a case directory from the workspace."""
    a = _app()
    d = a.case(case, must_exist=False)
    if not within(d, a.settings.workspace.resolve()):
        raise FoamError("Only cases inside the workspace can be deleted.")
    if not d.is_dir():
        raise FoamError(f"No case at {d}.")
    if a.jobs.running_for(a.name(d)):
        raise FoamError("Case has running jobs; stop them first.")
    if not confirm:
        return {"case": a.name(d), "deleted": False, "note": "Call again with confirm=true to delete."}
    shutil.rmtree(d)
    return {"case": a.name(d), "deleted": True}


# ============================================================================ files & dictionaries


@tool(READ_ONLY)
def list_files(
    case: CaseArg,
    subdir: Annotated[str, Field(description="Directory inside the case to list, e.g. 'system' or 'constant/triSurface'.")] = "",
    depth: int = 2,
) -> list[str]:
    """List files in a case (sizes in bytes; long runs of time directories are collapsed)."""
    a = _app()
    d = a.case(case)
    base = case_file(d, subdir) if subdir else d
    if not base.is_dir():
        raise FoamError(f"{subdir} is not a directory.")
    out: list[str] = []
    times = set(fields.time_dirs(d)) if base == d else set()
    shown_times = sorted(times, key=float)
    keep_times = set(shown_times[:2] + shown_times[-2:])

    def walk(p: Path, level: int, prefix: str) -> None:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        collapsed = False
        for e in entries:
            if p == d and e.name in times and e.name not in keep_times:
                if not collapsed:
                    out.append(f"{prefix}... {len(times) - len(keep_times)} more time directories")
                    collapsed = True
                continue
            if e.is_dir():
                out.append(f"{prefix}{e.name}/")
                if level < depth and not e.name.startswith("processor"):
                    walk(e, level + 1, prefix + "  ")
            else:
                out.append(f"{prefix}{e.name}  ({e.stat().st_size})")
            if len(out) > 500:
                out.append("... truncated")
                return

    walk(base, 1, "")
    return out


@tool(READ_ONLY)
def read_file(
    case: CaseArg,
    path: Annotated[str, Field(description="File path relative to the case, e.g. 'system/fvSolution' or '0/U'.")],
    start_line: int = 1,
    max_lines: int = 400,
    elide_lists: Annotated[bool, Field(description="Replace large nonuniform value lists with a placeholder.")] = True,
) -> str:
    """Read a text file from a case (dictionaries, logs, scripts). Large value lists are elided by default."""
    a = _app()
    d = a.case(case)
    p = case_file(d, path)
    if not p.is_file():
        raise FoamError(f"No such file: {path}")
    if p.stat().st_size > 50_000_000:
        raise FoamError("File is larger than 50 MB; use read_log with grep, or field_stats.")
    raw = p.read_bytes()
    if b"\0" in raw[:8000]:
        raise FoamError("Binary file; not shown.")
    text = raw.decode(errors="replace")
    if elide_lists:
        text = dictparse._elide_lists(text)
    lines = text.splitlines()
    chunk = lines[start_line - 1 : start_line - 1 + max_lines]
    footer = ""
    if start_line - 1 + max_lines < len(lines):
        footer = f"\n... ({len(lines)} lines total; continue with start_line={start_line + max_lines})"
    return "\n".join(chunk) + footer


@tool(WRITES)
def write_file(
    case: CaseArg,
    path: Annotated[str, Field(description="File path relative to the case.")],
    content: str,
    append: bool = False,
) -> dict:
    """Create or overwrite a file in a case (new dictionaries, STL-free text inputs, scripts). Prefer set_dict for edits."""
    a = _app()
    d = a.case(case)
    p = case_file(d, path)
    if p.name.startswith("All") and not a.settings.allow_scripts:
        raise FoamError("Writing case scripts is disabled on this server (--no-scripts).")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", newline="\n") as f:
        f.write(content)
    if p.name.startswith("All"):
        p.chmod(p.stat().st_mode | 0o755)
    out: dict = {"written": str(p.relative_to(d)), "bytes": p.stat().st_size}
    if p.parent.name in ("system", "constant", "0") or (p.parent.parent == d and fields._NUM_TIME.match(p.parent.name)):
        if "FoamFile" in content or p.parent.name == "system":
            res = a.foam.run_app("foamDictionary", [str(p.relative_to(d)), "-keywords"], d)
            if res.returncode != 0:
                out["warning"] = "OpenFOAM could not parse this file:\n" + (fatal_error(res.output) or logs_tail_text(res.output, 10))
            else:
                out["keywords"] = res.stdout.split()
    return out


@tool(WRITES)
def import_file(
    case: CaseArg,
    source: Annotated[str, Field(description="File or directory to copy: '$FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz', "
                                 "or a path inside the workspace/allowed roots (Windows paths like C:\\\\models\\\\car.stl are accepted under WSL).")],
    dest: Annotated[str, Field(description="Destination inside the case: a directory ending in '/' (e.g. 'constant/geometry/') or a file path.")],
) -> dict:
    """Copy geometry (STL/OBJ), meshes or other files into a case, e.g. the surface for snappyHexMesh."""
    a = _app()
    d = a.case(case)
    foam = a.foam
    raw = source.strip()
    if m := re.match(r"^\$\{?FOAM_TUTORIALS\}?/?(.*)$", raw):
        src = (foam.tutorials / m.group(1)).resolve()
        allowed = [foam.tutorials.resolve()]
    else:
        src = Path(from_windows(raw)).expanduser().resolve()
        allowed = [*a.settings.roots, foam.tutorials.resolve()]
    if not any(within(src, r) for r in allowed):
        raise FoamError(
            f"{src} is outside the workspace, allowed roots and tutorials. Start the server with --root <dir> to import from there."
        )
    if not src.exists():
        raise FoamError(f"Source not found: {src}")
    target = case_file(d, dest)
    if dest.endswith(("/", "\\")) or target.is_dir():
        target = target / src.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, target)
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.is_dir() else target.stat().st_size
    return {"case": a.name(d), "copied": str(src), "to": str(target.relative_to(d)), "bytes": size}


@tool(READ_ONLY)
def get_dict(
    case: CaseArg,
    file: Annotated[str, Field(description="Dictionary path relative to the case, e.g. 'system/controlDict', '0/U'.")],
    entry: Annotated[str | None, Field(description="Entry path with '/' separators, e.g. 'boundaryField/inlet' or 'SIMPLE/residualControl'. Omit for the whole file.")] = None,
    keywords: Annotated[bool, Field(description="Only list the keywords at that level.")] = False,
    expand: Annotated[bool, Field(description="Expand #include, $macros and #calc first.")] = False,
) -> str:
    """Read a dictionary or one entry through OpenFOAM's foamDictionary (authoritative parsing)."""
    a = _app()
    d = a.case(case)
    p = case_file(d, file)
    if not p.is_file():
        raise FoamError(f"No such file: {file}")
    argv = [str(p.relative_to(d))]
    if entry:
        argv += ["-entry", entry]
    if keywords:
        argv.append("-keywords")
    if expand:
        argv.append("-expand")
    res = a.foam.run_app("foamDictionary", argv, d)
    if res.returncode != 0:
        raise FoamError(fatal_error(res.output) or res.output[-2000:])
    text = dictparse._elide_lists(strip_banner(res.stdout) if not entry else res.stdout.strip())
    if len(text) > 60_000:
        text = text[:60_000] + "\n... truncated; request a narrower entry"
    return text


@tool(WRITES)
def set_dict(
    case: CaseArg,
    file: Annotated[str, Field(description="Dictionary path relative to the case, e.g. 'system/controlDict', '0/U', 'constant/momentumTransport'.")],
    set: Annotated[
        dict[str, str] | None,
        Field(description="Entries to set or add: {'endTime': '2000', 'boundaryField/inlet/value': 'uniform (5 0 0)', "
              "'RAS/model': 'kOmegaSST', 'boundaryField/outlet': '{ type zeroGradient; }'}. Values use OpenFOAM syntax without the trailing ';'."),
    ] = None,
    remove: Annotated[list[str] | None, Field(description="Entry paths to remove.")] = None,
) -> dict:
    """Edit dictionary entries safely via foamDictionary (set/add/remove), then read back the new values."""
    a = _app()
    d = a.case(case)
    p = case_file(d, file)
    if not p.is_file():
        raise FoamError(f"No such file: {file}. Use write_file to create it.")
    rel = str(p.relative_to(d))
    old_mtime = p.stat().st_mtime
    done: dict = {}
    for entry, value in (set or {}).items():
        res = a.foam.run_app("foamDictionary", [rel, "-entry", entry, "-set", str(value).rstrip().rstrip(";")], d)
        if res.returncode != 0:
            raise FoamError(f"Setting '{entry}' failed (earlier entries were applied):\n" + (fatal_error(res.output) or res.output[-1500:]))
        back = a.foam.run_app("foamDictionary", [rel, "-entry", entry], d)
        done[entry] = _truncate(back.stdout.strip(), 400)
    removed = []
    for entry in remove or []:
        res = a.foam.run_app("foamDictionary", [rel, "-entry", entry, "-remove"], d)
        if res.returncode != 0:
            raise FoamError(f"Removing '{entry}' failed:\n" + (fatal_error(res.output) or res.output[-1500:]))
        removed.append(entry)
    out = {"file": rel, "set": done, "removed": removed}
    if _edit_live(d, p, old_mtime):
        out["live"] = "A job is running on this case; the solver will re-read this file at its next time step if runTimeModifiable is on."
    return out


# ============================================================================ running


@tool(WRITES)
def run(
    case: CaseArg,
    app: Annotated[str, Field(description="OpenFOAM application (blockMesh, snappyHexMesh, foamRun, simpleFoam, decomposePar, reconstructPar, setFields, ...) or a case script such as Allrun.")],
    args: Annotated[list[str] | None, Field(description="Command-line arguments, e.g. ['-overwrite'] or ['-dict', 'system/blockMeshDict.fine'].")] = None,
    np: Annotated[int, Field(description="MPI ranks. >1 runs with mpirun and -parallel, decomposing the case first if needed.", ge=1)] = 1,
    wait_seconds: Annotated[float | None, Field(description="How long to wait for completion before returning a job id (default ~45 s). 0 returns immediately.")] = None,
    decompose_method: Annotated[str | None, Field(description="Decomposition method (scotch, hierarchical, simple...) if the case must be decomposed for np>1. Default: the case's decomposeParDict, else scotch.")] = None,
    log: Annotated[str | None, Field(description="Log file name (default log.<app>; the previous one is kept as .1).")] = None,
) -> dict:
    """Run an OpenFOAM application or case script as a background job; waits briefly and reports the outcome or progress."""
    a = _app()
    d = a.case(case)
    args = list(args or [])
    notes = []
    if np > 1 and not app.startswith(("All", "./All")) and app not in ("decomposePar", "reconstructPar", "reconstructParMesh"):
        if note := _ensure_decomposed(d, np, decompose_method, copy_zero=app in _MESHERS):
            notes.append(note)
    others = [j.id for j in a.jobs.running_for(a.name(d))]
    job = _start(d, app, args, np, log)
    job = _wait(job, a.settings.wait_seconds if wait_seconds is None else wait_seconds)
    rep = _job_report(job)
    if app == "checkMesh" and not job.running:
        rep["check"] = mesh.parse_check_mesh(Path(job.log).read_text(errors="replace"))
    if notes:
        rep["notes"] = notes
    if others:
        rep["warning"] = f"Other jobs were already running on this case: {others}"
    return rep


@tool(READ_ONLY)
def job_status(
    job_id: Annotated[str | None, Field(description="Job to inspect. Omit to list running and recent jobs.")] = None,
    wait_seconds: Annotated[float, Field(description="Wait up to this long for the job to finish before reporting.")] = 0,
) -> dict:
    """Status of a job (progress, residuals, errors, log tail) or a list of running and recent jobs."""
    a = _app()
    if job_id:
        job = _wait(a.jobs.get(job_id), wait_seconds)
        return _job_report(job)
    jobs = a.jobs.all()
    for j in jobs:
        _after_finish(j)
    running = [j.public() for j in jobs if j.running]
    recent = [j.public() for j in jobs if not j.running][:15]
    a.jobs.prune()
    return {"running": running, "recent": recent}


@tool(DESTRUCTIVE)
def stop_job(
    job_id: str,
    mode: Annotated[
        Literal["write", "now", "kill"],
        Field(description="'write': solver writes the current time and exits cleanly (controlDict stopAt writeNow); "
              "'now': exit at next step without writing; 'kill': terminate the process group immediately."),
    ] = "write",
) -> dict:
    """Stop a running job, gracefully (results written) or forcibly."""
    a = _app()
    job = a.jobs.get(job_id)
    if not job.running:
        return {"job": job.public(), "note": "Job is not running."}
    if mode in ("write", "now"):
        d = a.case(job.case)
        log = Path(job.log)
        if not (log.is_file() and a.logs.get(log).times):
            mode = "kill"
        else:
            control = d / "system" / "controlDict"
            old_mtime = control.stat().st_mtime
            prev = a.foam.run_app("foamDictionary", ["system/controlDict", "-entry", "stopAt", "-value"], d)
            original = prev.stdout.strip() or "endTime"
            if original in ("writeNow", "noWriteNow"):  # a previous stop is still pending
                original = job.restore.get("stopAt", "endTime")
            target = "writeNow" if mode == "write" else "noWriteNow"
            res = a.foam.run_app("foamDictionary", ["system/controlDict", "-entry", "stopAt", "-set", target], d)
            if res.returncode != 0:
                raise FoamError("Could not edit controlDict: " + res.output[-800:])
            a.jobs.mark_stop(job, mode, {"stopAt": original})
            _edit_live(d, control, old_mtime)
            job = _wait(job, 30)
            rep = {"job": job.public()}
            rep["note"] = (
                f"stopAt set to {target}; the solver stops at its next time step (requires runTimeModifiable yes). "
                f"stopAt will be restored to '{original}' when the job ends."
                if job.running
                else "Stopped cleanly."
            )
            return rep
    a.jobs.mark_stop(job, "kill")
    a.jobs.kill(job, signal.SIGTERM, a.foam)
    job = _wait(job, 5)
    if job.running:
        a.jobs.kill(job, signal.SIGKILL, a.foam)
        job = _wait(job, 3)
    return {"job": job.public()}


@tool(READ_ONLY)
def read_log(
    case: CaseArg,
    log: Annotated[str | None, Field(description="Log file, e.g. 'log.foamRun' or just 'foamRun'. Default: most recently modified log.")] = None,
    tail: Annotated[int, Field(description="Number of trailing lines to return (ignored with grep).")] = 80,
    grep: Annotated[str | None, Field(description="Regex; return matching lines (with line numbers) instead of the tail.")] = None,
) -> str:
    """Read the tail of a log, or grep it (e.g. 'FOAM Warning|bounding|Courant')."""
    a = _app()
    d = a.case(case)
    if log:
        name = log if log.startswith("log.") else f"log.{log}"
        p = case_file(d, name)
    else:
        candidates = sorted(d.glob("log.*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            raise FoamError("No log.* files in this case.")
        p = candidates[0]
    if not p.is_file():
        raise FoamError(f"No such log: {p.name}. Logs: {sorted(x.name for x in d.glob('log.*'))}")
    if grep:
        rx = re.compile(grep)
        hits = []
        with open(p, errors="replace") as f:
            for i, line in enumerate(f, 1):
                if rx.search(line):
                    hits.append(f"{i}: {line.rstrip()}")
                    if len(hits) >= 300:
                        hits.append("... more matches truncated")
                        break
        return f"[{p.name}]\n" + ("\n".join(hits) or "(no matches)")
    return f"[{p.name}]\n" + logs.tail(p, tail)


@tool(READ_ONLY)
def solver_progress(
    case: CaseArg,
    log: Annotated[str | None, Field(description="Solver log; default is the most recent log containing time steps.")] = None,
    plot: Annotated[bool, Field(description="Also return a residual (and Courant) plot image.")] = False,
):
    """Convergence report from a solver log: state (converged/diverging/failed...), residuals and trends, Courant, continuity, ETA."""
    a = _app()
    d = a.case(case)
    if log:
        p = case_file(d, log if log.startswith("log.") else f"log.{log}")
    else:
        p = logs.find_solver_log(d)
        if p is None:
            raise FoamError("No solver log with time steps found in this case.")
    data = a.logs.get(p)
    rep = {"case": a.name(d), "log": p.name, **logs.summarize(data, _end_time(d))}
    jobs = [j for j in a.jobs.running_for(a.name(d)) if Path(j.log) == p]
    rep["running"] = bool(jobs)
    if jobs:
        rep["job_id"] = jobs[0].id
    elif rep["state"] == "in_progress":
        rep["state"] = "stopped"
    if not plot:
        return rep
    cour = None
    if data.courant:
        idx = [c[0] for c in data.courant]
        tt = [data.times[i] if 0 <= i < len(data.times) else i for i in idx]
        cour = (tt, [c[1] for c in data.courant], [c[2] for c in data.courant])
    png = plotting.residuals(logs.series(data), f"{a.name(d)} - {p.name}", cour)
    return [rep, Image(data=png, format="png")]


# ============================================================================ mesh


@tool(READ_ONLY)
def check_mesh(
    case: CaseArg,
    run: Annotated[bool, Field(description="Run checkMesh now (false: parse the existing log.checkMesh).")] = True,
    args: Annotated[list[str] | None, Field(description="Extra checkMesh options, e.g. ['-allGeometry', '-allTopology'] or ['-latestTime'].")] = None,
    wait_seconds: float = 120,
) -> dict:
    """Run checkMesh and return structured mesh statistics, failed checks and practical advice."""
    a = _app()
    d = a.case(case)
    log = d / "log.checkMesh"
    if run:
        # A decomposed case keeps its current mesh in processor*/; check that one, in parallel.
        job = _wait(_start(d, "checkMesh", list(args or []), np=len(fields.processor_dirs(d)) or 1), wait_seconds)
        if job.running:
            return {"job": job.public(), "hint": f"checkMesh still running; call check_mesh('{case}', run=false) after it finishes."}
    if not log.is_file():
        raise FoamError("No log.checkMesh yet; call with run=true.")
    text = log.read_text(errors="replace")
    if err := fatal_error(text):
        raise FoamError("checkMesh failed:\n" + err)
    return {"case": a.name(d), **mesh.parse_check_mesh(text)}


# ============================================================================ post-processing


@tool(WRITES)
def post_process(
    case: CaseArg,
    func: Annotated[str, Field(description="Function, e.g. 'mag(U)', 'vorticity', 'yPlus', 'wallShearStress', \"patchAverage(p, patch=outlet)\", 'Q'. List options with foam_reference('functions').")],
    time: Annotated[str, Field(description="'latest', 'all', or a time value / range like '100:200'.")] = "latest",
    with_solver: Annotated[bool, Field(description="Construct the solver's physical models (needed for yPlus, wallShearStress, forces, turbulenceFields).")] = False,
    args: list[str] | None = None,
    wait_seconds: float = 120,
) -> dict:
    """Run a post-processing function object on saved results (writes fields and/or postProcessing/ data)."""
    a = _app()
    d = a.case(case)
    f = a.foam
    argv = ["-func", func]
    if time == "latest":
        argv.append("-latestTime")
    elif time != "all":
        argv += ["-time", time]
    solver = _solver_of(_control(d))
    if with_solver:
        if f.uses_foam_run:
            app_name = f.post_process_app
            argv += ["-solver", solver or ""]
        else:
            app_name = solver or f.post_process_app
            argv = ["-postProcess", *argv]
    else:
        app_name = f.post_process_app
    # Results of a decomposed run that was not reconstructed live in processor*/: process them in parallel.
    procs = fields.processor_dirs(d)
    np = len(procs) if procs and not fields.time_dirs(d)[1:] else 1
    watched = [d / "postProcessing", *(d / t for t in fields.time_dirs(d)), *(procs[:1] if np > 1 else [])]

    def snapshot() -> dict[Path, float]:
        return {p: p.stat().st_mtime for w in watched if w.is_dir() for p in w.rglob("*") if p.is_file()}

    before = snapshot()
    job = _wait(_start(d, app_name, [*argv, *(args or [])], np=np, log_name=f"log.{f.post_process_app}"), wait_seconds)
    rep = _job_report(job, tail_lines=30)
    if not job.running:
        changed =[p for p, m in snapshot().items() if before.get(p) != m]
        rep["postProcessing_files"] = sorted(str(p.relative_to(d)) for p in changed if "postProcessing" in p.parts)[:50]
        rep["fields_written"] = sorted({f"{p.parent.name}/{p.name}" for p in changed if "postProcessing" not in p.parts})[:50]
    return rep


@tool(READ_ONLY)
def list_postprocessing(case: CaseArg) -> list[dict]:
    """List function-object output files under postProcessing/ with their column names."""
    a = _app()
    return postproc.list_outputs(a.case(case))


@tool(READ_ONLY)
def read_postprocessing(
    case: CaseArg,
    path: Annotated[str, Field(description="File path from list_postprocessing, e.g. 'postProcessing/forceCoeffs/0/forceCoeffs.dat'.")],
    columns: Annotated[list[str] | None, Field(description="Columns to return/plot (default: all).")] = None,
    max_rows: Annotated[int, Field(description="Rows to include (evenly sampled). 0 returns statistics only.")] = 30,
    plot: Annotated[bool, Field(description="Also return a line plot of the selected columns.")] = False,
):
    """Read a postProcessing table (forces, coefficients, probes, sampled lines...) with statistics and optional plot."""
    a = _app()
    d = a.case(case)
    p = case_file(d, path)
    if not p.is_file():
        raise FoamError(f"No such file: {path}")
    names, arr, labels = postproc.load_table(p)
    try:
        rep = postproc.summarize_table(names, arr, columns, max_rows, labels)
    except KeyError as e:
        raise FoamError(str(e.args[0])) from None
    rep = {"file": str(p.relative_to(d)), **rep}
    if not plot or arr.shape[0] == 0:
        return rep
    cols = columns or [n for i, n in enumerate(names) if i and i not in labels][:6]
    png = plotting.table(names, arr, cols, f"{a.name(d)}: {p.relative_to(d)}")
    return [rep, Image(data=png, format="png")]


@tool(READ_ONLY)
def field_stats(
    case: CaseArg,
    field: Annotated[str, Field(description="Field name, e.g. 'U', 'p', 'k', 'alpha.water'.")],
    time: Annotated[str, Field(description="'latest', 'first' or a time directory name.")] = "latest",
    region: str | None = None,
) -> dict:
    """Min/max/mean of a field (internal and per patch) at a time, read directly from ASCII field files."""
    a = _app()
    d = a.case(case)
    times = fields.time_dirs(d)
    if not times:
        raise FoamError("No time directories.")
    t = times[-1] if time == "latest" else times[0] if time == "first" else time
    p = fields.field_path(d, t, field, region)
    if p is None:
        avail = sorted(x.name for x in (d / t).iterdir() if x.is_file()) if (d / t).is_dir() else []
        raise FoamError(f"Field '{field}' not found at time {t}. Available: {avail}")
    return {
        "case": a.name(d),
        "time": t,
        "field": field,
        "format": "[min, max, mean]; means are arithmetic over cells/faces, not volume/area-weighted",
        **fields.field_stats(p),
    }


@tool(READ_ONLY)
def render(
    case: CaseArg,
    field: Annotated[str | None, Field(description="Field to colour by (e.g. 'U', 'p'). Omit with mode='mesh'.")] = None,
    mode: Annotated[
        Literal["auto", "slice", "surface", "patches", "mesh", "contour", "streamlines"],
        Field(description="auto: whole domain for 2-D cases, mid-plane slice for 3-D. patches: colour boundary patches "
              "(e.g. pressure on a body). contour: iso-surface of contour_field at iso_value. streamlines: of vector `field`."),
    ] = "auto",
    time: Annotated[str, Field(description="'latest', 'first' or a time value.")] = "latest",
    component: Annotated[Literal["Magnitude", "X", "Y", "Z"] | None, Field(description="Vector component (default Magnitude).")] = None,
    slice_normal: Annotated[list[float] | str | None, Field(description="Slice normal, e.g. 'y' or [0,1,0].")] = None,
    slice_origin: Annotated[list[float] | None, Field(description="Slice origin (default: domain centre).")] = None,
    patches: Annotated[list[str] | None, Field(description="Patch/group names for mode='patches'.")] = None,
    view: Annotated[str | list[float], Field(description="Camera: 'auto', 'iso', '+x', '-x', '+y', '-y', '+z', '-z' (camera on that side, looking back), or a direction [dx, dy, dz] from the target to the camera, e.g. [-1, -1, 0.8] to look at the upstream side.")] = "auto",
    focus: Annotated[str | None, Field(description="Patch or group to frame (e.g. 'motorBikeGroup', 'wall'); slices pass through it by default.")] = None,
    zoom: Annotated[float | None, Field(description="Zoom factor >1 to magnify (centre of domain).")] = None,
    range: Annotated[list[float] | None, Field(description="[min, max] colour range.")] = None,
    colormap: Annotated[str, Field(description="coolwarm, viridis, jet, turbo, inferno, plasma, rainbow, blue-red, grayscale.")] = "coolwarm",
    edges: Annotated[bool, Field(description="Draw mesh edges.")] = False,
    iso_value: float | None = None,
    contour_field: str | None = None,
    seed_point1: list[float] | None = None,
    seed_point2: list[float] | None = None,
    width: int = 1400,
    height: int = 800,
    background: Literal["light", "dark"] = "light",
):
    """Render a picture of the flow or mesh with ParaView (headless) and return it as an image plus metadata."""
    a = _app()
    d = a.case(case)
    foam_file = d / f"{d.name}.foam"
    if not foam_file.exists():
        foam_file.touch()
    decomposed = not fields.time_dirs(d)[1:] and bool(fields.processor_dirs(d))
    images = d / "images"
    images.mkdir(exist_ok=True)
    stem = f"{field or 'mesh'}_{mode}_{time}".replace("/", "_").replace(" ", "")
    out = images / f"{stem}_{int(_time.time())}.png"
    params = {
        "foam_file": str(foam_file),
        "case_name": a.name(d),
        "out": str(out),
        "field": field,
        "mode": mode,
        "time": time,
        "component": component,
        "slice_normal": slice_normal,
        "slice_origin": slice_origin,
        "patches": patches,
        "view": view,
        "focus": focus,
        "zoom": zoom,
        "range": range,
        "colormap": colormap,
        "edges": edges,
        "iso_value": iso_value,
        "contour_field": contour_field,
        "seed_point1": seed_point1,
        "seed_point2": seed_point2,
        "width": width,
        "height": height,
        "background": background,
        "decomposed": decomposed,
    }
    meta = a.paraview.render(params)
    # Keep only the 20 newest images per case.
    olds = sorted(images.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[20:]
    for o in olds:
        o.unlink(missing_ok=True)
    meta["image"] = str(out)
    return [meta, Image(path=out)]

