"""OpenFOAM installation discovery, environment sourcing and command execution.

Three ways of reaching OpenFOAM are supported, all behind the same `FoamEnv` interface:

* **native** - an `etc/bashrc` on this machine (Linux packages, source builds on Linux/macOS),
  or an already-sourced environment;
* **launcher** - an `openfoam` launcher script that sources the environment itself
  (OpenFOAM.app on macOS via Homebrew, openfoam.com packages' `openfoamXXXX` commands);
* **container** - a Docker/Podman container with the workspace bind-mounted at the same path,
  the usual route for openfoam.org releases on macOS (and handy on any OS).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .errors import FoamError

_BASHRC_GLOBS = [
    "/opt/openfoam*/etc/bashrc",  # openfoam.org Debian/Ubuntu packages
    "/opt/OpenFOAM-*/etc/bashrc",
    "/opt/OpenFOAM/OpenFOAM-*/etc/bashrc",
    "/usr/lib/openfoam/openfoam*/etc/bashrc",  # openfoam.com (ESI) packages
    "/usr/share/openfoam/etc/bashrc",
    "~/OpenFOAM/OpenFOAM-*/etc/bashrc",  # source builds (Linux and macOS)
    "~/openfoam/OpenFOAM-*/etc/bashrc",
]

_LAUNCHER_GLOBS = [
    "/opt/homebrew/bin/openfoam",  # OpenFOAM.app via Homebrew (Apple silicon)
    "/usr/local/bin/openfoam",
    "/Applications/OpenFOAM*.app/Contents/Resources/etc/openfoam",
    "~/Applications/OpenFOAM*.app/Contents/Resources/etc/openfoam",
    "/usr/bin/openfoam[0-9]*",  # openfoam.com Linux packages
]

# Directories GUI-launched MCP clients often leave off PATH (notably on macOS).
_EXTRA_PATH = ["/opt/homebrew/bin", "/usr/local/bin", "/Applications/Docker.app/Contents/Resources/bin", "/usr/bin", "/bin"]

_SCRIPT_NAME = re.compile(r"^All[A-Za-z0-9_.-]*$")
_ENV_DUMP = "import json, os; print('OFMCP_ENV' + json.dumps(dict(os.environ)))"


def _version_key(path: str) -> tuple:
    nums = re.findall(r"\d+", path)
    return tuple(int(n) for n in nums) if nums else (0,)


def _expand(patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.expanduser(pattern)))
    return sorted(set(found), key=_version_key, reverse=True)  # newest version first


def find_bashrc() -> list[str]:
    return _expand(_BASHRC_GLOBS)


def find_launchers() -> list[str]:
    found = _expand(_LAUNCHER_GLOBS)
    on_path = shutil.which("openfoam")
    if on_path and on_path not in found:
        found.insert(0, on_path)
    return [f for f in found if os.access(f, os.X_OK)]


def which(name: str, path: str | None = None) -> str | None:
    search = os.pathsep.join(filter(None, [path or os.environ.get("PATH", ""), *_EXTRA_PATH]))
    return shutil.which(name, path=search)


def _parse_env_dump(text: str, origin: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("OFMCP_ENV"):
            env = json.loads(line[len("OFMCP_ENV") :])
            if "WM_PROJECT_DIR" not in env:
                break
            return env
    raise FoamError(f"{origin} did not produce an OpenFOAM environment (WM_PROJECT_DIR unset).")


def source_bashrc(bashrc: str) -> dict[str, str]:
    """Environment produced by sourcing an OpenFOAM bashrc (portable: no GNU `env -0` needed)."""
    proc = subprocess.run(
        ["bash", "-c", 'source "$1" >/dev/null 2>&1; exec "$2" -c "$3"', "_", bashrc, sys.executable, _ENV_DUMP],
        capture_output=True, text=True, timeout=120,
    )
    return _parse_env_dump(proc.stdout, f"Sourcing {bashrc}")


def launcher_env(launcher: str) -> dict[str, str]:
    """Environment from an openfoam.com-style launcher: `openfoam -c '<commands>'`."""
    cmd = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(_ENV_DUMP)}"
    try:
        proc = subprocess.run([launcher, "-c", cmd], capture_output=True, text=True, timeout=300)
    except OSError as e:
        raise FoamError(f"Cannot run launcher {launcher}: {e}") from e
    return _parse_env_dump(proc.stdout, f"Launcher {launcher}")


@dataclass
class Completed:
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr.strip() else "")).strip()


# ============================================================================ runners


class LocalRunner:
    """OpenFOAM processes run directly on this machine with the sourced environment."""

    kind = "native"

    def __init__(self, env: dict[str, str]):
        self.env = env

    @property
    def host_env(self) -> dict[str, str]:
        return self.env

    def wrap(self, argv: list[str], cwd: Path, pidfile: str | None = None) -> list[str]:
        return argv

    def list_apps(self, dirs: list[str]) -> dict[str, str]:
        apps: dict[str, str] = {}
        for d in reversed([Path(x) for x in dirs if x and Path(x).is_dir()]):  # earlier dirs override later ones
            for p in d.iterdir():
                if p.is_file() and os.access(p, os.X_OK):
                    apps[p.name] = str(p)
        return apps

    def which(self, name: str) -> str | None:
        return which(name, self.env.get("PATH"))

    def kill_job(self, pidfile: str | None, sig: str) -> None:
        """Nothing extra: killing the local process group is enough."""

    def describe(self) -> dict:
        return {"backend": self.kind}


class ContainerRunner:
    """OpenFOAM processes run inside a long-lived Docker/Podman container via `exec`.

    The workspace (and any extra roots) are bind-mounted at identical paths, so every path the
    server hands to OpenFOAM is valid on both sides and file tools work on the host unchanged.
    """

    kind = "container"
    _EXEC = 'rc=$1; pf=$2; shift 2; source "$rc" >/dev/null 2>&1; [ -n "$pf" ] && echo $$ > "$pf"; exec "$@"'

    def __init__(self, cli: str, container: str, bashrc: str, image: str | None = None):
        self.cli = cli
        self.container = container
        self.bashrc = bashrc
        self.image = image
        self.env: dict[str, str] = {}

    @property
    def host_env(self) -> dict[str, str]:
        return dict(os.environ)

    def wrap(self, argv: list[str], cwd: Path, pidfile: str | None = None) -> list[str]:
        return [self.cli, "exec", "-i", "-w", str(cwd), self.container, "bash", "-c", self._EXEC, "ofmcp",
                self.bashrc, pidfile or "", *argv]

    def exec_sh(self, script: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run([self.cli, "exec", self.container, "bash", "-c", script, "ofmcp", *args],
                              capture_output=True, text=True, errors="replace", timeout=timeout)

    def load_env(self) -> dict[str, str]:
        p = self.exec_sh('source "$1" >/dev/null 2>&1; env -0', self.bashrc)
        env = {}
        for item in p.stdout.split("\0"):
            if "=" in item:
                k, _, v = item.partition("=")
                env[k.strip("\n")] = v
        if "WM_PROJECT_DIR" not in env:
            raise FoamError(f"Sourcing {self.bashrc} inside container '{self.container}' failed: {p.stderr[-500:]}")
        self.env = env
        return env

    def list_apps(self, dirs: list[str]) -> dict[str, str]:
        p = self.exec_sh('for d in "$@"; do [ -d "$d" ] && find "$d" -maxdepth 1 -type f -perm -u+x; done', *dirs)
        apps: dict[str, str] = {}
        for line in reversed(p.stdout.splitlines()):
            if line.strip():
                apps.setdefault(Path(line).name, line.strip())
        return apps

    def which(self, name: str) -> str | None:
        p = self.exec_sh('source "$1" >/dev/null 2>&1; command -v "$2"', self.bashrc, name)
        return p.stdout.strip() or None

    def kill_job(self, pidfile: str | None, sig: str) -> None:
        """Signal the process (group) inside the container; killing the host `exec` client is not enough."""
        if pidfile:
            self.exec_sh(self._KILL_TREE, pidfile, sig, timeout=30)

    # Signal a process and all its descendants, found via /proc (no procps needed in the image).
    _KILL_TREE = r"""
p=$(cat "$1" 2>/dev/null) || exit 0
kids() {
  for s in /proc/[0-9]*/stat; do
    read -r line < "$s" 2>/dev/null || continue
    rest=${line##*) }; set -- $rest
    [ "$2" = "$P" ] && echo "${s//[^0-9]/}"
  done
}
all=$p; frontier=$p
while [ -n "$frontier" ]; do
  next=""
  for P in $frontier; do next="$next $(kids)"; done
  frontier=$(echo $next); all="$all $frontier"
done
kill -"$2" $all 2>/dev/null; true
"""

    def mirror(self, src: str, cache_root: Path) -> Path:
        """Copy a directory out of the container once (used for the tutorials tree)."""
        key = hashlib.sha1(f"{self.container}|{self.image}|{src}".encode()).hexdigest()[:12]
        dest = cache_root / key
        if not dest.is_dir():
            tmp = cache_root / f".{key}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            p = subprocess.run([self.cli, "cp", f"{self.container}:{src}", str(tmp)], capture_output=True, text=True)
            if p.returncode != 0:
                raise FoamError(f"Could not copy {src} out of the container: {p.stderr.strip()}")
            os.replace(tmp, dest)
        return dest

    def describe(self) -> dict:
        return {"backend": self.kind, "cli": self.cli, "container": self.container, "image": self.image}


def container_name(image: str, mounts: list[Path]) -> str:
    digest = hashlib.sha1("|".join([image, *map(str, mounts)]).encode()).hexdigest()[:10]
    return f"openfoam-mcp-{digest}"


def start_container(cli: str, container: str | None, image: str | None, mounts: list[Path]) -> tuple[str, str | None]:
    """Make sure a usable container is running. Returns (container name, image)."""
    name = container or container_name(image or "", mounts)
    state = subprocess.run([cli, "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
    if state.returncode != 0:
        if not image:
            raise FoamError(f"Container '{name}' does not exist.")
        args = [cli, "run", "-d", "--name", name, "--init", "--entrypoint", "sleep", "-e", "HOME=/tmp"]
        if Path(cli).name.startswith("podman"):
            args.append("--userns=keep-id")
        elif hasattr(os, "getuid"):
            args += ["-u", f"{os.getuid()}:{os.getgid()}"]  # results stay owned by the host user
        for m in mounts:
            m.mkdir(parents=True, exist_ok=True)
            args += ["-v", f"{m}:{m}"]
        p = subprocess.run([*args, image, "infinity"], capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise FoamError(f"Could not start a container from {image}: {p.stderr.strip()[-800:]}")
    elif state.stdout.strip() != "true":
        p = subprocess.run([cli, "start", name], capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise FoamError(f"Could not start container '{name}': {p.stderr.strip()}")
    return name, image


# ============================================================================ environment


@dataclass
class FoamEnv:
    env: dict[str, str]
    origin: str | None = None  # bashrc / launcher / container used
    runner: LocalRunner | ContainerRunner | None = None
    _apps: dict[str, str] | None = field(default=None, repr=False)
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "openfoam-mcp")

    def __post_init__(self):
        if self.runner is None:
            self.runner = LocalRunner(self.env)

    @classmethod
    def detect(
        cls,
        bashrc: str | None = None,
        launcher: str | None = None,
        container: str | None = None,
        image: str | None = None,
        container_cli: str | None = None,
        mounts: list[Path] | None = None,
    ) -> FoamEnv:
        if container or image:
            return cls._from_container(container, image, container_cli, mounts or [])
        if launcher:
            return cls(launcher_env(os.path.expanduser(launcher)), f"launcher {launcher}")
        if bashrc:
            return cls(source_bashrc(os.path.expanduser(bashrc)), bashrc)
        if os.environ.get("WM_PROJECT_DIR") and os.environ.get("FOAM_APPBIN"):
            return cls(dict(os.environ), "inherited environment")
        if candidates := find_bashrc():
            return cls(source_bashrc(candidates[0]), candidates[0])
        errors = []
        for cand in find_launchers():
            try:
                return cls(launcher_env(cand), f"launcher {cand}")
            except FoamError as e:
                errors.append(str(e))
        raise FoamError(_not_found_hint() + (f" ({'; '.join(errors)})" if errors else ""))

    @classmethod
    def _from_container(cls, container, image, cli, mounts) -> FoamEnv:
        cli = cli or which("docker") or which("podman")
        if not cli:
            raise FoamError("Container mode needs docker or podman on PATH (or pass --container-cli).")
        name, image = start_container(cli, container, image, mounts)
        probe = ContainerRunner(cli, name, "", image)
        listed = probe.exec_sh(
            "for f in " + " ".join(_BASHRC_GLOBS).replace("~", "$HOME") + '; do [ -f "$f" ] && echo "$f"; done; true'
        ).stdout.split()
        if not listed:
            raise FoamError(f"No OpenFOAM etc/bashrc found inside container '{name}'.")
        runner = ContainerRunner(cli, name, sorted(listed, key=_version_key, reverse=True)[0], image)
        env = runner.load_env()
        for m in mounts:  # the design relies on identical paths on both sides
            m.mkdir(parents=True, exist_ok=True)
            marker = m / f".ofmcp-mount-{os.getpid()}"
            marker.write_text("x")
            try:
                ok = runner.exec_sh('[ -f "$1" ]', str(marker)).returncode == 0
            finally:
                marker.unlink(missing_ok=True)
            if not ok:
                raise FoamError(f"Container '{name}' must bind-mount {m} at the same path (-v {m}:{m}).")
        return cls(env, f"container {name}", runner)

    # ---- identity -------------------------------------------------------
    @property
    def version(self) -> str:
        return self.env.get("WM_PROJECT_VERSION", "?")

    @property
    def project_dir(self) -> Path:
        return Path(self.env["WM_PROJECT_DIR"])

    @property
    def bashrc(self) -> str | None:
        return self.origin

    @cached_property
    def flavor(self) -> str:
        """'org' for openfoam.org (Foundation) builds, 'com' for openfoam.com (ESI) builds."""
        if self.env.get("FOAM_API") or self.version.startswith("v"):
            return "com"
        return "org"

    @cached_property
    def major(self) -> int | None:
        m = re.match(r"v?(\d+)", self.version)
        return int(m.group(1)) if m else None

    @property
    def uses_foam_run(self) -> bool:
        """openfoam.org >= 11 runs modular solvers through foamRun (`solver` entry in controlDict)."""
        return self.flavor == "org" and (self.major or 0) >= 11

    @property
    def post_process_app(self) -> str:
        return "foamPostProcess" if self.flavor == "org" and (self.major or 0) >= 9 else "postProcess"

    @cached_property
    def tutorials(self) -> Path:
        src = self.env.get("FOAM_TUTORIALS", str(self.project_dir / "tutorials"))
        if isinstance(self.runner, ContainerRunner):
            return self.runner.mirror(src, self.cache_dir / "tutorials")
        return Path(src)

    @property
    def host_env(self) -> dict[str, str]:
        return self.runner.host_env

    # ---- applications ---------------------------------------------------
    @property
    def app_dirs(self) -> list[str]:
        keys = ("FOAM_USER_APPBIN", "FOAM_SITE_APPBIN", "FOAM_APPBIN")
        return [*(self.env[k] for k in keys if self.env.get(k)), str(self.project_dir / "bin")]

    @property
    def apps(self) -> dict[str, str]:
        if self._apps is None:
            self._apps = self.runner.list_apps(self.app_dirs)
        return self._apps

    def resolve_app(self, name: str, case_dir: Path | None = None, allow_scripts: bool = True) -> list[str]:
        """Map an application name to an argv prefix, enforcing the allow-list.

        Allowed: OpenFOAM applications and scripts from the installation's bin directories, and
        case-local `All*` scripts (Allrun, Allclean, ...) when scripts are enabled.
        """
        name = name.strip()
        script = name.removeprefix("./")
        if case_dir is not None and _SCRIPT_NAME.match(script) and (case_dir / script).is_file():
            if not allow_scripts:
                raise FoamError("Running case scripts is disabled on this server (--no-scripts).")
            return ["bash", str(case_dir / script)]
        if "/" in name or name not in self.apps:
            close = [a for a in self.apps if len(name) >= 4 and name.lower() in a.lower()][:10]
            hint = f" Similar: {', '.join(close)}." if close else ""
            raise FoamError(f"'{name}' is not an OpenFOAM application or case script.{hint}")
        return [self.apps[name]]

    @cached_property
    def mpirun(self) -> str | None:
        return self.runner.which("mpirun")

    # ---- execution ------------------------------------------------------
    def wrap(self, argv: list[str], cwd: Path, pidfile: str | None = None) -> list[str]:
        return self.runner.wrap(argv, cwd, pidfile)

    def run(self, argv: list[str], cwd: Path | None = None, timeout: float = 120) -> Completed:
        cwd = cwd or Path.cwd()
        try:
            p = subprocess.run(
                self.wrap(argv, cwd), cwd=cwd, env=self.host_env,
                capture_output=True, text=True, errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise FoamError(f"{Path(argv[0]).name} timed out after {timeout:.0f}s") from e
        except FileNotFoundError as e:
            raise FoamError(f"Command not found: {argv[0]}") from e
        return Completed(p.returncode, p.stdout, p.stderr)

    def run_app(self, app: str, args: list[str], cwd: Path, timeout: float = 120) -> Completed:
        return self.run([*self.resolve_app(app), *args], cwd=cwd, timeout=timeout)


def _not_found_hint() -> str:
    if sys.platform == "darwin":
        return (
            "OpenFOAM not found. On macOS either install the native OpenFOAM.app "
            "(`brew install gerlero/openfoam/openfoam`, openfoam.com builds) or run OpenFOAM in Docker "
            "(e.g. --container-image opencfd/openfoam-default, or an openfoam.org image). "
            "Source builds are found under ~/OpenFOAM; otherwise pass --bashrc or --launcher."
        )
    return (
        "OpenFOAM not found. Install it (openfoam.org: Ubuntu packages under /opt/openfoamNN; "
        "openfoam.com: packages under /usr/lib/openfoam), source its etc/bashrc before starting the server, "
        "or pass --bashrc /path/to/etc/bashrc, --launcher, or --container-image."
    )


def strip_banner(text: str) -> str:
    """Remove the OpenFOAM header banner and the `// * * *` separator from tool output."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("// * * *"):
            return "\n".join(lines[i + 1 :]).strip()
    return text.strip()


def fatal_error(text: str) -> str | None:
    """Extract the `--> FOAM FATAL ERROR` block from OpenFOAM output, if any."""
    m = re.search(r"--> FOAM FATAL (?:IO )?ERROR.*?(?=\n\s*FOAM (?:exiting|aborting)|\Z)", text, re.S)
    return m.group(0).strip() if m else None
