"""Background job management that survives MCP server restarts.

Every job runs detached in its own process group under a small bash wrapper that records
the exit code to disk. Job metadata lives in `<workspace>/.openfoam-mcp/jobs/<id>.json`,
so a restarted server (or a different MCP client) can still see, monitor and stop it.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import FoamError

if TYPE_CHECKING:
    from .foam import FoamEnv

_WRAPPER = '"$@" > "$OFMCP_LOG" 2>&1 < /dev/null; echo $? > "$OFMCP_RC"'


@dataclass
class Job:
    id: str
    case: str
    app: str
    argv: list[str]
    log: str
    pid: int
    started: float
    np: int = 1
    proc_start: str | None = None  # /proc start ticks, guards against PID reuse
    finished: float | None = None
    returncode: int | None = None
    stop_requested: str | None = None
    restore: dict = field(default_factory=dict)  # controlDict entries to restore after a graceful stop
    pidfile: str | None = None  # in-container PID, for jobs run through a container backend

    @property
    def running(self) -> bool:
        return self.returncode is None

    def public(self) -> dict:
        d = {
            "job_id": self.id,
            "case": self.case,
            "app": self.app,
            "command": " ".join(Path(self.argv[0]).name if i == 0 else a for i, a in enumerate(self.argv)),
            "np": self.np,
            "log": Path(self.log).name,
            "state": "running" if self.running else ("succeeded" if self.returncode == 0 else "failed"),
            "elapsed_s": round((self.finished or time.time()) - self.started, 1),
        }
        if self.returncode is not None:
            d["returncode"] = self.returncode
        if self.stop_requested:
            d["stop_requested"] = self.stop_requested
            if not self.running:
                d["state"] = "stopped"
        return d


def _proc_start(pid: int) -> str | None:
    """A process's start time, used as identity to guard against PID reuse; None if not running.

    Linux reads /proc; macOS and other Unixes fall back to `ps -o lstart=`.
    """
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc/self/stat").exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or None


class JobManager:
    def __init__(self, state_dir: Path):
        self.dir = state_dir / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    # ---- persistence ----------------------------------------------------
    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _save(self, job: Job) -> None:
        tmp = self._path(job.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), indent=1))
        os.replace(tmp, self._path(job.id))

    def _load(self, job_id: str) -> Job:
        try:
            return Job(**json.loads(self._path(job_id).read_text()))
        except FileNotFoundError:
            raise FoamError(f"No job with id '{job_id}'. Use job_status() to list jobs.") from None

    # ---- lifecycle ------------------------------------------------------
    def start(self, case_dir: Path, case_name: str, app: str, argv: list[str], log: Path, foam: FoamEnv, np: int = 1) -> Job:
        job_id = f"{app.replace('/', '_').replace('.', '')[:24]}-{secrets.token_hex(3)}"
        rc_file = self.dir / f"{job_id}.rc"
        pidfile = str(self.dir / f"{job_id}.pid") if foam.runner.kind == "container" else None
        wenv = dict(foam.host_env, OFMCP_LOG=str(log), OFMCP_RC=str(rc_file))
        proc = subprocess.Popen(
            ["bash", "-c", _WRAPPER, "ofmcp", *foam.wrap(argv, case_dir, pidfile)],
            cwd=case_dir,
            env=wenv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._procs[job_id] = proc
        job = Job(
            id=job_id,
            case=case_name,
            app=app,
            argv=argv,
            log=str(log),
            pid=proc.pid,
            started=time.time(),
            np=np,
            proc_start=_proc_start(proc.pid),
            pidfile=pidfile,
        )
        self._save(job)
        return job

    def refresh(self, job: Job) -> Job:
        if job.returncode is not None:
            return job
        proc = self._procs.get(job.id)
        if proc is not None:
            proc.poll()  # reap our own children so they do not linger as zombies
        rc_file = self.dir / f"{job.id}.rc"
        alive = _proc_start(job.pid) is not None and _proc_start(job.pid) == job.proc_start
        if rc_file.exists() and rc_file.read_text().strip():
            job.returncode = int(rc_file.read_text().strip())
        elif not alive:
            job.returncode = -1  # vanished without recording an exit code (killed, or machine restarted)
        else:
            return job
        job.finished = time.time()
        self._procs.pop(job.id, None)
        self._save(job)
        return job

    def get(self, job_id: str) -> Job:
        return self.refresh(self._load(job_id))

    def all(self) -> list[Job]:
        jobs = []
        for p in self.dir.glob("*.json"):
            try:
                jobs.append(self.refresh(Job(**json.loads(p.read_text()))))
            except (ValueError, TypeError):
                continue
        return sorted(jobs, key=lambda j: j.started, reverse=True)

    def running_for(self, case_name: str) -> list[Job]:
        return [j for j in self.all() if j.running and j.case == case_name]

    def mark_stop(self, job: Job, mode: str, restore: dict | None = None) -> None:
        job.stop_requested = mode
        if restore:
            job.restore = restore
        self._save(job)

    def kill(self, job: Job, sig: int = signal.SIGTERM, foam: FoamEnv | None = None) -> None:
        if not job.running:
            return
        if job.pidfile and foam is not None:
            foam.runner.kill_job(job.pidfile, signal.Signals(sig).name.removeprefix("SIG"))
        try:
            os.killpg(job.pid, sig)
        except ProcessLookupError:
            pass

    def save(self, job: Job) -> None:
        self._save(job)

    def prune(self, keep: int = 200) -> None:
        jobs = [j for j in self.all() if not j.running]
        for j in jobs[keep:]:
            for suffix in (".json", ".rc", ".pid"):
                (self.dir / f"{j.id}{suffix}").unlink(missing_ok=True)
