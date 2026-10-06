"""Backends and job management, portable across Linux and macOS.

Container mode is exercised with a stand-in `docker` executable that runs `exec` commands on the
host with a scrubbed environment (as a container would have), so the full container code path -
environment sourcing inside the "container", app discovery, job PID files, in-container kill and
tutorial mirroring - runs without Docker.
"""

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from openfoam_mcp import jobs as jobs_mod
from openfoam_mcp.errors import FoamError
from openfoam_mcp.foam import ContainerRunner, FoamEnv, find_bashrc, launcher_env, source_bashrc
from openfoam_mcp.jobs import JobManager

FAKE_DOCKER = r"""#!/usr/bin/env bash
cmd=$1; shift
case "$cmd" in
  inspect) echo true ;;
  start|run) echo fake ;;
  exec)
    wd=""
    while [[ "$1" == -* ]]; do
      case "$1" in -w) wd=$2; shift 2 ;; -e) shift 2 ;; *) shift ;; esac
    done
    shift  # container name
    [ -n "$wd" ] && cd "$wd"
    exec env -i HOME=/tmp PATH=/usr/bin:/bin "$@" ;;
  cp) cp -r "${1#*:}" "$2" ;;
  *) echo "unsupported: $cmd" >&2; exit 2 ;;
esac
"""


@pytest.fixture
def fake_install(tmp_path):
    """A minimal fake OpenFOAM tree: bashrc, an app that prints and sleeps, a tutorials dir."""
    root = tmp_path / "OpenFOAM-99"
    (root / "etc").mkdir(parents=True)
    appbin = root / "platforms" / "bin"
    appbin.mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "tutorials" / "demo" / "system").mkdir(parents=True)
    (root / "tutorials" / "demo" / "system" / "controlDict").write_text("solver demo;\n")
    (root / "etc" / "bashrc").write_text(
        f"export WM_PROJECT_DIR={root}\nexport WM_PROJECT_VERSION=99\nexport FOAM_APPBIN={appbin}\n"
        f"export FOAM_TUTORIALS={root}/tutorials\nexport PATH={appbin}:$PATH\n"
    )
    app = appbin / "fakeSolver"
    app.write_text('#!/bin/sh\necho "Time = 1s"\necho "running $*"\nsleep "${1:-0}"\necho End\n')
    app.chmod(0o755)
    return root


def test_source_bashrc_portable(fake_install):
    env = source_bashrc(str(fake_install / "etc" / "bashrc"))
    assert env["WM_PROJECT_VERSION"] == "99"


def test_launcher(fake_install, tmp_path):
    launcher = tmp_path / "openfoam"
    # Mimics openfoam.com's etc/openfoam: `openfoam -c '<commands>'` sources bashrc then runs bash -c.
    launcher.write_text(f'#!/bin/bash\n[ "$1" = -c ] || exit 2\nsource {fake_install}/etc/bashrc\nexec bash -c "$2"\n')
    launcher.chmod(0o755)
    env = launcher_env(str(launcher))
    assert env["WM_PROJECT_DIR"] == str(fake_install)
    foam = FoamEnv.detect(launcher=str(launcher))
    assert "fakeSolver" in foam.apps and foam.origin.startswith("launcher")


def test_job_lifecycle_and_kill(fake_install, tmp_path):
    foam = FoamEnv(source_bashrc(str(fake_install / "etc" / "bashrc")))
    case = tmp_path / "case"
    case.mkdir()
    jm = JobManager(tmp_path / "state")
    quick = jm.start(case, "case", "fakeSolver", [foam.apps["fakeSolver"], "0"], case / "log.quick", foam)
    for _ in range(50):
        if not jm.refresh(quick).running:
            break
        time.sleep(0.1)
    assert quick.returncode == 0 and "End" in (case / "log.quick").read_text()

    slow = jm.start(case, "case", "fakeSolver", [foam.apps["fakeSolver"], "60"], case / "log.slow", foam)
    time.sleep(0.5)
    assert jm.refresh(slow).running
    # A fresh manager (as after a server restart) still sees the job as running.
    assert JobManager(tmp_path / "state").get(slow.id).running
    jm.kill(slow, signal.SIGTERM, foam)
    for _ in range(50):
        if not jm.refresh(slow).running:
            break
        time.sleep(0.1)
    assert not slow.running


def test_proc_start_ps_fallback(monkeypatch):
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if str(self) == "/proc/self/stat" else real_exists(self))
    assert jobs_mod._proc_start(os.getpid())
    assert jobs_mod._proc_start(2**22 + 12345) is None


@pytest.fixture
def fake_docker(tmp_path):
    exe = tmp_path / "bin" / "docker"
    exe.parent.mkdir()
    exe.write_text(FAKE_DOCKER)
    exe.chmod(0o755)
    return str(exe)


def test_container_runner(fake_install, fake_docker, tmp_path):
    runner = ContainerRunner(fake_docker, "fake", str(fake_install / "etc" / "bashrc"))
    env = runner.load_env()
    assert env["WM_PROJECT_VERSION"] == "99"
    foam = FoamEnv(env, "container fake", runner, cache_dir=tmp_path / "cache")
    assert "fakeSolver" in foam.apps
    res = foam.run_app("fakeSolver", ["0"], cwd=tmp_path)
    assert "running 0" in res.stdout
    # tutorials are mirrored to a host cache
    assert (foam.tutorials / "demo" / "system" / "controlDict").is_file()
    assert foam.tutorials.is_relative_to(tmp_path / "cache")


LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="the in-container kill walks /proc (containers are Linux)")


@LINUX_ONLY
def test_container_job_killed_inside(fake_install, fake_docker, tmp_path):
    runner = ContainerRunner(fake_docker, "fake", str(fake_install / "etc" / "bashrc"))
    foam = FoamEnv(runner.load_env(), "container fake", runner)
    case = tmp_path / "case"
    case.mkdir()
    jm = JobManager(tmp_path / "state")
    job = jm.start(case, "case", "fakeSolver", [foam.apps["fakeSolver"], "60"], case / "log.fakeSolver", foam)
    time.sleep(1.0)
    assert job.pidfile and Path(job.pidfile).read_text().strip()
    inner = int(Path(job.pidfile).read_text())
    jm.kill(job, signal.SIGTERM, foam)
    for _ in range(50):
        if not jm.refresh(job).running:
            break
        time.sleep(0.1)
    assert not job.running
    with pytest.raises(ProcessLookupError):
        os.kill(inner, 0)  # the in-"container" process itself is gone


@LINUX_ONLY
def test_container_kill_reaches_script_children(fake_install, fake_docker, tmp_path):
    """Stopping an Allrun-style script must also stop the solver it spawned inside the container."""
    runner = ContainerRunner(fake_docker, "fake", str(fake_install / "etc" / "bashrc"))
    foam = FoamEnv(runner.load_env(), "container fake", runner)
    case = tmp_path / "case"
    case.mkdir()
    marker = tmp_path / "child.pid"
    (case / "Allrun").write_text(f"#!/bin/bash\nsleep 60 &\necho $! > {marker}\nwait\n")
    jm = JobManager(tmp_path / "state")
    job = jm.start(case, "case", "Allrun", foam.resolve_app("Allrun", case), case / "log.Allrun", foam)
    for _ in range(50):
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.1)
    child = int(marker.read_text())
    jm.kill(job, signal.SIGTERM, foam)
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


# ---------------------------------------------------------------- real OpenFOAM through the container path

REAL = find_bashrc()


@pytest.mark.integration
@pytest.mark.skipif(not REAL or sys.platform != "linux", reason="needs a native OpenFOAM install to stand in for a container")
def test_full_case_through_container_backend(fake_docker, tmp_path, monkeypatch):
    from openfoam_mcp import server as S
    from openfoam_mcp.config import Settings

    ws = tmp_path / "ws"
    S.configure(Settings(workspace=ws, container="fake", container_cli=fake_docker, wait_seconds=120))
    S._app().foam.cache_dir = tmp_path / "cache"
    info = S.foam_info()
    assert info["openfoam"]["backend"] == "container"
    tuts = S.list_tutorials(query="pitzDailySteady")
    assert tuts["results"] and tuts["tutorials_dir"].startswith(str(tmp_path / "cache"))
    S.create_case("pitz", "incompressibleFluid/pitzDailySteady")
    S.set_dict("pitz", "system/controlDict", set={"endTime": "5000", "writeInterval": "1000"})
    S.set_dict("pitz", "system/fvSolution", remove=["SIMPLE/residualControl"])
    assert S.run("pitz", "blockMesh", ["-dict", "$FOAM_TUTORIALS/resources/blockMesh/pitzDaily"])["state"] == "succeeded"
    r = S.run("pitz", "foamRun", wait_seconds=3)
    assert r["state"] == "running"
    stopped = S.stop_job(r["job_id"], mode="kill")
    assert stopped["job"]["state"] in ("stopped", "failed")
    assert not S.job_status(r["job_id"])["state"] == "running"
    with pytest.raises(FoamError):
        S.run("pitz", "rm", ["-rf", "/"])
