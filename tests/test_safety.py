"""Path sandboxing and application allow-listing."""

from pathlib import Path

import pytest

from openfoam_mcp.config import Settings
from openfoam_mcp.errors import FoamError
from openfoam_mcp.foam import FoamEnv
from openfoam_mcp.paths import case_file, from_windows, resolve_case


@pytest.fixture
def settings(tmp_path):
    ws = tmp_path / "ws"
    (ws / "c1" / "system").mkdir(parents=True)
    return Settings(workspace=ws)


def test_resolve_case_inside_workspace(settings):
    assert resolve_case(settings, "c1") == (settings.workspace / "c1").resolve()


@pytest.mark.parametrize("bad", ["../outside", "/etc", "c1/../../x", ""])
def test_resolve_case_rejects_escapes(settings, bad):
    with pytest.raises(FoamError):
        resolve_case(settings, bad, must_exist=False)


def test_extra_roots(settings, tmp_path):
    other = tmp_path / "other" / "case"
    (other / "system").mkdir(parents=True)
    with pytest.raises(FoamError):
        resolve_case(settings, str(other))
    settings.extra_roots = [tmp_path / "other"]
    assert resolve_case(settings, str(other)) == other.resolve()


def test_case_file_rejects_escape(settings):
    case = settings.workspace / "c1"
    assert case_file(case, "system/controlDict") == (case / "system" / "controlDict").resolve()
    for bad in ("../c2/x", "/etc/passwd", "system/../../x"):
        with pytest.raises(FoamError):
            case_file(case, bad)


def test_windows_paths():
    assert from_windows(r"C:\Users\me\case") == "/mnt/c/Users/me/case"
    assert from_windows("D:/data") == "/mnt/d/data"
    assert from_windows("/home/me") == "/home/me"


def _fake_foam(tmp_path: Path) -> FoamEnv:
    appbin = tmp_path / "platforms" / "bin"
    appbin.mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    for name in ("blockMesh", "foamRun"):
        exe = appbin / name
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    return FoamEnv({"WM_PROJECT_DIR": str(tmp_path), "FOAM_APPBIN": str(appbin), "WM_PROJECT_VERSION": "14"})


def test_app_allow_list(tmp_path):
    foam = _fake_foam(tmp_path)
    case = tmp_path / "case"
    case.mkdir()
    (case / "Allrun").write_text("#!/bin/sh\n")
    assert foam.resolve_app("blockMesh")[0].endswith("blockMesh")
    assert foam.resolve_app("./Allrun", case) == ["bash", str(case / "Allrun")]
    for bad in ("rm", "/bin/sh", "../platforms/bin/blockMesh", "bash"):
        with pytest.raises(FoamError):
            foam.resolve_app(bad, case)
    with pytest.raises(FoamError):
        foam.resolve_app("Allrun", case, allow_scripts=False)


def test_flavor_detection(tmp_path):
    foam = _fake_foam(tmp_path)
    assert foam.flavor == "org" and foam.uses_foam_run and foam.post_process_app == "foamPostProcess"
    esi = FoamEnv({"WM_PROJECT_DIR": str(tmp_path), "WM_PROJECT_VERSION": "v2406"})
    assert esi.flavor == "com" and not esi.uses_foam_run and esi.post_process_app == "postProcess"
