"""GPU engine support code (no GPU needed), plus an optional end-to-end run when the engine is installed."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openfoam_mcp import gpu
from openfoam_mcp.errors import FoamError


def _frames(tmp_path: Path, n: int = 6) -> list[Path]:
    out = []
    for i in range(n):
        img = np.zeros((90, 160, 3), np.uint8)
        img[:, : (i + 1) * 20] = (40 * i, 200, 255 - 40 * i)
        p = tmp_path / f"vortex_{i:05d}.png"
        Image.fromarray(img).save(p)
        out.append(p)
    return out


def test_last_progress(tmp_path):
    log = tmp_path / "log.gpu"
    log.write_text(
        'warming up\nOFMCP_PROGRESS {"stage": "setup", "cells": 100}\nstep 10\n'
        'OFMCP_PROGRESS {"stage": "running", "pct": 42.0, "eta_s": 30}\nstep 20\n'
    )
    assert gpu.last_progress(log) == {"stage": "running", "pct": 42.0, "eta_s": 30}
    assert gpu.last_progress(tmp_path / "missing") is None


@pytest.mark.parametrize("suffix", [".mp4", ".gif"])
def test_encode(tmp_path, suffix):
    target = gpu.encode(_frames(tmp_path), tmp_path / f"out{suffix}", fps=12)
    assert target.is_file() and target.stat().st_size > 1000


def test_encode_needs_frames(tmp_path):
    with pytest.raises(FoamError):
        gpu.encode([], tmp_path / "x.mp4", 24)


def test_contact_sheet(tmp_path):
    png = gpu.contact_sheet(_frames(tmp_path))
    im = Image.open(__import__("io").BytesIO(png))
    assert im.size[0] == 1600  # two columns of 800 px


def test_setup_env_commands(tmp_path, monkeypatch):
    """gpu-setup installs XLB without its declared deps and pins the Warp version that voxelizes correctly."""
    calls = []
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(gpu.subprocess, "run", lambda argv, **kw: calls.append(argv) or type("P", (), {"returncode": 0})())
    gpu.setup_env(tmp_path / "env", log=lambda *_: None)
    flat = [" ".join(c) for c in calls]
    assert any("venv --python 3.11" in c for c in flat)
    assert any("warp-lang==1.8.1" in c for c in flat)
    assert any("jax[cuda12]==0.10.2" in c for c in flat)
    assert any("--no-deps xlb==0.3.1" in c for c in flat)


# ---------------------------------------------------------------- optional real run


def _write_sphere_stl(path: Path, radius: float, n_lat: int = 24, n_lon: int = 48) -> None:
    """ASCII STL of a UV sphere (the test environment has no mesh library)."""
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon + 1)
    pts = lambda i, j: radius * np.array([np.sin(th[i]) * np.cos(ph[j]), np.sin(th[i]) * np.sin(ph[j]), np.cos(th[i])])  # noqa: E731
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b, c, d = pts(i, j), pts(i + 1, j), pts(i + 1, j + 1), pts(i, j + 1)
            if i > 0:
                tris.append((a, b, d))
            if i < n_lat - 1:
                tris.append((b, c, d))
    lines = ["solid sphere"]
    for t in tris:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        n = n / (np.linalg.norm(n) or 1)
        lines += [f"facet normal {n[0]} {n[1]} {n[2]}", "outer loop", *(f"vertex {v[0]} {v[1]} {v[2]}" for v in t), "endloop", "endfacet"]
    path.write_text("\n".join(lines + ["endsolid sphere", ""]))

ENGINE = gpu.GpuEngine(None)
try:
    ENGINE.python()
    HAVE_GPU = True
except FoamError:
    HAVE_GPU = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_GPU, reason="GPU engine not installed (openfoam-mcp gpu-setup)")
def test_gpu_simulate_sphere(tmp_path):
    from openfoam_mcp import server as S
    from openfoam_mcp.config import Settings

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_sphere_stl(ws / "sphere.stl", radius=0.1)
    S.configure(Settings(workspace=ws, wait_seconds=600))
    r = S.gpu_simulate("sphere", str(ws / "sphere.stl"), resolution=24, flow_throughs=0.6, frames=6, view="side")
    assert r["state"] == "succeeded", r
    meta = json.loads((ws / "sphere" / "meta.json").read_text())
    assert meta["cell_reynolds"] <= 100 and meta["blockage_ratio"] > 0
    assert (ws / "sphere" / "vortex.mp4").stat().st_size > 1000
    forces = np.loadtxt(ws / "sphere" / "forces.csv", delimiter=",", skiprows=1)
    assert np.all(np.isfinite(forces))
    video, sheet = S.animate("sphere", kind="slice", format="gif", fps=6)
    assert Path(video["video"]).suffix == ".gif" and video["frames"] >= 6


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_GPU, reason="GPU engine not installed (openfoam-mcp gpu-setup)")
def test_gpu_ground_contact_and_direction(tmp_path):
    """A body standing on the floor, wind from -y: links to the floor must not count as body force."""
    from openfoam_mcp import server as S
    from openfoam_mcp.config import Settings

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_sphere_stl(ws / "s.stl", radius=0.1)
    S.configure(Settings(workspace=ws, wait_seconds=600))
    r = S.gpu_simulate("g", str(ws / "s.stl"), direction="-y", ground=True, resolution=24, flow_throughs=0.8, frames=4, view="side")
    assert r["state"] == "succeeded", r
    assert "WARNING" not in (ws / "g" / "log.gpu").read_text()
    f = np.loadtxt(ws / "g" / "forces.csv", delimiter=",", skiprows=1)[-1, 1:]
    # Before the fix, floor links added thousands of newtons of vertical force on a 20 cm sphere.
    assert abs(f[2]) < 10 and 0 < f[0] < 10, f
