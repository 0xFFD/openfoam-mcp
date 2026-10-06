"""GPU engine support: locating a Python with XLB/JAX/Warp, GPU info, progress parsing, video encoding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import FoamError

WORKER = Path(__file__).with_name("_gpu_lbm.py")
MARKER = "gpu.json"
_PROBE = "import xlb, warp, jax; print('OFMCP_GPU_OK', jax.default_backend(), warp.__version__, xlb.__version__)"
INSTALL_HINT = (
    "The GPU engine needs an NVIDIA GPU (Linux or WSL2) and its own Python environment: run "
    "`openfoam-mcp gpu-setup` once, or point --gpu-python at an interpreter with XLB 0.3.1, JAX (CUDA) and Warp 1.8.1."
)
DEFAULT_ENV = Path.home() / ".local" / "share" / "openfoam-mcp" / "gpu-env"
# Validated together. XLB 0.3.1 declares warp-lang>=1.10, but with Warp >= 1.9 its STL voxelizer silently
# builds the wrong body (>= 1.12 fails outright), so XLB is installed without its declared dependencies.
GPU_PACKAGES = [
    "jax[cuda12]==0.10.2", "warp-lang==1.8.1", "trimesh>=4.4", "numpy>=2.1.2", "matplotlib>=3.9.2", "pyvista>=0.44.1",
    "numpy-stl>=3.1.2", "pydantic>=2.9.1", "imageio>=2.30", "imageio-ffmpeg>=0.5", "pillow>=9",
]
GPU_NO_DEPS = ["xlb==0.3.1"]


def env_python(env: Path) -> Path:
    return env / "bin" / "python"


def setup_env(env: Path = DEFAULT_ENV, python_version: str = "3.11", log=print) -> Path:
    """Create (or update) the dedicated GPU environment and return its interpreter."""
    uv = shutil.which("uv") or (str(Path.home() / ".local/bin/uv") if (Path.home() / ".local/bin/uv").exists() else None)
    env.parent.mkdir(parents=True, exist_ok=True)

    def sh(argv):
        log("$ " + " ".join(argv))
        p = subprocess.run(argv, capture_output=True, text=True)
        if p.returncode != 0:
            raise FoamError(f"{argv[0]} failed: {(p.stderr or p.stdout)[-2000:]}")

    if uv:
        if not env_python(env).exists():
            sh([uv, "venv", "--python", python_version, str(env)])
        sh([uv, "pip", "install", "--python", str(env_python(env)), *GPU_PACKAGES])
        sh([uv, "pip", "install", "--python", str(env_python(env)), "--no-deps", *GPU_NO_DEPS])
    else:
        base = shutil.which(f"python{python_version}")
        if not base:
            raise FoamError(f"Need uv or python{python_version} on PATH to build the GPU environment.")
        if not env_python(env).exists():
            sh([base, "-m", "venv", str(env)])
        sh([str(env_python(env)), "-m", "pip", "install", "-q", *GPU_PACKAGES])
        sh([str(env_python(env)), "-m", "pip", "install", "-q", "--no-deps", *GPU_NO_DEPS])
    return env_python(env)


class GpuEngine:
    def __init__(self, explicit: str | None):
        self.explicit = explicit
        self._python: str | None = None
        self._info: dict | None = None
        self._error: str | None = None

    def python(self) -> str:
        if self._python:
            return self._python
        if self._error:
            raise FoamError(self._error)
        if self.explicit:
            candidates = [self.explicit]
        else:
            candidates = [str(env_python(DEFAULT_ENV))] if env_python(DEFAULT_ENV).exists() else []
            candidates.append(sys.executable)
        tried = []
        for exe in candidates:
            try:
                p = subprocess.run([exe, "-c", _PROBE], capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
            except (OSError, subprocess.TimeoutExpired) as e:
                tried.append(f"{exe}: {e}")
                continue
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("OFMCP_GPU_OK")), None)
            if line:
                _, backend, warp_v, xlb_v = line.split()
                if backend != "gpu":
                    tried.append(f"{exe}: JAX has no GPU backend (found '{backend}')")
                    continue
                self._python = exe
                self._info = {"python": exe, "warp": warp_v, "xlb": xlb_v}
                return exe
            tried.append(f"{exe}: {(p.stderr or p.stdout).strip().splitlines()[-1:] or ['no output']}")
        self._error = INSTALL_HINT + " Tried: " + "; ".join(str(t) for t in tried)
        raise FoamError(self._error)

    def info(self) -> dict:
        out: dict = {"devices": nvidia_smi()}
        try:
            self.python()
            out.update(self._info or {})
        except FoamError as e:
            out["unavailable"] = str(e)
        return out


def nvidia_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        p = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            out.append({"name": parts[0], "memory_total_mb": int(float(parts[1])), "memory_free_mb": int(float(parts[2]))})
    return out


def last_progress(log: Path) -> dict | None:
    """The most recent `OFMCP_PROGRESS {json}` record in a GPU job log."""
    if not log.is_file():
        return None
    size = log.stat().st_size
    with open(log, "rb") as f:
        f.seek(max(0, size - 200_000))
        tail = f.read().decode(errors="replace")
    for line in reversed(tail.splitlines()):
        if line.startswith("OFMCP_PROGRESS "):
            try:
                return json.loads(line[len("OFMCP_PROGRESS ") :])
            except ValueError:
                return None
    return None


def encode(frames: list[Path], target: Path, fps: int) -> Path:
    """Encode PNG frames to MP4 (H.264) or GIF depending on the target suffix."""
    import imageio.v2 as imageio

    if not frames:
        raise FoamError("No frames to encode.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".gif":
        imgs = [imageio.imread(f) for f in frames]
        imageio.mimsave(target, imgs, duration=1000 / fps, loop=0)
    else:
        with imageio.get_writer(target, fps=fps, codec="libx264", quality=8, macro_block_size=2, pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(imageio.imread(f)[..., :3])
    return target


def contact_sheet(frames: list[Path], n: int = 4) -> bytes:
    """A 2x2 grid of evenly spaced frames, so an agent can see the animation without playing it."""
    import io

    from PIL import Image

    picks = [frames[round(i * (len(frames) - 1) / max(n - 1, 1))] for i in range(min(n, len(frames)))]
    imgs = [Image.open(p).convert("RGB") for p in picks]
    w, h = imgs[0].size
    scale = 800 / w
    tw, th = int(w * scale), int(h * scale)
    cols = 2 if len(imgs) > 1 else 1
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()
