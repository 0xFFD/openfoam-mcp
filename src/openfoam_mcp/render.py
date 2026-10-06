"""Locate a ParaView-capable Python and drive `_pv_render.py` in a subprocess."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

from .errors import FoamError

SCRIPT = Path(__file__).with_name("_pv_render.py")
_DIST = "/usr/lib/python3/dist-packages"
# Official ParaView bundles: macOS .app and Kitware's Linux tarballs (often unpacked under /opt).
_PVPYTHON_GLOBS = [
    "/Applications/ParaView*.app/Contents/bin/pvpython",
    "~/Applications/ParaView*.app/Contents/bin/pvpython",
    "/opt/ParaView*/bin/pvpython",
    "~/ParaView*/bin/pvpython",
]


class ParaView:
    def __init__(self, explicit: str | None, base_env: dict[str, str]):
        self.explicit = explicit
        self.base_env = base_env
        self._resolved: tuple[list[str], dict[str, str], str] | None = None
        self._error: str | None = None

    def _interpreters(self):
        env = dict(self.base_env)
        if self.explicit:
            yield [self.explicit], env
            return
        pv = shutil.which("pvpython", path=env.get("PATH"))
        if pv:
            # Ubuntu's pvpython does not put its own modules on sys.path.
            fixed = dict(env, PYTHONPATH=os.pathsep.join(filter(None, [_DIST, env.get("PYTHONPATH")])))
            yield [pv, "--force-offscreen-rendering"], fixed
            yield [pv, "--force-offscreen-rendering"], env
        bundled = []
        for pattern in _PVPYTHON_GLOBS:
            bundled.extend(glob.glob(os.path.expanduser(pattern)))
        for exe in sorted(set(bundled), reverse=True):  # newest version first
            if exe != pv:
                clean = {k: v for k, v in env.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
                yield [exe, "--force-offscreen-rendering"], clean
        seen = set()
        for py in ("python3", "/usr/bin/python3"):
            exe = shutil.which(py, path=env.get("PATH")) if not py.startswith("/") else py
            if exe and Path(exe).exists() and os.path.realpath(exe) not in seen:
                seen.add(os.path.realpath(exe))
                yield [exe], env

    def _candidates(self):
        """Interpreter x display combinations, most direct first.

        MCP clients often launch servers with a scrubbed environment (no DISPLAY), and distro ParaView
        builds need an X server even for off-screen rendering. So also try the local X socket (WSLg or a
        desktop session) and finally a virtual framebuffer via xvfb-run.
        """
        xvfb = shutil.which("xvfb-run", path=self.base_env.get("PATH"))
        x_socket = Path("/tmp/.X11-unix/X0").exists()
        for argv, env in self._interpreters():
            if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
                yield [], argv, env
            elif x_socket:
                yield [], argv, dict(env, DISPLAY=":0")
            else:
                yield [], argv, env  # a ParaView built with EGL/OSMesa needs no display
            if xvfb:
                yield [xvfb, "-a"], argv, env

    def resolve(self) -> tuple[list[str], dict[str, str], str]:
        if self._resolved:
            return self._resolved
        if self._error:
            raise FoamError(self._error)
        tried = []
        probe = SCRIPT.with_name("_pv_probe.py")
        for prefix, argv, env in self._candidates():
            try:
                p = subprocess.run(
                    [*prefix, *argv, str(probe)], env=env, capture_output=True, text=True, timeout=60
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                tried.append(f"{argv[0]}: {e}")
                continue
            for line in p.stdout.splitlines():
                if line.startswith("OFMCP_OK"):
                    self._resolved = ([*prefix, *argv], env, line.split(" ", 1)[1].strip())
                    return self._resolved
            last = (p.stderr or p.stdout).strip().splitlines()[-1:] or [f"exit {p.returncode}"]
            tried.append(f"{' '.join(prefix + argv[:1])}: {last[0][:160]}")
        self._error = (
            "No working ParaView found (needed for render). Install ParaView with Python support "
            "(`sudo apt install paraview python3-paraview`), plus `xvfb` on machines without a display, "
            "or pass --paraview-python. Tried: " + "; ".join(tried)
        )
        raise FoamError(self._error)

    @property
    def available(self) -> bool:
        try:
            self.resolve()
            return True
        except FoamError:
            return False

    def render(self, params: dict, timeout: float = 300) -> dict:
        argv, env, _ = self.resolve()
        try:
            p = subprocess.run(
                [*argv, str(SCRIPT), json.dumps(params)],
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise FoamError(f"Rendering timed out after {timeout:.0f}s") from e
        for line in reversed(p.stdout.splitlines()):
            if line.startswith("OFMCP_RESULT "):
                result = json.loads(line[len("OFMCP_RESULT ") :])
                if "error" in result:
                    raise FoamError(f"Render failed: {result['error']}")
                return result
        err = "\n".join((p.stderr or p.stdout).strip().splitlines()[-15:])
        raise FoamError(f"Render failed (exit {p.returncode}):\n{err}")
