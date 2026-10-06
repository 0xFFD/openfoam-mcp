"""Runtime settings, resolved from CLI flags and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WORKSPACE = Path.home() / "openfoam-mcp" / "cases"


@dataclass
class Settings:
    workspace: Path = DEFAULT_WORKSPACE
    # Extra directories (besides the workspace) that tools may read and write.
    extra_roots: list[Path] = field(default_factory=list)
    # Explicit OpenFOAM etc/bashrc; auto-detected when None.
    bashrc: str | None = None
    # openfoam.com-style launcher (`openfoam -c ...`), e.g. OpenFOAM.app on macOS.
    launcher: str | None = None
    # Run OpenFOAM inside a container: an existing container name, or an image to start one from.
    container: str | None = None
    container_image: str | None = None
    container_cli: str | None = None  # docker or podman; auto-detected when None
    # Interpreter able to `import paraview.simple`; auto-detected when None.
    paraview_python: str | None = None
    # Interpreter with XLB, JAX (CUDA) and Warp for the GPU engine; the server's own Python when None.
    gpu_python: str | None = None
    # Whether case-local scripts (Allrun, Allclean, ...) may be executed.
    allow_scripts: bool = True
    # Default seconds a tool waits for a job before handing back a job id.
    wait_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> Settings:
        s = cls()
        if v := os.environ.get("OPENFOAM_MCP_WORKSPACE"):
            s.workspace = Path(v).expanduser()
        if v := os.environ.get("OPENFOAM_MCP_ROOTS"):
            s.extra_roots = [Path(p).expanduser() for p in v.split(os.pathsep) if p]
        s.bashrc = os.environ.get("OPENFOAM_MCP_BASHRC") or None
        s.launcher = os.environ.get("OPENFOAM_MCP_LAUNCHER") or None
        s.container = os.environ.get("OPENFOAM_MCP_CONTAINER") or None
        s.container_image = os.environ.get("OPENFOAM_MCP_CONTAINER_IMAGE") or None
        s.container_cli = os.environ.get("OPENFOAM_MCP_CONTAINER_CLI") or None
        s.paraview_python = os.environ.get("OPENFOAM_MCP_PARAVIEW_PYTHON") or None
        s.gpu_python = os.environ.get("OPENFOAM_MCP_GPU_PYTHON") or None
        if os.environ.get("OPENFOAM_MCP_NO_SCRIPTS"):
            s.allow_scripts = False
        if v := os.environ.get("OPENFOAM_MCP_WAIT_SECONDS"):
            s.wait_seconds = float(v)
        return s

    @property
    def roots(self) -> list[Path]:
        return [self.workspace.resolve(), *(p.resolve() for p in self.extra_roots)]

    @property
    def state_dir(self) -> Path:
        return self.workspace / ".openfoam-mcp"
