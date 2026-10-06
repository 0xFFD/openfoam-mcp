"""Path resolution with sandboxing to the configured roots."""

from __future__ import annotations

import re
from pathlib import Path

from .config import Settings
from .errors import FoamError

_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def from_windows(path: str) -> str:
    """Translate `C:\\foo\\bar` to `/mnt/c/foo/bar` so Windows-side clients can pass native paths."""
    m = _WIN_PATH.match(path.strip())
    if not m:
        return path
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_case(settings: Settings, case: str, must_exist: bool = True) -> Path:
    """Resolve a case name (relative to the workspace) or an absolute path inside an allowed root."""
    if not case or not case.strip():
        raise FoamError("A case name or path is required.")
    raw = Path(from_windows(case)).expanduser()
    path = (raw if raw.is_absolute() else settings.workspace / raw).resolve()
    if not any(within(path, r) for r in settings.roots):
        roots = ", ".join(str(r) for r in settings.roots)
        raise FoamError(f"{path} is outside the allowed roots ({roots}). Start the server with --root to allow it.")
    if path in settings.roots:
        raise FoamError("The case must be a directory inside the workspace, not the workspace itself.")
    if must_exist:
        if not path.is_dir():
            raise FoamError(f"Case not found: {path}. Use list_cases or create_case.")
        if not (path / "system").is_dir() and not (path / "gpu.json").is_file():
            raise FoamError(f"{path} is not an OpenFOAM case (no system/ directory) or a GPU case.")
    return path


def case_file(case_dir: Path, rel: str) -> Path:
    """Resolve a path relative to a case, refusing anything that escapes it."""
    p = Path(from_windows(rel))
    if p.is_absolute():
        target = p.resolve()
    else:
        target = (case_dir / p).resolve()
    if not within(target, case_dir.resolve()):
        raise FoamError(f"'{rel}' is outside the case directory.")
    return target


def display(case_dir: Path, settings: Settings) -> str:
    """Short name for a case: relative to the workspace when possible."""
    ws = settings.workspace.resolve()
    return str(case_dir.relative_to(ws)) if within(case_dir, ws) else str(case_dir)
