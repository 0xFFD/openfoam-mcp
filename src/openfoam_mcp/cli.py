"""Command line: `openfoam-mcp [serve|doctor|install|uninstall|config]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .config import Settings


def _server_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("server options")
    g.add_argument("--workspace", help="Directory for cases (default ~/openfoam-mcp/cases).")
    g.add_argument("--root", action="append", default=[], help="Extra directory tools may access (repeatable).")
    g.add_argument("--bashrc", help="OpenFOAM etc/bashrc to source (default: auto-detect).")
    g.add_argument("--launcher", help="openfoam.com-style launcher, e.g. OpenFOAM.app's `openfoam` on macOS.")
    g.add_argument("--container", help="Run OpenFOAM in this existing Docker/Podman container.")
    g.add_argument("--container-image", help="Run OpenFOAM in a container started from this image.")
    g.add_argument("--container-cli", help="docker or podman executable (default: auto-detect).")
    g.add_argument("--paraview-python", help="Python/pvpython able to `import paraview.simple`.")
    g.add_argument("--no-scripts", action="store_true", help="Forbid running case scripts such as Allrun.")
    g.add_argument("--wait-seconds", type=float, help="Default seconds tools wait for a job (default 45).")


def _settings(ns: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if ns.workspace:
        s.workspace = Path(ns.workspace).expanduser().resolve()
    if ns.root:
        s.extra_roots += [Path(r).expanduser().resolve() for r in ns.root]
    if ns.bashrc:
        s.bashrc = ns.bashrc
    for key in ("launcher", "container", "container_image", "container_cli"):
        if getattr(ns, key):
            setattr(s, key, getattr(ns, key))
    if ns.paraview_python:
        s.paraview_python = ns.paraview_python
    if ns.no_scripts:
        s.allow_scripts = False
    if ns.wait_seconds is not None:
        s.wait_seconds = ns.wait_seconds
    return s


def _forward(ns: argparse.Namespace) -> list[str]:
    """Server options to bake into client configs."""
    out: list[str] = []
    if ns.workspace:
        out += ["--workspace", str(Path(ns.workspace).expanduser().resolve())]
    for r in ns.root:
        out += ["--root", str(Path(r).expanduser().resolve())]
    if ns.bashrc:
        out += ["--bashrc", ns.bashrc]
    for key in ("launcher", "container", "container_image", "container_cli"):
        if getattr(ns, key):
            out += ["--" + key.replace("_", "-"), getattr(ns, key)]
    if ns.paraview_python:
        out += ["--paraview-python", ns.paraview_python]
    if ns.no_scripts:
        out.append("--no-scripts")
    if ns.wait_seconds is not None:
        out += ["--wait-seconds", str(ns.wait_seconds)]
    return out


def cmd_serve(ns: argparse.Namespace) -> int:
    from . import server

    server.configure(_settings(ns))
    if ns.transport == "stdio":
        server.mcp.run("stdio")
    else:
        server.mcp.run("streamable-http", host=ns.host, port=ns.port)
    return 0


def cmd_doctor(ns: argparse.Namespace) -> int:
    from .errors import FoamError
    from .foam import FoamEnv
    from .render import ParaView

    s = _settings(ns)
    ok = True

    def line(status: str, label: str, detail: str = "") -> None:
        print(f"[{status:^4}] {label}{(': ' + detail) if detail else ''}")

    line("ok", "openfoam-mcp", f"{__version__} on Python {sys.version.split()[0]} ({sys.executable}), {sys.platform}")
    try:
        f = FoamEnv.detect(
            bashrc=s.bashrc, launcher=s.launcher, container=s.container, image=s.container_image,
            container_cli=s.container_cli, mounts=s.roots,
        )
        line("ok", "OpenFOAM", f"{f.version} ({'openfoam.org' if f.flavor == 'org' else 'openfoam.com'}) at {f.project_dir}")
        line("ok", "source", f"{f.origin} [{f.runner.kind}]")
        line("ok", "applications", f"{len(f.apps)} found; solver style: {'foamRun' if f.uses_foam_run else 'per-solver apps'}")
        for app in ("blockMesh", "checkMesh", "foamDictionary", f.post_process_app):
            if app not in f.apps:
                ok = False
                line("FAIL", app, "not found")
        line("ok" if f.mpirun else "warn", "mpirun", f.mpirun or "not found (parallel runs unavailable)")
        line("ok" if f.tutorials.is_dir() else "warn", "tutorials", str(f.tutorials))
        try:
            argv, _, ver = ParaView(s.paraview_python, f.host_env).resolve()
            line("ok", "ParaView", f"{ver} via {' '.join(argv)}")
            if sys.platform == "darwin":
                line("info", "macOS", "cases live on a case-insensitive filesystem by default; avoid file names differing only by case")
        except FoamError as e:
            line("warn", "ParaView", str(e))
    except FoamError as e:
        ok = False
        line("FAIL", "OpenFOAM", str(e))
    try:
        s.workspace.mkdir(parents=True, exist_ok=True)
        probe = s.workspace / ".write-test"
        probe.write_text("x")
        probe.unlink()
        line("ok", "workspace", str(s.workspace))
    except OSError as e:
        ok = False
        line("FAIL", "workspace", f"{s.workspace}: {e}")
    print(f"[{'ok' if os.cpu_count() else '?':^4}] cpus: {os.cpu_count()}")
    return 0 if ok else 1


def cmd_install(ns: argparse.Namespace, remove: bool = False) -> int:
    from .install import run_install

    clients = [c.strip() for c in ns.client.split(",")] if ns.client else None
    return run_install(clients, _forward(ns), ns.dry_run, remove, ns.force)


def cmd_config(ns: argparse.Namespace) -> int:
    from .install import print_config

    print_config(ns.client, ns.windows, _forward(ns))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openfoam-mcp", description="MCP server for OpenFOAM.")
    parser.add_argument("--version", action="version", version=f"openfoam-mcp {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("serve", help="Run the MCP server (default).")
    _server_options(p)
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    p = sub.add_parser("doctor", help="Check the OpenFOAM/ParaView/MPI setup.")
    _server_options(p)

    for name, help_ in (("install", "Register the server with detected MCP clients."), ("uninstall", "Remove the server from MCP clients.")):
        p = sub.add_parser(name, help=help_)
        _server_options(p)
        p.add_argument("--client", help="Comma-separated: claude-code,codex,gemini,cursor,windsurf,vscode,claude-desktop,opencode.")
        p.add_argument("--dry-run", action="store_true", help="Print what would be written.")
        p.add_argument("--force", action="store_true", help="Configure named clients even if not detected.")

    p = sub.add_parser("config", help="Print config snippets for every client.")
    _server_options(p)
    p.add_argument("--client")
    p.add_argument("--windows", action="store_true", help="Snippets for Windows clients launching via wsl.exe.")

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("--") and args[0] not in ("--help", "--version"):
        args.insert(0, "serve")
    ns = parser.parse_args(args)
    if ns.command in (None, "serve"):
        return cmd_serve(ns)
    if ns.command == "doctor":
        return cmd_doctor(ns)
    if ns.command == "install":
        return cmd_install(ns)
    if ns.command == "uninstall":
        return cmd_install(ns, remove=True)
    if ns.command == "config":
        return cmd_config(ns)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
