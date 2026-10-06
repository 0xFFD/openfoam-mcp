"""Register the server with MCP clients (Claude Code, Codex, Gemini CLI, Cursor, VS Code, ...).

When run inside WSL, clients installed on the Windows side are configured too, launching the
server through `wsl.exe -d <distro> --exec ...`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tomlkit

NAME = "openfoam"


# ---------------------------------------------------------------- environment


def in_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def windows_dirs() -> dict[str, Path] | None:
    """USERPROFILE and APPDATA of the Windows user, as WSL paths."""
    if not in_wsl() or not shutil.which("cmd.exe"):
        return None
    try:
        out = subprocess.run(
            ["cmd.exe", "/c", "echo %USERPROFILE%& echo %APPDATA%"],
            capture_output=True, text=True, timeout=20, cwd="/mnt/c",
        ).stdout.splitlines()
        prof, appdata = out[0].strip(), out[1].strip()
        conv = lambda w: Path(subprocess.run(["wslpath", "-u", w], capture_output=True, text=True).stdout.strip())  # noqa: E731
        return {"home": conv(prof), "appdata": conv(appdata)}
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


@dataclass
class Launch:
    command: str
    args: list[str]

    @property
    def argv(self) -> list[str]:
        return [self.command, *self.args]


def launch(server_args: list[str], windows: bool) -> Launch:
    base = ["-m", "openfoam_mcp", "serve", *server_args]
    if windows:
        distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
        return Launch("wsl.exe", ["-d", distro, "--exec", sys.executable, *base])
    return Launch(sys.executable, base)


# ---------------------------------------------------------------- clients


@dataclass
class Target:
    key: str
    label: str
    path: Path
    marker: Path  # existence means the client is installed
    fmt: str  # mcpServers | claude-code | vscode | opencode | codex
    windows: bool

    def describe(self) -> str:
        side = "Windows" if self.windows else ("WSL" if in_wsl() else "local")
        return f"{self.label} [{side}] {self.path}"


def targets() -> list[Target]:
    out: list[Target] = []
    homes: list[tuple[Path, Path | None, bool]] = []
    home = Path.home()
    linux_appdata = home / ".config"
    if sys.platform == "darwin":
        linux_appdata = home / "Library" / "Application Support"
    homes.append((home, linux_appdata, False))
    if w := windows_dirs():
        homes.append((w["home"], w["appdata"], True))
    for h, appdata, win in homes:
        codex_home = Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") and not win else h / ".codex"
        out += [
            Target("claude-code", "Claude Code", h / ".claude.json", h / ".claude", "claude-code", win),
            Target("codex", "Codex CLI", codex_home / "config.toml", codex_home, "codex", win),
            Target("gemini", "Gemini CLI", h / ".gemini" / "settings.json", h / ".gemini", "mcpServers", win),
            Target("cursor", "Cursor", h / ".cursor" / "mcp.json", h / ".cursor", "mcpServers", win),
            Target("windsurf", "Windsurf", h / ".codeium" / "windsurf" / "mcp_config.json", h / ".codeium" / "windsurf", "mcpServers", win),
            Target("opencode", "opencode", h / ".config" / "opencode" / "opencode.json", h / ".config" / "opencode", "opencode", win),
        ]
        if appdata:
            out += [
                Target("claude-desktop", "Claude Desktop", appdata / "Claude" / "claude_desktop_config.json", appdata / "Claude", "mcpServers", win),
                Target("vscode", "VS Code", appdata / "Code" / "User" / "mcp.json", appdata / "Code" / "User", "vscode", win),
            ]
    return out


def entry_for(fmt: str, spec: Launch) -> dict:
    if fmt == "claude-code":
        return {"type": "stdio", "command": spec.command, "args": spec.args, "env": {}}
    if fmt == "vscode":
        return {"type": "stdio", "command": spec.command, "args": spec.args}
    if fmt == "opencode":
        return {"type": "local", "command": spec.argv, "enabled": True}
    if fmt == "gemini":
        return {"command": spec.command, "args": spec.args, "timeout": 600000}
    return {"command": spec.command, "args": spec.args}


def snippet(t: Target, spec: Launch) -> str:
    if t.fmt == "codex":
        doc = tomlkit.document()
        doc.add("mcp_servers", {NAME: _codex_table(spec)})
        return tomlkit.dumps(doc)
    key = {"vscode": "servers", "opencode": "mcp"}.get(t.fmt, "mcpServers")
    fmt = "gemini" if t.key == "gemini" else t.fmt
    return json.dumps({key: {NAME: entry_for(fmt, spec)}}, indent=2)


def _codex_table(spec: Launch):
    tbl = tomlkit.table()
    tbl.add("command", spec.command)
    tbl.add("args", spec.args)
    tbl.add("startup_timeout_sec", 30)
    tbl.add("tool_timeout_sec", 600)
    tbl.add(tomlkit.nl())
    return tbl


# ---------------------------------------------------------------- writing


def _backup(path: Path) -> None:
    bak = path.with_name(path.name + ".bak-openfoam-mcp")
    if path.exists() and not bak.exists():
        # Contents only: copying metadata fails with EPERM on Windows drives mounted in WSL.
        shutil.copyfile(path, bak)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-openfoam-mcp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not plain JSON ({e}); add the snippet by hand") from None


def _claude_cli(win: bool) -> list[str] | None:
    if win:
        return ["cmd.exe", "/c", "claude"] if shutil.which("cmd.exe") else None
    exe = shutil.which("claude")
    return [exe] if exe else None


def apply(t: Target, spec: Launch, remove: bool = False) -> str:
    if t.fmt == "claude-code":
        cli = _claude_cli(t.windows)
        if cli:
            kw = {"capture_output": True, "text": True, "timeout": 60}
            if t.windows:
                kw["cwd"] = "/mnt/c"
            subprocess.run([*cli, "mcp", "remove", "--scope", "user", NAME], **kw)
            if remove:
                return "removed (claude mcp remove)"
            r = subprocess.run([*cli, "mcp", "add", "--scope", "user", NAME, "--", *spec.argv], **kw)
            if r.returncode == 0:
                return "added (claude mcp add --scope user)"
        # Fall back to editing ~/.claude.json directly.
    if t.fmt == "codex":
        doc = tomlkit.parse(t.path.read_text(encoding="utf-8")) if t.path.exists() else tomlkit.document()
        servers = doc.get("mcp_servers")
        if remove:
            if servers is None or NAME not in servers:
                return "not present"
            del servers[NAME]
        else:
            if servers is None:
                servers = tomlkit.table(is_super_table=True)
                doc.add("mcp_servers", servers)
            # Item assignment also works when the [mcp_servers.*] tables are scattered through the file
            # (tomlkit then returns an out-of-order proxy without .add()).
            servers[NAME] = _codex_table(spec)
        _backup(t.path)
        _atomic_write(t.path, tomlkit.dumps(doc))
        return "removed" if remove else "written"

    data = _load_json(t.path)
    key = {"vscode": "servers", "opencode": "mcp"}.get(t.fmt, "mcpServers")
    section = data.setdefault(key, {})
    if remove:
        if NAME not in section:
            return "not present"
        del section[NAME]
    else:
        section[NAME] = entry_for("gemini" if t.key == "gemini" else t.fmt, spec)
    if t.fmt == "opencode":
        data.setdefault("$schema", "https://opencode.ai/config.json")
    _backup(t.path)
    _atomic_write(t.path, json.dumps(data, indent=2) + "\n")
    return "removed" if remove else "written"


def run_install(clients: list[str] | None, server_args: list[str], dry_run: bool, remove: bool, all_clients: bool) -> int:
    found = targets()
    if clients:
        unknown = set(clients) - {t.key for t in found}
        if unknown:
            print(f"Unknown client(s): {', '.join(sorted(unknown))}. Known: {', '.join(sorted({t.key for t in found}))}")
            return 2
        chosen = [t for t in found if t.key in clients and (t.marker.exists() or t.path.exists() or all_clients)]
        # A client named explicitly but not detected anywhere is still configured on the native side.
        for c in clients:
            if not any(t.key == c for t in chosen):
                chosen += [t for t in found if t.key == c and not t.windows][:1]
    else:
        chosen = [t for t in found if t.marker.exists() or t.path.exists()]
    if not chosen:
        print("No MCP clients detected. Use --client <name> to configure one explicitly, or `openfoam-mcp config`.")
        return 1
    rc = 0
    for t in chosen:
        spec = launch(server_args, t.windows)
        if dry_run:
            print(f"\n# {t.describe()}\n{snippet(t, spec)}")
            continue
        try:
            print(f"{'-' if remove else '+'} {t.describe()}: {apply(t, spec, remove)}")
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            rc = 1
            print(f"! {t.describe()}: {e}\n{snippet(t, spec)}")
    if not dry_run and not remove:
        print("\nRestart the client(s) to load the 'openfoam' server.")
    return rc


def print_config(client: str | None, windows: bool, server_args: list[str]) -> None:
    spec = launch(server_args, windows)
    seen = set()
    for t in targets():
        if (client and t.key != client) or t.key in seen:
            continue
        seen.add(t.key)
        print(f"# {t.label}  ({t.path.name})\n{snippet(t, spec)}\n")
