"""Client configuration writers, exercised against a fake home directory."""

import json

import pytest
import tomlkit

from openfoam_mcp import install


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(install, "windows_dirs", lambda: None)
    monkeypatch.setattr(install, "_claude_cli", lambda win: None)
    return tmp_path


def _targets(home):
    return {t.key: t for t in install.targets()}


def test_codex_preserves_existing_config(home):
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "foo"\n')
    t = _targets(home)["codex"]
    spec = install.launch(["--workspace", "/w"], windows=False)
    install.apply(t, spec)
    doc = tomlkit.parse(cfg.read_text())
    assert doc["model"] == "x"
    assert doc["mcp_servers"]["other"]["command"] == "foo"
    entry = doc["mcp_servers"]["openfoam"]
    assert entry["args"][-2:] == ["--workspace", "/w"]
    assert entry["tool_timeout_sec"] == 600
    assert (cfg.parent / "config.toml.bak-openfoam-mcp").exists()
    install.apply(t, spec, remove=True)
    assert "openfoam" not in tomlkit.parse(cfg.read_text())["mcp_servers"]


def test_codex_scattered_server_tables(home):
    """Real configs often interleave [mcp_servers.*] with other tables."""
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(
        '[mcp_servers.a]\ncommand = "a"\n\n[projects."c:\\\\x"]\ntrust_level = "trusted"\n\n'
        '[mcp_servers.b]\ncommand = "b"\n'
    )
    t = _targets(home)["codex"]
    install.apply(t, install.launch([], windows=False))
    install.apply(t, install.launch(["--no-scripts"], windows=False))  # re-install replaces the entry
    doc = tomlkit.parse(cfg.read_text())
    assert {"a", "b", "openfoam"} <= set(doc["mcp_servers"])
    assert doc["mcp_servers"]["openfoam"]["args"][-1] == "--no-scripts"
    assert doc["projects"]["c:\\x"]["trust_level"] == "trusted"
    install.apply(t, install.launch([], windows=False), remove=True)
    assert "openfoam" not in tomlkit.parse(cfg.read_text())["mcp_servers"]


@pytest.mark.parametrize(
    "key,section",
    [("cursor", "mcpServers"), ("gemini", "mcpServers"), ("claude-code", "mcpServers"), ("vscode", "servers"), ("opencode", "mcp")],
)
def test_json_clients(home, key, section):
    t = _targets(home)[key]
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.path.write_text(json.dumps({section: {"keep": {"command": "x"}}, "other": 1}))
    install.apply(t, install.launch([], windows=False))
    data = json.loads(t.path.read_text())
    assert data["other"] == 1 and "keep" in data[section]
    entry = data[section]["openfoam"]
    argv = entry["command"] if key == "opencode" else [entry["command"], *entry["args"]]
    assert argv[-3:] == ["-m", "openfoam_mcp", "serve"]


def test_windows_launch_uses_wsl(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    spec = install.launch(["--workspace", "/w"], windows=True)
    assert spec.command == "wsl.exe"
    assert spec.args[:3] == ["-d", "Ubuntu-22.04", "--exec"]
    assert spec.args[-2:] == ["--workspace", "/w"]


def test_jsonc_is_not_clobbered(home):
    t = _targets(home)["vscode"]
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.path.write_text('{ // comment\n "servers": {} }')
    with pytest.raises(ValueError):
        install.apply(t, install.launch([], windows=False))
    assert "// comment" in t.path.read_text()
