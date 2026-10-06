# Contributing

Thanks for helping! Bug reports from real setups (macOS, Docker/Podman, openfoam.com builds,
clusters) are especially valuable — please include the output of `openfoam-mcp doctor`.

## Development setup

```bash
git clone https://github.com/0xFFD/openfoam-mcp && cd openfoam-mcp
uv sync                     # creates .venv with dev dependencies
uv run pytest               # unit tests everywhere; integration tests when OpenFOAM is installed
uv run ruff check src tests
```

Run the server against a scratch workspace while developing:

```bash
uv run openfoam-mcp serve --workspace /tmp/ofmcp-dev
npx @modelcontextprotocol/inspector uv run openfoam-mcp serve --workspace /tmp/ofmcp-dev
```

## Guidelines

- Keep tools general-purpose: prefer improving a primitive over adding a scenario-specific tool.
- Tool output is read by language models: keep it compact, structured and actionable
  (verdicts and advice beat raw dumps).
- Never shell out with user-controlled strings; applications go through `FoamEnv.resolve_app`
  and paths through `paths.resolve_case` / `paths.case_file`.
- Add a test for every parser change, ideally using real OpenFOAM output in `tests/data/`.
- Integration tests must skip cleanly when OpenFOAM is not installed.
