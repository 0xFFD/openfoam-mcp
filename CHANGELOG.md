# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-06

First public release.

### Added
- 24 MCP tools covering the whole CFD loop: tutorials and case management, dictionary
  reading/editing through `foamDictionary`, meshing and `checkMesh` analysis, background solver
  jobs, convergence monitoring, post-processing, field statistics and ParaView rendering.
- Persistent background jobs that survive client/server restarts, with graceful
  write-and-stop and live edits made visible to running solvers.
- Incremental solver-log parser with converged / diverging / failed / crashed verdicts.
- Automatic decomposition and MPI execution for `np > 1`.
- Support for openfoam.org (11+, `foamRun`) and openfoam.com (ESI) installations.
- Backends: native Linux and macOS installations, openfoam.com launchers (e.g. OpenFOAM.app on
  macOS) and Docker/Podman containers.
- `openfoam-mcp install` for Claude Code, Codex CLI, Gemini CLI, Cursor, VS Code, Windsurf,
  Claude Desktop and opencode, including Windows clients reaching OpenFOAM in WSL.
- `openfoam-mcp doctor` environment diagnostics.
- `import_file` for geometry (STL/OBJ) and `$FOAM_TUTORIALS/resources`, including Windows paths under WSL.
- Rendering `focus` on a patch or group; Q-criterion and other iso-surfaces.
- Field statistics from binary as well as ASCII field files.
- Steady-state plateau detection in convergence reports.
- Parallel `checkMesh` and post-processing on decomposed cases; `-copyZero` decomposition before
  parallel meshing; automatic fallback to scotch when a simple/hierarchical split does not match `np`.
