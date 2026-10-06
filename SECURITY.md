# Security policy

## Model

openfoam-mcp lets an AI agent operate OpenFOAM on your machine. It confines file access to the
workspace and explicitly allowed roots and only runs OpenFOAM applications or case `All*` scripts,
but it is **not a sandbox**: case scripts and OpenFOAM features such as `#codeStream` and coded
boundary conditions execute arbitrary code by design.

Recommendations:

- Start the server with `--no-scripts` when working with cases from untrusted sources.
- Use the container backend (`--container-image`) or a dedicated user account for isolation.
- Keep the workspace separate from personal data; do not add broad `--root` directories.

## Reporting a vulnerability

Please report security issues privately via GitHub's "Report a vulnerability" (Security
Advisories) on the repository rather than in public issues. Include steps to reproduce and the
affected version. We aim to respond within a week.
