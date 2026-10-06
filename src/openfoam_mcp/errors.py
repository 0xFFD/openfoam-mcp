"""Error type surfaced to the MCP client as a tool error (isError=true)."""

from mcp.server.mcpserver.exceptions import ToolError


class FoamError(ToolError):
    """A user-facing failure: bad arguments, missing files, failed OpenFOAM commands."""
