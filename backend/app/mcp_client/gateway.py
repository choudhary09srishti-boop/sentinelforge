"""The only way the API talks to tools: every call goes through a real MCP client session."""

import json
from typing import Any

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent


class McpToolError(Exception):
    """A structured tool failure reported by the MCP server."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _parse_error(result: CallToolResult) -> McpToolError:
    text = next((c.text for c in result.content if isinstance(c, TextContent)), "")
    _, _, payload = text.partition(": ")
    try:
        data = json.loads(payload)
        return McpToolError(str(data["code"]), str(data["message"]))
    except (ValueError, KeyError, TypeError):
        # Unstructured failures (validation, crashes) are never echoed to callers.
        return McpToolError("tool_error", "Tool call failed.")


class McpGateway:
    def __init__(self, server: MCPServer, *, timeout_seconds: float = 20.0) -> None:
        self._server = server
        self._timeout = timeout_seconds
        self.call_count = 0

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self.call_count += 1
        async with Client(self._server, read_timeout_seconds=self._timeout) as client:
            result = await client.call_tool(tool, arguments or {})
        if result.is_error:
            raise _parse_error(result)
        data: dict[str, Any] | None = result.structured_content
        if data is None:
            raise McpToolError("bad_response", "Tool returned no structured result.")
        return data
