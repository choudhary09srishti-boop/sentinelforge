"""Sandbox access for the agent, routed through the MCP `run_sandbox_tests` tool.

The agent never touches the sandbox runner directly: the call goes through the MCP gateway like
every other core operation. A failed tool call becomes an honest error result, never a pass.
"""

from typing import Any

from app.agent.graph import ToolCaller
from app.mcp_client import McpToolError
from app.sandbox.runner import SandboxResult


class McpSandbox:
    def __init__(self, tools: ToolCaller) -> None:
        self._tools = tools

    async def run_tests(
        self, patch_path: str | None = None, patch_content: str | None = None
    ) -> SandboxResult:
        arguments: dict[str, Any] = {}
        if patch_path is not None:
            arguments["patch_path"] = patch_path
        if patch_content is not None:
            arguments["patch_content"] = patch_content
        try:
            data = await self._tools.call("run_sandbox_tests", arguments)
        except McpToolError as exc:
            status = "timeout" if exc.code == "timeout" else "error"
            return SandboxResult(
                status=status,
                exit_code=None,
                output=f"The sandbox tool failed ({exc.code}): {exc.message}",
                duration_seconds=0.0,
            )
        return SandboxResult.model_validate(data)
