"""The agent reaches the sandbox only through the MCP run_sandbox_tests tool."""

import pytest

from app.agent.mcp_sandbox import McpSandbox
from app.core.config import Settings
from app.mcp_client import McpGateway, McpToolError
from app.mcp_server import create_mcp_server
from app.rag.service import RagService
from app.sandbox.runner import SandboxError, SandboxResult

PATCH = "def add(a, b):\n    return a + b\n"


class FakeSandbox:
    def __init__(self, result: SandboxResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str | None, str | None]] = []

    async def run_tests(
        self, patch_path: str | None = None, patch_content: str | None = None
    ) -> SandboxResult:
        self.calls.append((patch_path, patch_content))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def gateway_for(settings: Settings, rag: RagService, sandbox: FakeSandbox) -> McpGateway:
    server = create_mcp_server(settings, rag, sandbox=sandbox)
    return McpGateway(server, timeout_seconds=30)


async def test_tests_run_through_the_mcp_tool(settings: Settings, rag_service: RagService) -> None:
    fake = FakeSandbox(
        SandboxResult(status="passed", exit_code=0, output="2 passed", duration_seconds=0.5)
    )
    result = await McpSandbox(gateway_for(settings, rag_service, fake)).run_tests(
        "src/app.py", PATCH
    )
    assert result.status == "passed"
    assert fake.calls == [("src/app.py", PATCH)]


async def test_sandbox_failure_is_never_reported_as_passed(
    settings: Settings, rag_service: RagService
) -> None:
    fake = FakeSandbox(error=SandboxError("sandbox exploded"))
    result = await McpSandbox(gateway_for(settings, rag_service, fake)).run_tests(
        "src/app.py", PATCH
    )
    assert result.status == "error"
    assert result.exit_code is None


async def test_patch_arguments_must_come_together(
    settings: Settings, rag_service: RagService
) -> None:
    fake = FakeSandbox(SandboxResult(status="passed", exit_code=0, output="", duration_seconds=0))
    gateway = gateway_for(settings, rag_service, fake)
    with pytest.raises(McpToolError) as caught:
        await gateway.call("run_sandbox_tests", {"patch_path": "src/app.py"})
    assert caught.value.code == "invalid_argument"
    assert fake.calls == []
