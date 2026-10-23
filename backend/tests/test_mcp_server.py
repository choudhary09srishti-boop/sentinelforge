import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from app.core.config import Settings
from app.mcp_server import create_mcp_server
from app.mcp_server.workspace import Workspace

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(settings: Settings) -> MCPServer:
    return create_mcp_server(settings)


def error_payload(result: CallToolResult) -> dict[str, str]:
    assert result.is_error
    first = result.content[0]
    assert isinstance(first, TextContent)
    return json.loads(first.text.partition(": ")[2])  # type: ignore[no-any-return]


async def test_discovery_lists_tools_resource_and_prompt(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        resources = (await client.list_resources()).resources
        prompts = (await client.list_prompts()).prompts
    assert len(tools) >= 6
    assert {"list_repo_files", "read_repo_file", "search_repo", "propose_patch"} <= set(tools)
    assert "path" in tools["read_repo_file"].input_schema["properties"]
    assert [str(r.uri) for r in resources] == ["sentinelforge://policy/security"]
    assert [p.name for p in prompts] == ["plan_coding_task"]


async def test_tool_call_returns_structured_result(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("list_repo_files", {"path": "."})
    assert not result.is_error
    assert result.structured_content is not None
    paths = {e["path"] for e in result.structured_content["entries"]}
    assert "README.md" in paths and ".env" not in paths


async def test_invalid_arguments_are_rejected(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("list_repo_files", {"max_entries": "abc"})
    assert result.is_error


async def test_path_traversal_returns_structured_error_without_leaking_paths(
    server: MCPServer, workspace_dir: Path
) -> None:
    async with Client(server) as client:
        result = await client.call_tool("read_repo_file", {"path": "../../etc/passwd"})
    assert error_payload(result)["code"] == "invalid_path"
    assert str(workspace_dir) not in result.content[0].model_dump_json()


async def test_sensitive_file_is_blocked_through_mcp(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("read_repo_file", {"path": ".env"})
    assert error_payload(result)["code"] == "forbidden"
    assert "topsecret" not in result.content[0].model_dump_json()


async def test_slow_tool_hits_timeout_with_structured_error(
    workspace_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(self: Workspace) -> None:
        time.sleep(1.0)

    monkeypatch.setattr(Workspace, "summary", slow)
    quick = Settings(
        _env_file=None,
        app_env="test",
        workspace_root=str(workspace_dir),
        tool_timeout_seconds=0.2,
    )
    async with Client(create_mcp_server(quick)) as client:
        result = await client.call_tool("repo_summary", {})
    assert error_payload(result)["code"] == "timeout"


async def test_resource_and_prompt_content(server: MCPServer) -> None:
    async with Client(server) as client:
        resource = await client.read_resource("sentinelforge://policy/security")
        prompt = await client.get_prompt("plan_coding_task", {"task": "fix add()"})
    assert "untrusted" in resource.contents[0].text  # type: ignore[union-attr]
    assert "fix add()" in prompt.messages[0].content.text  # type: ignore[union-attr]


async def test_real_stdio_transport_round_trip(workspace_dir: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(BACKEND_DIR),
        env={
            **os.environ,
            "PYTHONPATH": str(BACKEND_DIR),
            "WORKSPACE_ROOT": str(workspace_dir),
            "APP_ENV": "test",
        },
    )
    async with Client(params) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        health = await client.call_tool("server_health", {})
        listing = await client.call_tool("list_repo_files", {})
    assert "propose_patch" in tools
    assert health.structured_content is not None
    assert health.structured_content["status"] == "ok"
    assert listing.structured_content is not None
    assert "README.md" in {e["path"] for e in listing.structured_content["entries"]}


def test_mcp_health_route(server: MCPServer) -> None:
    response = TestClient(server.streamable_http_app()).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
