import base64
import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from app.core.config import Settings
from app.mcp_server import create_mcp_server
from app.rag.service import RagService

GUIDE = b"# Guide\n## Authentication\n" + b"Send the X-API-Key header with every request. " * 6


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def error_code(result: CallToolResult) -> str:
    assert result.is_error
    first = result.content[0]
    assert isinstance(first, TextContent)
    return str(json.loads(first.text.partition(": ")[2])["code"])


@pytest.fixture
def server(settings: Settings, rag_service: RagService) -> MCPServer:
    return create_mcp_server(settings, rag=rag_service)


async def test_document_tools_are_discoverable(server: MCPServer) -> None:
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    expected = {"ingest_document", "search_documents", "list_documents", "delete_document"}
    assert expected | {"index_workspace"} <= names
    assert len(names) >= 11


async def test_ingest_search_and_delete_through_mcp(server: MCPServer) -> None:
    async with Client(server) as client:
        ingested = await client.call_tool(
            "ingest_document", {"name": "guide.md", "content_base64": b64(GUIDE)}
        )
        assert ingested.structured_content is not None
        assert ingested.structured_content["status"] == "indexed"

        found = await client.call_tool("search_documents", {"query": "which header holds the key"})
        assert found.structured_content is not None
        top = found.structured_content["hits"][0]
        assert top["citation"]["source"] == "guide.md"
        assert top["citation"]["start_line"] == 1

        doc_id = ingested.structured_content["document_id"]
        deleted = await client.call_tool("delete_document", {"document_id": doc_id})
        assert deleted.structured_content is not None
        assert deleted.structured_content["chunks_removed"] >= 1
        listing = await client.call_tool("list_documents", {})
        assert listing.structured_content is not None
        assert listing.structured_content["documents"] == []


async def test_bad_inputs_return_structured_errors(server: MCPServer) -> None:
    async with Client(server) as client:
        bad_b64 = await client.call_tool(
            "ingest_document", {"name": "a.md", "content_base64": "!!"}
        )
        bad_type = await client.call_tool(
            "ingest_document", {"name": "a.exe", "content_base64": b64(b"MZ")}
        )
        bad_k = await client.call_tool("search_documents", {"query": "x", "top_k": 0})
        bad_id = await client.call_tool("delete_document", {"document_id": "../../etc"})
    assert error_code(bad_b64) == "invalid_content"
    assert error_code(bad_type) == "unsupported_type"
    assert bad_k.is_error and bad_id.is_error


async def test_index_workspace_skips_secrets_and_is_searchable(server: MCPServer) -> None:
    async with Client(server) as client:
        indexed = await client.call_tool("index_workspace", {})
        assert indexed.structured_content is not None
        assert indexed.structured_content["indexed"] == 2  # README.md and src/app.py, not .env
        found = await client.call_tool(
            "search_documents", {"query": "topsecret", "mode": "keyword"}
        )
        semantic = await client.call_tool("search_documents", {"query": "topsecret"})
        code = await client.call_tool("search_documents", {"query": "def add return"})
    assert found.structured_content is not None and found.structured_content["hits"] == []
    assert semantic.structured_content is not None
    assert all("topsecret" not in h["text"] for h in semantic.structured_content["hits"])
    assert code.structured_content is not None
    assert code.structured_content["hits"][0]["citation"]["source"] == "src/app.py"


async def test_index_workspace_rejects_traversal(server: MCPServer, workspace_dir: Path) -> None:
    async with Client(server) as client:
        result = await client.call_tool("index_workspace", {"path": "../.."})
    assert error_code(result) == "invalid_path"
