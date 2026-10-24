"""Call an MCP tool from a route and translate tool failures into consistent HTTP errors."""

from typing import Any

from fastapi import HTTPException

from app.mcp_client import McpGateway, McpToolError

STATUS_BY_CODE = {
    "invalid_path": 400,
    "invalid_argument": 400,
    "invalid_name": 400,
    "invalid_content": 400,
    "not_a_file": 400,
    "not_a_directory": 400,
    "forbidden": 403,
    "not_found": 404,
    "too_large": 413,
    "binary_file": 415,
    "unsupported_type": 415,
    "empty_document": 422,
    "parse_error": 422,
    "embedding_unavailable": 503,
    "timeout": 504,
}
ERROR_STATUSES = sorted({*STATUS_BY_CODE.values(), 401})


async def call_tool(gateway: McpGateway, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return await gateway.call(tool, arguments)
    except McpToolError as exc:
        status = STATUS_BY_CODE.get(exc.code, 500)
        raise HTTPException(status_code=status, detail=exc.message) from None
