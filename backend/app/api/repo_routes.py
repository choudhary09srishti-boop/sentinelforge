"""Repository routes. Each one is a thin wrapper that calls an MCP tool via the gateway."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import GatewayDep, require_api_key
from app.api.schemas import ErrorResponse
from app.api.tool_call import ERROR_STATUSES, call_tool
from app.mcp_server.models import (
    FileContent,
    ListFilesResult,
    PatchProposal,
    RepoSummary,
    SearchResult,
)

router = APIRouter(
    prefix="/api/v1/repo",
    tags=["repository"],
    dependencies=[Depends(require_api_key)],
    responses={code: {"model": ErrorResponse} for code in ERROR_STATUSES},
)


class PatchRequest(BaseModel):
    path: str = Field(min_length=1, max_length=300)
    new_content: str = Field(max_length=200_000)


@router.get("/files", response_model=ListFilesResult)
async def list_files(
    gateway: GatewayDep,
    path: Annotated[str, Query(max_length=300)] = ".",
    max_entries: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> ListFilesResult:
    data = await call_tool(gateway, "list_repo_files", {"path": path, "max_entries": max_entries})
    return ListFilesResult.model_validate(data)


@router.get("/file", response_model=FileContent)
async def read_file(
    gateway: GatewayDep,
    path: Annotated[str, Query(min_length=1, max_length=300)],
    start_line: Annotated[int, Query(ge=1)] = 1,
    end_line: Annotated[int | None, Query(ge=1)] = None,
) -> FileContent:
    args = {"path": path, "start_line": start_line, "end_line": end_line}
    return FileContent.model_validate(await call_tool(gateway, "read_repo_file", args))


@router.get("/search", response_model=SearchResult)
async def search(
    gateway: GatewayDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    max_results: Annotated[int, Query(ge=1, le=200)] = 50,
) -> SearchResult:
    data = await call_tool(gateway, "search_repo", {"query": q, "max_results": max_results})
    return SearchResult.model_validate(data)


@router.get("/summary", response_model=RepoSummary)
async def summary(gateway: GatewayDep) -> RepoSummary:
    return RepoSummary.model_validate(await call_tool(gateway, "repo_summary", {}))


@router.post("/patches/propose", response_model=PatchProposal)
async def propose_patch(gateway: GatewayDep, body: PatchRequest) -> PatchProposal:
    """Return a diff only. Nothing is written or applied."""
    data = await call_tool(gateway, "propose_patch", body.model_dump())
    return PatchProposal.model_validate(data)
