"""Document routes. Every operation is an MCP tool call made through the gateway."""

import base64
from pathlib import PurePosixPath
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import GatewayDep, SettingsDep, require_api_key
from app.api.schemas import ErrorResponse
from app.api.tool_call import ERROR_STATUSES, call_tool
from app.rag.models import (
    DeleteResult,
    DocumentList,
    IndexWorkspaceResult,
    IngestResult,
    SearchResponse,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS

router = APIRouter(
    prefix="/api/v1",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
    responses={code: {"model": ErrorResponse} for code in ERROR_STATUSES},
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["How do I authenticate?"])
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"


class IndexWorkspaceRequest(BaseModel):
    path: str = Field(default=".", max_length=300)
    max_files: int = Field(default=50, ge=1, le=200)


@router.post("/documents", response_model=IngestResult)
async def upload_document(
    gateway: GatewayDep, settings: SettingsDep, file: UploadFile
) -> IngestResult:
    """Upload and index one document. Only the file name is used, never any client path."""
    name = PurePosixPath((file.filename or "").replace("\\", "/")).name
    if PurePosixPath(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    encoded = base64.b64encode(data).decode("ascii")
    result = await call_tool(gateway, "ingest_document", {"name": name, "content_base64": encoded})
    return IngestResult.model_validate(result)


@router.get("/documents", response_model=DocumentList)
async def list_documents(gateway: GatewayDep) -> DocumentList:
    return DocumentList.model_validate(await call_tool(gateway, "list_documents", {}))


@router.delete("/documents/{document_id}", response_model=DeleteResult)
async def delete_document(
    gateway: GatewayDep, document_id: Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]
) -> DeleteResult:
    result = await call_tool(gateway, "delete_document", {"document_id": document_id})
    return DeleteResult.model_validate(result)


@router.post("/documents/index-workspace", response_model=IndexWorkspaceResult)
async def index_workspace(gateway: GatewayDep, body: IndexWorkspaceRequest) -> IndexWorkspaceResult:
    result = await call_tool(gateway, "index_workspace", body.model_dump())
    return IndexWorkspaceResult.model_validate(result)


@router.post("/search", response_model=SearchResponse)
async def search_documents(gateway: GatewayDep, body: SearchRequest) -> SearchResponse:
    """Retrieve relevant chunks with citations and prompt-injection risk flags."""
    return SearchResponse.model_validate(
        await call_tool(gateway, "search_documents", body.model_dump())
    )
