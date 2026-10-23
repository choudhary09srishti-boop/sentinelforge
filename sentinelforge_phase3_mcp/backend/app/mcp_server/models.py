"""Typed results returned by MCP tools and reused by the HTTP API."""

from typing import Literal

from pydantic import BaseModel


class FileEntry(BaseModel):
    path: str
    kind: Literal["file", "dir"]
    size: int | None = None


class ListFilesResult(BaseModel):
    path: str
    entries: list[FileEntry]
    truncated: bool


class FileContent(BaseModel):
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    truncated: bool


class SearchMatch(BaseModel):
    path: str
    line: int
    text: str


class SearchResult(BaseModel):
    query: str
    matches: list[SearchMatch]
    files_scanned: int
    truncated: bool


class LargeFile(BaseModel):
    path: str
    size: int


class RepoSummary(BaseModel):
    file_count: int
    total_bytes: int
    by_extension: dict[str, int]
    largest_files: list[LargeFile]
    truncated: bool


class PatchProposal(BaseModel):
    """A proposed change. Nothing is written: applying requires separate human approval."""

    path: str
    is_new_file: bool
    has_changes: bool
    diff: str
    additions: int
    deletions: int
    base_sha256: str
    new_sha256: str
    patch_id: str
    requires_approval: bool = True
    applied: bool = False


class HealthInfo(BaseModel):
    status: Literal["ok", "degraded"]
    server_version: str
    workspace_available: bool
    provider_mode: str
    external_provider: bool
