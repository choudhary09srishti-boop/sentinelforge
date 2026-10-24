"""Typed results of the RAG layer, shared by the MCP tools and the HTTP API."""

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    document_id: str
    source: str
    page: int | None = None
    start_line: int
    end_line: int
    section: str | None = None


class SearchHit(BaseModel):
    chunk_id: int
    text: str
    citation: Citation
    score: float
    vector_score: float | None = None
    keyword_rank: int | None = None
    risk_flags: list[str]


class SearchResponse(BaseModel):
    query: str
    mode_used: Literal["hybrid", "semantic", "keyword"]
    hits: list[SearchHit]
    index_chunks: int
    embedding_model: str
    note: str | None = None


class IngestResult(BaseModel):
    document_id: str
    name: str
    status: Literal["indexed", "reindexed", "unchanged", "duplicate"]
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    duplicate_chunks_skipped: int
    risky_chunks: int
    duplicate_of: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    kind: str
    size_bytes: int
    content_hash: str
    chunk_count: int
    risky_chunks: int
    embedding_model: str
    indexed_at: str


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteResult(BaseModel):
    document_id: str
    name: str
    chunks_removed: int


class IndexWorkspaceResult(BaseModel):
    files_seen: int
    indexed: int
    reindexed: int
    unchanged: int
    duplicates: int
    failed: list[str]
    truncated: bool
