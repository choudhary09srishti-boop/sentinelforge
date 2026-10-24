"""RAG service: validate, parse, chunk, embed incrementally, index, and search with citations."""

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np

from app.core.config import Settings
from app.rag.chunker import chunk_lines
from app.rag.embeddings import Embedder, FastEmbedEmbedder
from app.rag.errors import RagError
from app.rag.models import (
    Citation,
    DeleteResult,
    DocumentInfo,
    DocumentList,
    IngestResult,
    SearchHit,
    SearchResponse,
)
from app.rag.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.rag.safety import scan_for_injection
from app.rag.store import ChunkRow, ChunkStore, DocumentRow

Mode = Literal["hybrid", "semantic", "keyword"]
CANDIDATES = 50
RRF_K = 60
EMBED_BATCH = 32
_DOC_ID = re.compile(r"^[0-9a-f]{16}$")
_BAD_NAME_CHARS = re.compile(r'[<>:"|?*]')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_id_for(name: str) -> str:
    return _sha256(name.encode())[:16]


def clean_name(name: str) -> str:
    """A safe display name: relative, no traversal, no control or markup characters."""
    cleaned = name.replace("\\", "/").strip().lstrip("/")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if (
        not parts
        or ".." in parts
        or len(cleaned) > 200
        or any(ord(ch) < 32 for ch in cleaned)
        or _BAD_NAME_CHARS.search(cleaned)
    ):
        raise RagError("invalid_name", "Document name is not allowed.")
    return "/".join(parts)


def _rrf(rankings: list[list[int]]) -> dict[int, float]:
    """Reciprocal Rank Fusion: combine rankings without comparing incompatible scores."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


class RagService:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        *,
        max_upload_bytes: int = 5 * 1024 * 1024,
        chunk_max_chars: int = 1000,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.max_upload_bytes = max_upload_bytes
        self.chunk_max_chars = chunk_max_chars

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagService":
        store = ChunkStore(Path(settings.data_dir) / "sentinelforge.sqlite3")
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
        return cls(
            store,
            embedder,
            max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
            chunk_max_chars=settings.chunk_max_chars,
        )

    # ---- ingestion ----------------------------------------------------------------------

    def ingest(self, name: str, data: bytes) -> IngestResult:
        name = clean_name(name)
        extension = PurePosixPath(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise RagError("unsupported_type", f"Unsupported file type. Allowed: {allowed}.")
        if not data:
            raise RagError("empty_document", "File is empty.")
        if len(data) > self.max_upload_bytes:
            raise RagError("too_large", f"File exceeds {self.max_upload_bytes} bytes.")

        model = self.embedder.model_name
        content_hash = _sha256(data)
        doc_id = document_id_for(name)
        existing = self.store.get_document(doc_id)

        if existing and existing.content_hash == content_hash and existing.embedding_model == model:
            return self._result(existing, "unchanged", embedded=0, reused=0, skipped=0)
        duplicate = self.store.get_document_by_hash(content_hash)
        if duplicate and duplicate.id != doc_id:
            return IngestResult(
                document_id=duplicate.id,
                name=name,
                status="duplicate",
                chunks_total=duplicate.chunk_count,
                chunks_embedded=0,
                chunks_reused=0,
                duplicate_chunks_skipped=0,
                risky_chunks=duplicate.risky_chunks,
                duplicate_of=duplicate.name,
            )

        drafts = chunk_lines(parse_document(extension, data), self.chunk_max_chars)
        unique, seen = [], set[str]()
        for draft in drafts:
            digest = _sha256(draft.text.encode())
            if digest not in seen:
                seen.add(digest)
                unique.append((digest, draft))
        if not unique:
            raise RagError("empty_document", "No readable text found.")

        reusable = self.store.existing_embeddings(doc_id, model) if existing else {}
        missing = [(h, d.text) for h, d in unique if h not in reusable]
        blobs = dict(reusable)
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start : start + EMBED_BATCH]
            vectors = self.embedder.embed_documents([text for _, text in batch])
            for (digest, _), vector in zip(batch, vectors, strict=True):
                blobs[digest] = np.asarray(vector, dtype=np.float32).tobytes()

        rows = [
            ChunkRow(
                doc_id=doc_id,
                ordinal=ordinal,
                text=draft.text,
                chunk_hash=digest,
                start_line=draft.start_line,
                end_line=draft.end_line,
                page=draft.page,
                section=draft.section,
                risk_flags=scan_for_injection(draft.text),
                embedding=blobs[digest],
            )
            for ordinal, (digest, draft) in enumerate(unique)
        ]
        doc = DocumentRow(
            id=doc_id,
            name=name,
            kind=extension.lstrip("."),
            size_bytes=len(data),
            content_hash=content_hash,
            embedding_model=model,
            indexed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.store.replace_document(doc, rows)
        return IngestResult(
            document_id=doc_id,
            name=name,
            status="reindexed" if existing else "indexed",
            chunks_total=len(rows),
            chunks_embedded=len(missing),
            chunks_reused=len(rows) - len(missing),
            duplicate_chunks_skipped=len(drafts) - len(unique),
            risky_chunks=sum(1 for r in rows if r.risk_flags),
        )

    @staticmethod
    def _result(doc: DocumentRow, status: Literal["unchanged"], **counts: int) -> IngestResult:
        return IngestResult(
            document_id=doc.id,
            name=doc.name,
            status=status,
            chunks_total=doc.chunk_count,
            chunks_embedded=counts["embedded"],
            chunks_reused=counts["reused"],
            duplicate_chunks_skipped=counts["skipped"],
            risky_chunks=doc.risky_chunks,
        )

    # ---- management ---------------------------------------------------------------------

    def list_documents(self) -> DocumentList:
        docs = self.store.list_documents()
        return DocumentList(
            documents=[
                DocumentInfo(
                    document_id=d.id,
                    name=d.name,
                    kind=d.kind,
                    size_bytes=d.size_bytes,
                    content_hash=d.content_hash,
                    chunk_count=d.chunk_count,
                    risky_chunks=d.risky_chunks,
                    embedding_model=d.embedding_model,
                    indexed_at=d.indexed_at,
                )
                for d in docs
            ],
            total_chunks=sum(d.chunk_count for d in docs),
        )

    def delete_document(self, document_id: str) -> DeleteResult:
        if not _DOC_ID.match(document_id):
            raise RagError("invalid_name", "Document id is not valid.")
        doc = self.store.get_document(document_id)
        if doc is None:
            raise RagError("not_found", "Document not found.")
        removed = self.store.delete_document(document_id)
        return DeleteResult(document_id=doc.id, name=doc.name, chunks_removed=removed)

    # ---- search -------------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, mode: Mode = "hybrid") -> SearchResponse:
        query = query.strip()
        if not query or len(query) > 500:
            raise RagError("invalid_argument", "Query must be 1-500 characters.")
        top_k = max(1, min(top_k, 20))
        total = self.store.count_chunks()
        model = self.embedder.model_name
        mode_used: Mode = mode
        note: str | None = None
        vector_ranked: list[int] = []
        vector_scores: dict[int, float] = {}

        if mode != "keyword" and total:
            try:
                query_vector = self.embedder.embed_query(query)
                ids, matrix = self.store.load_vectors(model)
                if ids:
                    scores = matrix @ query_vector
                    order = np.argsort(-scores)[:CANDIDATES]
                    vector_ranked = [ids[i] for i in order]
                    vector_scores = {ids[i]: float(scores[i]) for i in order}
            except RagError as exc:
                if exc.code != "embedding_unavailable" or mode == "semantic":
                    raise
                mode_used = "keyword"
                note = "Embedding model unavailable: results are keyword-only."

        keyword_ranked = (
            self.store.keyword_search(query, CANDIDATES) if mode_used != "semantic" else []
        )
        fused = _rrf([vector_ranked, keyword_ranked])
        top_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
        chunks = self.store.get_chunks(top_ids)
        hits = [
            SearchHit(
                chunk_id=i,
                text=chunks[i].text,
                citation=Citation(
                    document_id=chunks[i].doc_id,
                    source=chunks[i].doc_name,
                    page=chunks[i].page,
                    start_line=chunks[i].start_line,
                    end_line=chunks[i].end_line,
                    section=chunks[i].section,
                ),
                score=round(fused[i], 6),
                vector_score=round(vector_scores[i], 4) if i in vector_scores else None,
                keyword_rank=keyword_ranked.index(i) + 1 if i in keyword_ranked else None,
                risk_flags=chunks[i].risk_flags,
            )
            for i in top_ids
            if i in chunks
        ]
        return SearchResponse(
            query=query,
            mode_used=mode_used,
            hits=hits,
            index_chunks=total,
            embedding_model=model,
            note=note,
        )

    def source_counts(self) -> Counter[str]:
        return Counter(d.kind for d in self.store.list_documents())
