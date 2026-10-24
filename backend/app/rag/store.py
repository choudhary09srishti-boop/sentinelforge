"""SQLite store: documents, chunks, embeddings (float32 blobs) and an FTS5 keyword index."""

import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.rag.embeddings import Vectors

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
  size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL,
  embedding_model TEXT NOT NULL, indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, text TEXT NOT NULL, chunk_hash TEXT NOT NULL,
  start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, page INTEGER, section TEXT,
  risk_flags TEXT NOT NULL DEFAULT '', embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


@dataclass
class DocumentRow:
    id: str
    name: str
    kind: str
    size_bytes: int
    content_hash: str
    embedding_model: str
    indexed_at: str
    chunk_count: int = 0
    risky_chunks: int = 0


@dataclass
class ChunkRow:
    doc_id: str
    ordinal: int
    text: str
    chunk_hash: str
    start_line: int
    end_line: int
    page: int | None
    section: str | None
    embedding: bytes
    risk_flags: list[str] = field(default_factory=list)
    id: int = 0
    doc_name: str = ""


def _doc_from(row: sqlite3.Row) -> DocumentRow:
    return DocumentRow(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        size_bytes=row["size_bytes"],
        content_hash=row["content_hash"],
        embedding_model=row["embedding_model"],
        indexed_at=row["indexed_at"],
        chunk_count=row["chunk_count"],
        risky_chunks=row["risky_chunks"] or 0,
    )


_DOC_SELECT = """
SELECT d.*, COUNT(c.id) AS chunk_count, SUM(c.risk_flags != '') AS risky_chunks
FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
"""


class ChunkStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._write_lock = threading.Lock()
        self._vector_cache: tuple[tuple[str, int, int], list[int], Vectors] | None = None
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- documents ----------------------------------------------------------------------

    def get_document(self, doc_id: str) -> DocumentRow | None:
        with self._connect() as conn:
            row = conn.execute(f"{_DOC_SELECT} WHERE d.id = ? GROUP BY d.id", (doc_id,)).fetchone()
        return _doc_from(row) if row and row["id"] else None

    def get_document_by_hash(self, content_hash: str) -> DocumentRow | None:
        with self._connect() as conn:
            row = conn.execute(
                f"{_DOC_SELECT} WHERE d.content_hash = ? GROUP BY d.id LIMIT 1", (content_hash,)
            ).fetchone()
        return _doc_from(row) if row and row["id"] else None

    def list_documents(self) -> list[DocumentRow]:
        with self._connect() as conn:
            rows = conn.execute(f"{_DOC_SELECT} GROUP BY d.id ORDER BY d.name").fetchall()
        return [_doc_from(r) for r in rows]

    def existing_embeddings(self, doc_id: str, model: str) -> dict[str, bytes]:
        """chunk_hash -> embedding for an already indexed document, to avoid re-embedding."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.chunk_hash, c.embedding FROM chunks c JOIN documents d ON d.id = c.doc_id"
                " WHERE c.doc_id = ? AND d.embedding_model = ?",
                (doc_id, model),
            ).fetchall()
        return {r["chunk_hash"]: r["embedding"] for r in rows}

    def replace_document(self, doc: DocumentRow, chunks: Sequence[ChunkRow]) -> None:
        """Atomically swap a document's chunks and keyword index entries."""
        with self._write_lock, self._connect() as conn:
            self._delete_chunks(conn, doc.id)
            conn.execute("DELETE FROM documents WHERE id = ?", (doc.id,))
            conn.execute(
                "INSERT INTO documents (id, name, kind, size_bytes, content_hash,"
                " embedding_model, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.name,
                    doc.kind,
                    doc.size_bytes,
                    doc.content_hash,
                    doc.embedding_model,
                    doc.indexed_at,
                ),
            )
            for chunk in chunks:
                cursor = conn.execute(
                    "INSERT INTO chunks (doc_id, ordinal, text, chunk_hash, start_line, end_line,"
                    " page, section, risk_flags, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.doc_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.chunk_hash,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.page,
                        chunk.section,
                        ",".join(chunk.risk_flags),
                        chunk.embedding,
                    ),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, chunk.text),
                )

    def delete_document(self, doc_id: str) -> int:
        """Delete a document with all its chunks and index entries; returns chunks removed."""
        with self._write_lock, self._connect() as conn:
            removed = self._delete_chunks(conn, doc_id)
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return removed

    @staticmethod
    def _delete_chunks(conn: sqlite3.Connection, doc_id: str) -> int:
        conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        return conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount

    # ---- search -------------------------------------------------------------------------

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def load_vectors(self, model: str) -> tuple[list[int], Vectors]:
        """Chunk ids and embedding matrix for one model, cached until the index changes."""
        with self._connect() as conn:
            count, max_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(c.id), 0) FROM chunks c"
                " JOIN documents d ON d.id = c.doc_id WHERE d.embedding_model = ?",
                (model,),
            ).fetchone()
            key = (model, int(count), int(max_id))
            if self._vector_cache and self._vector_cache[0] == key:
                return self._vector_cache[1], self._vector_cache[2]
            rows = conn.execute(
                "SELECT c.id, c.embedding FROM chunks c JOIN documents d ON d.id = c.doc_id"
                " WHERE d.embedding_model = ? ORDER BY c.id",
                (model,),
            ).fetchall()
        ids = [int(r["id"]) for r in rows]
        matrix: Vectors = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._vector_cache = (key, ids, matrix)
        return ids, matrix

    def keyword_search(self, query: str, limit: int) -> list[int]:
        """Chunk ids ranked by BM25. Query terms are reduced to safe quoted word tokens."""
        tokens = re.findall(r"\w+", query.lower())[:20]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)"
                " LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [int(r["rowid"]) for r in rows]

    def get_chunks(self, ids: Sequence[int]) -> dict[int, ChunkRow]:
        found: dict[int, ChunkRow] = {}
        with self._connect() as conn:
            for chunk_id in ids:  # at most top_k (<= 20) tiny indexed lookups
                r = conn.execute(
                    "SELECT c.*, d.name AS doc_name FROM chunks c"
                    " JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
                    (chunk_id,),
                ).fetchone()
                if r is None:
                    continue
                found[chunk_id] = ChunkRow(
                    id=int(r["id"]),
                    doc_id=r["doc_id"],
                    doc_name=r["doc_name"],
                    ordinal=r["ordinal"],
                    text=r["text"],
                    chunk_hash=r["chunk_hash"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    page=r["page"],
                    section=r["section"],
                    risk_flags=[f for f in r["risk_flags"].split(",") if f],
                    embedding=r["embedding"],
                )
        return found
