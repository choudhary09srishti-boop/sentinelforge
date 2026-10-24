"""Deterministic test embedders. They exist only in tests, never in the application."""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.rag.embeddings import Vectors, normalize
from app.rag.errors import RagError


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording gives similar vectors, fully offline."""

    model_name = "fake-hash-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.embedded: list[str] = []

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0  # noqa: S324
        return normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.embedded.extend(texts)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._vector(text)


class UnavailableEmbedder:
    model_name = "fake-hash-embedder"

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        raise RagError("embedding_unavailable", "model missing")

    def embed_query(self, text: str) -> Vectors:
        raise RagError("embedding_unavailable", "model missing")
