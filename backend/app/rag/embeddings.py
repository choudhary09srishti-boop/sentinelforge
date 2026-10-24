"""Embedding providers. The real one is a small CPU ONNX model that runs fully locally."""

import threading
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from app.rag.errors import RagError

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def normalize(vectors: npt.NDArray[Any]) -> Vectors:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.asarray(array / np.where(norms == 0, 1.0, norms), dtype=np.float32)


class FastEmbedEmbedder:
    """fastembed (ONNX, no PyTorch). Downloads the model once, then runs offline."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self.model_name, cache_dir=self._cache_dir
                    )
                except Exception:
                    raise RagError(
                        "embedding_unavailable",
                        "Embedding model could not be loaded. Run the warm-up step once "
                        "while online (python -m app.rag.warmup).",
                    ) from None
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        model = self._load()
        with self._lock:
            return normalize(np.array(list(model.passage_embed(list(texts)))))

    def embed_query(self, text: str) -> Vectors:
        model = self._load()
        with self._lock:
            return normalize(np.array(list(model.query_embed([text]))[0]))
