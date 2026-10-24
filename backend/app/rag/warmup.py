"""Download and verify the embedding model once: python -m app.rag.warmup"""

import time

from app.core.config import get_settings
from app.rag.embeddings import FastEmbedEmbedder


def main() -> None:
    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
    started = time.perf_counter()
    vectors = embedder.embed_documents(["SentinelForge warm-up sentence."])
    elapsed = time.perf_counter() - started
    print(f"model={settings.embedding_model} dim={vectors.shape[1]} load+embed={elapsed:.1f}s")
    print(f"cache_dir={settings.model_cache_dir}")


if __name__ == "__main__":
    main()
