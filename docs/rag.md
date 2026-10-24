# Local RAG

Everything runs on the server. The only network use is the one-time model download.

- **Parsing**: PDF (pypdf, page numbers), Markdown (heading sections), TXT, JSON (normalised to
  indent=2, keyed by top-level key), Python / JavaScript / TypeScript (definition names).
- **Chunking**: about 1000 characters, never across pages, split at section boundaries, one line of
  overlap. Every chunk keeps its start/end line, page and section for citations.
- **Embeddings**: `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU, no PyTorch). Load it once
  with `python -m app.rag.warmup`.
- **Index**: SQLite. Vectors are float32 blobs searched with numpy; keywords use FTS5 BM25. Results
  are fused with Reciprocal Rank Fusion. If the model is unavailable, search falls back to
  keyword-only and says so.
- **Hashing, dedup, reindex, delete**: documents are keyed by name and content SHA-256. Same
  content is `unchanged`; the same content under another name is a `duplicate`; changed content is
  `reindexed` and only changed chunks are re-embedded; repeated chunks are skipped; delete removes
  chunks, vectors and keyword entries together.
- **Prompt-injection defence**: chunks are scanned at ingestion (8 heuristic rules) and carry
  `risk_flags`. `format_untrusted_context` wraps results as escaped, delimited data with a fixed
  warning. Flags are a detection layer only: the structural rule is that retrieved text is data and
  can never authorise a tool call. The heuristics will not catch every attack.
- **Limits**: semantic search always returns the nearest chunks, even if none is truly relevant,
  so callers should read the citations. The index is per process; one server worker is assumed.

## Evaluation

`python scripts/run_retrieval_eval.py` indexes `samples/docs` with the real model, asks the eight
questions in `app/rag/eval.py`, and writes measured results to `docs/retrieval_eval_results.md`.
