"""Run the 8-question retrieval evaluation with the REAL local embedding model.

Usage (from the project root, venv active):
    python scripts/run_retrieval_eval.py [--mode hybrid|semantic|keyword] [--top-k 3]

Needs the model once: python -m app.rag.warmup  (downloads it, then it runs offline).
Writes docs/retrieval_eval_results.md with the measured numbers.
"""

import argparse
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import FastEmbedEmbedder  # noqa: E402
from app.rag.eval import evaluate  # noqa: E402
from app.rag.service import RagService  # noqa: E402
from app.rag.store import ChunkStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
    store = ChunkStore(Path(tempfile.mkdtemp()) / "eval.sqlite3")
    service = RagService(store, embedder, chunk_max_chars=settings.chunk_max_chars)

    started = time.perf_counter()
    for path in sorted((ROOT / "samples" / "docs").iterdir()):
        service.ingest(path.name, path.read_bytes())
    index_seconds = time.perf_counter() - started
    report = evaluate(service, top_k=args.top_k, mode=args.mode)

    lines = [
        "# Retrieval evaluation results",
        "",
        f"- Run at: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Embedding model: `{settings.embedding_model}` (local, CPU)",
        f"- Mode: `{report.mode}`, top_k={report.top_k}",
        f"- Indexing time for the sample corpus: {index_seconds:.1f}s",
        f"- hit@1 = {report.hit_at_1:.2f}, hit@3 = {report.hit_at_3:.2f}, MRR = {report.mrr:.2f}",
        "",
        "| # | Question | Expected source | Rank found | Top sources |",
        "|---|---|---|---|---|",
    ]
    for number, row in enumerate(report.results, start=1):
        rank = row.found_rank if row.found_rank else "miss"
        lines.append(
            f"| {number} | {row.question} | {row.expected_source} | {rank} | "
            f"{', '.join(row.top_sources)} |"
        )
    output = "\n".join(lines) + "\n"
    (ROOT / "docs" / "retrieval_eval_results.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
