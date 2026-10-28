"""Safe sample data for the public demo.

Runs on every start, so the demo never depends on a persistent disk: the sample workspace is
copied into WORKSPACE_ROOT and the sample documents plus workspace files are indexed. The
same copy backs the demo reset, which restores the intentionally buggy sample files.
"""

import logging
import shutil
import threading
from pathlib import Path

from app.core.config import Settings
from app.core.domain import DomainError
from app.mcp_server.indexing import index_workspace
from app.mcp_server.workspace import Workspace
from app.rag.service import RagService

logger = logging.getLogger("sentinelforge")

MAX_SAMPLE_BYTES = 1_000_000
SKIPPED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".git"})


def seed_workspace(settings: Settings) -> int:
    """Copy samples/workspace into WORKSPACE_ROOT, overwriting. Returns files written."""
    source = Path(settings.samples_dir) / "workspace"
    target = Path(settings.workspace_root)
    if not source.is_dir() or source.resolve() == target.resolve():
        return 0
    written = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if SKIPPED_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SAMPLE_BYTES:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        written += 1
    return written


def index_sample_docs(settings: Settings, rag: RagService) -> tuple[int, int]:
    """Ingest every file in samples/docs. Returns (processed, failed)."""
    docs = Path(settings.samples_dir) / "docs"
    if not docs.is_dir():
        return 0, 0
    processed = failed = 0
    for path in sorted(docs.iterdir()):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SAMPLE_BYTES:
            continue
        try:
            rag.ingest(path.name, path.read_bytes())
            processed += 1
        except DomainError as exc:
            failed += 1
            logger.warning("Sample document %s was not indexed (%s)", path.name, exc.code)
    return processed, failed


def start_bootstrap(settings: Settings, rag: RagService) -> threading.Thread:
    """Index sample data in the background so a cold start can open its port immediately."""

    def work() -> None:
        try:
            processed, failed = index_sample_docs(settings, rag)
            workspace = Workspace(
                Path(settings.workspace_root), max_read_bytes=settings.max_read_bytes
            )
            result = index_workspace(workspace, rag, ".", 50)
            logger.info(
                "Sample data ready: %d documents (%d failed), %d workspace files",
                processed,
                failed,
                result.files_seen,
            )
        except Exception:
            logger.exception("Sample data bootstrap failed")

    thread = threading.Thread(target=work, name="sentinelforge-bootstrap", daemon=True)
    thread.start()
    return thread
