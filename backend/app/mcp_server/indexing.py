"""Index the text and code files of a workspace directory into the document index."""

from app.core.domain import DomainError
from app.mcp_server.workspace import Workspace
from app.rag.models import IndexWorkspaceResult
from app.rag.parsers import WORKSPACE_TEXT_EXTENSIONS
from app.rag.service import RagService


def index_workspace(
    workspace: Workspace, rag: RagService, path: str, max_files: int
) -> IndexWorkspaceResult:
    names, truncated = workspace.list_supported(path, WORKSPACE_TEXT_EXTENSIONS, max_files)
    counts = {"indexed": 0, "reindexed": 0, "unchanged": 0, "duplicate": 0}
    failed: list[str] = []
    for name in names:
        try:
            result = rag.ingest(name, workspace.read_for_index(name))
        except DomainError as exc:
            failed.append(f"{name}: {exc.code}")
            continue
        counts[result.status] += 1
    return IndexWorkspaceResult(
        files_seen=len(names),
        indexed=counts["indexed"],
        reindexed=counts["reindexed"],
        unchanged=counts["unchanged"],
        duplicates=counts["duplicate"],
        failed=failed,
        truncated=truncated,
    )
