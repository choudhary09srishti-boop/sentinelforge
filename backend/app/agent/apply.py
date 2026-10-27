"""Apply a human-approved patch to the real workspace.

Only the approval API calls this; no MCP tool and no agent node can. The patch is bound to
what the human saw: the file must still match the proposal's base hash and the content must
match the approved patch id, so a stale or tampered patch is refused. Writes are atomic.
"""

import contextlib
import hashlib
import os
import shutil
import tempfile

from pydantic import BaseModel

from app.mcp_server.workspace import Workspace, WorkspaceError


class ApplyResult(BaseModel):
    path: str
    applied: bool
    previous_sha256: str
    new_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_approved_patch(
    workspace: Workspace, *, path: str, new_content: str, base_sha256: str, patch_id: str
) -> ApplyResult:
    target = workspace.resolve(path)  # traversal, symlink escape, secrets, ignored dirs
    if target == workspace.root or target.is_dir():
        raise WorkspaceError("not_a_file", "Path is not a file.")
    new_bytes = new_content.encode("utf-8")
    if len(new_bytes) > workspace.max_read_bytes:
        raise WorkspaceError("too_large", f"New content exceeds {workspace.max_read_bytes} bytes.")
    exists = target.exists()
    old_bytes = workspace.read_for_index(path) if exists else b""
    rel = target.relative_to(workspace.root).as_posix()
    old_hash, new_hash = _sha256(old_bytes), _sha256(new_bytes)
    if old_hash != base_sha256:
        raise WorkspaceError(
            "conflict", "The file changed since the patch was proposed. Propose it again."
        )
    if patch_id != _sha256(f"{rel}\n{old_hash}\n{new_hash}".encode()):
        raise WorkspaceError("conflict", "The content does not match the approved patch.")
    if old_bytes == new_bytes:
        raise WorkspaceError("invalid_argument", "The patch makes no changes.")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".sf_apply_")
    except OSError:
        raise WorkspaceError("write_failed", "Could not write the file.") from None
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(new_bytes)
        if exists:
            shutil.copymode(target, tmp_name)
        os.replace(tmp_name, target)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise WorkspaceError("write_failed", "Could not write the file.") from None
    return ApplyResult(path=rel, applied=True, previous_sha256=old_hash, new_sha256=new_hash)
