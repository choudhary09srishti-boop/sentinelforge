import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.agent.apply import apply_approved_patch
from app.mcp_server.workspace import Workspace, WorkspaceError

OLD = "def add(a, b):\n    return a - b\n"
NEW = "def add(a, b):\n    return a + b\n"
BAD_ID = "0" * 64


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_ws(tmp_path: Path, files: dict[str, bytes], **kwargs: Any) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return Workspace(root, **kwargs)


def approved(ws: Workspace, path: str, new: str) -> dict[str, str]:
    """What the human approves: exactly what propose_patch returned for this content."""
    proposal = ws.propose_patch(path, new)
    return {
        "path": path,
        "new_content": new,
        "base_sha256": proposal.base_sha256,
        "patch_id": proposal.patch_id,
    }


def names(ws: Workspace) -> list[str]:
    return sorted(p.name for p in ws.root.iterdir())


def test_applies_exact_content_and_touches_nothing_else(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"calculator.py": OLD.encode(), "other.py": b"x = 1\n"})
    result = apply_approved_patch(ws, **approved(ws, "calculator.py", NEW))
    assert result.applied is True and result.path == "calculator.py"
    assert result.previous_sha256 == sha(OLD.encode()) and result.new_sha256 == sha(NEW.encode())
    assert (ws.root / "calculator.py").read_bytes() == NEW.encode()
    assert (ws.root / "other.py").read_bytes() == b"x = 1\n"
    assert names(ws) == ["calculator.py", "other.py"]


def test_crlf_and_non_ascii_bytes_are_written_exactly(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"a.py": b"x = 1\n"})
    new = "x = 'caf\u00e9'\r\ny = 2\r\n"
    apply_approved_patch(ws, **approved(ws, "a.py", new))
    assert (ws.root / "a.py").read_bytes() == new.encode("utf-8")


def test_security_stale_base_is_rejected_and_file_left_alone(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"calculator.py": OLD.encode()})
    request = approved(ws, "calculator.py", NEW)
    (ws.root / "calculator.py").write_bytes(b"changed by someone else\n")
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "conflict"
    assert (ws.root / "calculator.py").read_bytes() == b"changed by someone else\n"
    assert names(ws) == ["calculator.py"]


def test_security_tampered_content_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"calculator.py": OLD.encode()})
    request = approved(ws, "calculator.py", NEW)
    request["new_content"] = NEW + "import os\nos.system('evil')\n"
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "conflict"
    assert (ws.root / "calculator.py").read_bytes() == OLD.encode()


def test_security_wrong_patch_id_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"calculator.py": OLD.encode()})
    request = approved(ws, "calculator.py", NEW)
    request["patch_id"] = BAD_ID
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "conflict"
    assert (ws.root / "calculator.py").read_bytes() == OLD.encode()


def test_new_file_is_created_with_parent_folders(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"keep.py": b"x = 1\n"})
    result = apply_approved_patch(ws, **approved(ws, "pkg/sub/new.py", "y = 2\n"))
    assert result.path == "pkg/sub/new.py" and result.previous_sha256 == sha(b"")
    assert (ws.root / "pkg" / "sub" / "new.py").read_bytes() == b"y = 2\n"


def test_security_file_created_after_proposal_blocks_new_file_patch(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"keep.py": b"x = 1\n"})
    request = approved(ws, "new.py", "y = 2\n")
    (ws.root / "new.py").write_bytes(b"someone else wrote this\n")
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "conflict"
    assert (ws.root / "new.py").read_bytes() == b"someone else wrote this\n"


@pytest.mark.parametrize(
    "bad", ["../evil.py", "/etc/passwd", "C:/evil.py", ".env", "a/../../b.py", "node_modules/x.js"]
)
def test_security_traversal_and_sensitive_paths_are_rejected(tmp_path: Path, bad: str) -> None:
    ws = make_ws(tmp_path, {"keep.py": b"x = 1\n"})
    request = {"path": bad, "new_content": "x = 1\n", "base_sha256": sha(b""), "patch_id": BAD_ID}
    with pytest.raises(WorkspaceError):
        apply_approved_patch(ws, **request)
    assert not (tmp_path / "evil.py").exists() and not (ws.root / ".env").exists()
    assert names(ws) == ["keep.py"]


def test_security_symlink_escape_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"keep.py": b"x = 1\n"})
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    try:
        (ws.root / "link.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not permitted on this machine.")
    request = {
        "path": "link.py",
        "new_content": "owned\n",
        "base_sha256": sha(b"secret"),
        "patch_id": BAD_ID,
    }
    with pytest.raises(WorkspaceError):
        apply_approved_patch(ws, **request)
    assert outside.read_bytes() == b"secret"


def test_security_oversized_content_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"a.py": OLD.encode()}, max_read_bytes=1000)
    request = {
        "path": "a.py",
        "new_content": "x" * 2000,
        "base_sha256": sha(OLD.encode()),
        "patch_id": BAD_ID,
    }
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "too_large"
    assert (ws.root / "a.py").read_bytes() == OLD.encode()


def test_patch_without_changes_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"a.py": OLD.encode()})
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **approved(ws, "a.py", OLD))
    assert info.value.code == "invalid_argument"


def test_directory_target_is_rejected(tmp_path: Path) -> None:
    ws = make_ws(tmp_path, {"pkg/a.py": b"x = 1\n"})
    request = {"path": "pkg", "new_content": "x\n", "base_sha256": sha(b""), "patch_id": BAD_ID}
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "not_a_file"


def test_failed_write_leaves_original_intact_and_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = make_ws(tmp_path, {"calculator.py": OLD.encode()})
    request = approved(ws, "calculator.py", NEW)

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("app.agent.apply.os.replace", boom)
    with pytest.raises(WorkspaceError) as info:
        apply_approved_patch(ws, **request)
    assert info.value.code == "write_failed"
    assert (ws.root / "calculator.py").read_bytes() == OLD.encode()
    assert names(ws) == ["calculator.py"]
