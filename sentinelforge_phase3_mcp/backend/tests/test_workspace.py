import hashlib
from pathlib import Path

import pytest

from app.mcp_server.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(workspace_dir: Path) -> Workspace:
    return Workspace(workspace_dir)


def code_of(exc: pytest.ExceptionInfo[WorkspaceError]) -> str:
    return exc.value.code


def test_list_files_hides_sensitive_and_ignored(ws: Workspace, workspace_dir: Path) -> None:
    (workspace_dir / ".git").mkdir()
    names = {e.path for e in ws.list_files(".", 100).entries}
    assert {"README.md", "src"} <= names
    assert ".env" not in names
    assert ".git" not in names


def test_read_file_returns_requested_line_range(ws: Workspace) -> None:
    result = ws.read_file("src/app.py", start_line=2, end_line=2)
    assert result.content == "    return a - b"
    assert (result.start_line, result.end_line, result.total_lines) == (2, 2, 2)


def test_read_file_caps_lines_per_call(ws: Workspace, workspace_dir: Path) -> None:
    (workspace_dir / "big.txt").write_text("\n".join(["x"] * 600), encoding="utf-8")
    result = ws.read_file("big.txt")
    assert result.truncated is True
    assert result.end_line == 500


@pytest.mark.parametrize(
    "bad_path",
    ["../secret", "src/../../x", "..\\secret", "/etc/passwd", "C:\\Windows\\win.ini", "a\x00b"],
)
def test_rejects_path_traversal_and_absolute_paths(ws: Workspace, bad_path: str) -> None:
    with pytest.raises(WorkspaceError) as exc:
        ws.read_file(bad_path)
    assert code_of(exc) == "invalid_path"


def test_rejects_symlink_escape(ws: Workspace, workspace_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside data", encoding="utf-8")
    try:
        (workspace_dir / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available on this system")
    with pytest.raises(WorkspaceError) as exc:
        ws.read_file("link.txt")
    assert code_of(exc) == "invalid_path"


def test_blocks_sensitive_files(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError) as exc:
        ws.read_file(".env")
    assert code_of(exc) == "forbidden"


def test_rejects_oversized_file(workspace_dir: Path) -> None:
    small = Workspace(workspace_dir, max_read_bytes=10)
    with pytest.raises(WorkspaceError) as exc:
        small.read_file("README.md")
    assert code_of(exc) == "too_large"


def test_rejects_binary_file(ws: Workspace, workspace_dir: Path) -> None:
    (workspace_dir / "blob.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(WorkspaceError) as exc:
        ws.read_file("blob.bin")
    assert code_of(exc) == "binary_file"


def test_missing_file_is_not_found(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError) as exc:
        ws.read_file("nope.py")
    assert code_of(exc) == "not_found"


def test_search_returns_file_and_line_and_skips_secrets(ws: Workspace) -> None:
    found = ws.search("return a", 10)
    assert [(m.path, m.line) for m in found.matches] == [("src/app.py", 2)]
    assert ws.search("topsecret", 10).matches == []


def test_search_truncates_at_max_results(ws: Workspace, workspace_dir: Path) -> None:
    (workspace_dir / "many.txt").write_text("hit\n" * 20, encoding="utf-8")
    result = ws.search("hit", 5)
    assert len(result.matches) == 5
    assert result.truncated is True


def test_search_rejects_empty_query(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError) as exc:
        ws.search("   ", 5)
    assert code_of(exc) == "invalid_argument"


def test_summary_counts_files_without_secrets(ws: Workspace) -> None:
    summary = ws.summary()
    assert summary.file_count == 2
    assert summary.by_extension == {".md": 1, ".py": 1}


def test_propose_patch_builds_diff_and_never_writes(ws: Workspace, workspace_dir: Path) -> None:
    target = workspace_dir / "src" / "app.py"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    proposal = ws.propose_patch("src/app.py", "def add(a, b):\n    return a + b\n")
    assert "-    return a - b" in proposal.diff
    assert "+    return a + b" in proposal.diff
    assert (proposal.additions, proposal.deletions) == (1, 1)
    assert proposal.base_sha256 == before
    assert proposal.applied is False and proposal.requires_approval is True
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_propose_patch_for_new_file(ws: Workspace, workspace_dir: Path) -> None:
    proposal = ws.propose_patch("src/new.py", "x = 1\n")
    assert proposal.is_new_file is True
    assert not (workspace_dir / "src" / "new.py").exists()


def test_propose_patch_patch_id_changes_with_content(ws: Workspace) -> None:
    first = ws.propose_patch("README.md", "one\n")
    second = ws.propose_patch("README.md", "two\n")
    assert first.patch_id != second.patch_id


@pytest.mark.parametrize("bad_path", [".env", "../outside.py", "/abs.py"])
def test_propose_patch_rejects_unsafe_targets(ws: Workspace, bad_path: str) -> None:
    with pytest.raises(WorkspaceError):
        ws.propose_patch(bad_path, "x")


def test_propose_patch_rejects_oversized_content(workspace_dir: Path) -> None:
    small = Workspace(workspace_dir, max_read_bytes=10)
    with pytest.raises(WorkspaceError) as exc:
        small.propose_patch("README.md", "x" * 50)
    assert code_of(exc) == "too_large"
