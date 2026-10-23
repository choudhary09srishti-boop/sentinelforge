"""Safe access to one workspace directory. Pure logic: no MCP or HTTP code lives here."""

import difflib
import fnmatch
import hashlib
import os
import re
from pathlib import Path

from app.mcp_server.models import (
    FileContent,
    FileEntry,
    LargeFile,
    ListFilesResult,
    PatchProposal,
    RepoSummary,
    SearchMatch,
    SearchResult,
)

IGNORED_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
    | {".pytest_cache"}
)
SENSITIVE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    "credentials*",
)
SENSITIVE_ALLOWED = frozenset({".env.example"})
MAX_LINES_PER_READ = 500
MAX_QUERY_LENGTH = 200
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class WorkspaceError(Exception):
    """An anticipated failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def is_sensitive(name: str) -> bool:
    lowered = name.lower()
    if lowered in SENSITIVE_ALLOWED:
        return False
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SENSITIVE_PATTERNS)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Workspace:
    def __init__(self, root: Path, *, max_read_bytes: int = 200_000, max_files: int = 5000) -> None:
        self.root = root.resolve()
        self.max_read_bytes = max_read_bytes
        self.max_files = max_files

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    # ---- path safety -------------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """Map a user-supplied relative path to a safe absolute path inside the workspace."""
        if not rel or "\x00" in rel:
            raise WorkspaceError("invalid_path", "Path is empty or contains invalid characters.")
        norm = rel.replace("\\", "/")
        if norm.startswith("/") or _WINDOWS_DRIVE.match(norm):
            raise WorkspaceError("invalid_path", "Absolute paths are not allowed.")
        parts = [p for p in norm.split("/") if p not in ("", ".")]
        if ".." in parts:
            raise WorkspaceError("invalid_path", "Path traversal is not allowed.")
        candidate = self.root.joinpath(*parts).resolve()  # follows symlinks
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError("invalid_path", "Path escapes the workspace.")
        for part in candidate.relative_to(self.root).parts:
            if part in IGNORED_DIRS or is_sensitive(part):
                raise WorkspaceError("forbidden", "Access to this path is not allowed.")
        return candidate

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _read_bytes(self, path: Path) -> bytes:
        if not path.exists():
            raise WorkspaceError("not_found", "File not found.")
        if not path.is_file():
            raise WorkspaceError("not_a_file", "Path is not a file.")
        if path.stat().st_size > self.max_read_bytes:
            raise WorkspaceError("too_large", f"File exceeds {self.max_read_bytes} bytes.")
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            raise WorkspaceError("binary_file", "Binary files cannot be read.")
        return data

    def _collect_files(self) -> tuple[list[Path], bool]:
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.is_symlink() or is_sensitive(name):
                    continue
                if len(files) >= self.max_files:
                    return files, True
                files.append(path)
        return files, False

    # ---- operations --------------------------------------------------------------------

    def list_files(self, path: str, max_entries: int) -> ListFilesResult:
        directory = self.resolve(path)
        if not directory.exists():
            raise WorkspaceError("not_found", "Directory not found.")
        if not directory.is_dir():
            raise WorkspaceError("not_a_directory", "Path is not a directory.")
        entries: list[FileEntry] = []
        truncated = False
        for child in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if child.is_symlink() or child.name in IGNORED_DIRS or is_sensitive(child.name):
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            if child.is_dir():
                entries.append(FileEntry(path=self._rel(child), kind="dir"))
            else:
                entries.append(
                    FileEntry(path=self._rel(child), kind="file", size=child.stat().st_size)
                )
        return ListFilesResult(path=self._rel(directory), entries=entries, truncated=truncated)

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> FileContent:
        target = self.resolve(path)
        lines = self._read_bytes(target).decode("utf-8", errors="replace").splitlines()
        total = len(lines)
        rel = self._rel(target)
        if end_line is not None and end_line < start_line:
            raise WorkspaceError("invalid_argument", "end_line must be >= start_line.")
        if total == 0:
            return FileContent(
                path=rel, start_line=1, end_line=0, total_lines=0, content="", truncated=False
            )
        if start_line > total:
            raise WorkspaceError("invalid_argument", "start_line is beyond the end of the file.")
        end = min(end_line or total, total)
        truncated = end - start_line + 1 > MAX_LINES_PER_READ
        if truncated:
            end = start_line + MAX_LINES_PER_READ - 1
        content = "\n".join(lines[start_line - 1 : end])
        return FileContent(
            path=rel,
            start_line=start_line,
            end_line=end,
            total_lines=total,
            content=content,
            truncated=truncated,
        )

    def search(self, query: str, max_results: int, case_sensitive: bool = False) -> SearchResult:
        """Literal substring search. Deliberately not regex, to rule out ReDoS."""
        if not query.strip() or len(query) > MAX_QUERY_LENGTH:
            raise WorkspaceError("invalid_argument", "Query must be 1-200 characters.")
        needle = query if case_sensitive else query.lower()
        files, files_truncated = self._collect_files()
        matches: list[SearchMatch] = []
        scanned = 0
        truncated = files_truncated
        for file in files:
            try:
                text = self._read_bytes(file).decode("utf-8", errors="replace")
            except WorkspaceError:
                continue
            scanned += 1
            for number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if needle not in haystack:
                    continue
                if len(matches) >= max_results:
                    truncated = True
                    break
                matches.append(SearchMatch(path=self._rel(file), line=number, text=line[:300]))
            if truncated and len(matches) >= max_results:
                break
        return SearchResult(
            query=query, matches=matches, files_scanned=scanned, truncated=truncated
        )

    def summary(self) -> RepoSummary:
        files, truncated = self._collect_files()
        by_extension: dict[str, int] = {}
        sizes: list[LargeFile] = []
        total = 0
        for file in files:
            size = file.stat().st_size
            total += size
            ext = file.suffix.lower() or "(none)"
            by_extension[ext] = by_extension.get(ext, 0) + 1
            sizes.append(LargeFile(path=self._rel(file), size=size))
        sizes.sort(key=lambda f: f.size, reverse=True)
        return RepoSummary(
            file_count=len(files),
            total_bytes=total,
            by_extension=dict(sorted(by_extension.items(), key=lambda kv: -kv[1])[:15]),
            largest_files=sizes[:5],
            truncated=truncated,
        )

    def propose_patch(self, path: str, new_content: str) -> PatchProposal:
        """Build a unified diff. Never writes. Hashes tie approval to this exact version."""
        target = self.resolve(path)
        if target == self.root:
            raise WorkspaceError("not_a_file", "Path is not a file.")
        new_bytes = new_content.encode("utf-8")
        if len(new_bytes) > self.max_read_bytes:
            raise WorkspaceError("too_large", f"New content exceeds {self.max_read_bytes} bytes.")
        is_new = not target.exists()
        old_bytes = b"" if is_new else self._read_bytes(target)
        rel = self._rel(target)
        diff = "".join(
            difflib.unified_diff(
                old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="/dev/null" if is_new else f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        diff_lines = diff.splitlines()
        base, new = _sha256(old_bytes), _sha256(new_bytes)
        return PatchProposal(
            path=rel,
            is_new_file=is_new,
            has_changes=old_bytes != new_bytes,
            diff=diff,
            additions=sum(1 for d in diff_lines if d.startswith("+") and not d.startswith("+++")),
            deletions=sum(1 for d in diff_lines if d.startswith("-") and not d.startswith("---")),
            base_sha256=base,
            new_sha256=new,
            patch_id=_sha256(f"{rel}\n{base}\n{new}".encode()),
        )
