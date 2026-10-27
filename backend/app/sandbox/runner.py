"""Run the project's tests against a proposed patch inside a throwaway, isolated sandbox.

The workspace is copied to a temporary directory, the patch is applied to that copy only,
and pytest runs inside bubblewrap: no network, read-only system, empty environment, CPU /
memory / process / file-size limits and a hard time limit. On Windows the sandbox is reached
through WSL. If no isolation is available, nothing is executed on the host.

Limit: a malicious patch could make the tests exit early, so a result is evidence for the
human reviewer, never a substitute for reading the diff.
"""

import asyncio
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.mcp_server.workspace import IGNORED_DIRS, Workspace, WorkspaceError, is_sensitive

Status = Literal["passed", "failed", "no_tests", "error", "timeout", "unavailable"]

ISOLATION = (
    "bubblewrap: no network, read-only system, empty environment, resource limits, throwaway copy"
)
MAX_COPY_FILES = 2000
MAX_COPY_BYTES = 20 * 1024 * 1024
MAX_PATCH_BYTES = 200_000
TIMEOUT_EXIT_CODES = frozenset({124, 137, 152})
_PASSED = re.compile(r"\b\d+ passed\b")

# Arguments: $1 source dir, $2 exit marker, $3 time limit (s), $4 memory limit (bytes),
# $5 max output bytes. Exit 70-72 mean the sandbox itself could not start.
_SCRIPT = r"""
SRC="$1"; MARK="$2"; TIMEOUT="$3"; MEM="$4"; MAXOUT="$5"
WS=$(mktemp -d /tmp/sf_sbx.XXXXXX) || exit 70
trap 'rm -rf "$WS"' EXIT
cp -r "$SRC/." "$WS/" || exit 71
BW=(bwrap --unshare-all --die-with-parent --new-session
  --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64
  --proc /proc --dev /dev --tmpfs /tmp
  --bind "$WS" /work --chdir /work
  --clearenv --setenv PATH /usr/bin --setenv HOME /tmp --setenv PYTHONDONTWRITEBYTECODE 1)
"${BW[@]}" true >/dev/null 2>&1 || exit 72
"${BW[@]}" prlimit --as="$MEM" --cpu="$TIMEOUT" --nproc=512 --fsize=10485760 --nofile=256 \
  timeout -k 2 "$TIMEOUT" python3 -m pytest -q -p no:cacheprovider 2>&1 | head -c "$MAXOUT"
RC=${PIPESTATUS[0]}
echo
echo "$MARK$RC"
"""


class SandboxError(Exception):
    """The sandbox could not be prepared (safe to show to the user)."""


class SandboxResult(BaseModel):
    status: Status
    exit_code: int | None = None
    output: str = ""
    duration_seconds: float = 0.0
    isolation: str = ISOLATION


def _launcher() -> list[str] | None:
    if os.name == "nt":
        return ["wsl", "-e", "bash", "-c"] if shutil.which("wsl") else None
    if shutil.which("bash") and shutil.which("bwrap"):
        return ["bash", "-c"]
    return None


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) != 2 or drive[1] != ":":
        raise SandboxError("Only local drive paths can be used with the sandbox.")
    return f"/mnt/{drive[0].lower()}{resolved.as_posix()[2:]}"


def _copy_workspace(root: Path, dest: Path) -> None:
    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        base = Path(dirpath)
        for name in sorted(filenames):
            source = base / name
            if source.is_symlink() or is_sensitive(name):
                continue
            files += 1
            total += source.stat().st_size
            if files > MAX_COPY_FILES or total > MAX_COPY_BYTES:
                raise SandboxError("The workspace is too large to copy into the sandbox.")
            target = dest / base.relative_to(root) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def _prepare_copy(
    workspace: Workspace, dest: Path, patch_path: str | None, patch_content: str | None
) -> None:
    """Copy the workspace, then write the patch into the COPY. The original is never touched."""
    if (patch_path is None) != (patch_content is None):
        raise SandboxError("patch_path and patch_content must be given together.")
    if not workspace.available:
        raise SandboxError("The workspace is not available.")
    _copy_workspace(workspace.root, dest)
    if patch_path is not None and patch_content is not None:
        data = patch_content.encode("utf-8")
        if len(data) > MAX_PATCH_BYTES:
            raise SandboxError("The patch content is too large.")
        target = dest / workspace.resolve(patch_path).relative_to(workspace.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _parse_output(text: str, marker: str) -> tuple[str, int] | None:
    """Split the run output from the exit code. The marker is random per run and the last
    occurrence wins, so test output cannot forge a result."""
    index = text.rfind(marker)
    if index == -1:
        return None
    tail = text[index + len(marker) :].strip().split()
    if not tail or not tail[0].isdigit():
        return None
    return text[:index].rstrip(), int(tail[0])


def _classify(code: int, output: str) -> Status:
    if code in TIMEOUT_EXIT_CODES:
        return "timeout"
    if code == 0:
        # Exit code 0 alone is not enough: pytest must also have reported passed tests.
        return "passed" if _PASSED.search(output) else "error"
    if code == 1:
        return "failed"
    if code == 5:
        return "no_tests"
    return "error"


class SandboxRunner:
    def __init__(
        self,
        workspace_root: Path,
        *,
        timeout_seconds: int = 60,
        memory_mb: int = 1024,
        max_output_bytes: int = 20_000,
    ) -> None:
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 1 and 300.")
        if not 128 <= memory_mb <= 4096:
            raise ValueError("memory_mb must be between 128 and 4096.")
        if not 1_000 <= max_output_bytes <= 200_000:
            raise ValueError("max_output_bytes must be between 1000 and 200000.")
        self._workspace = Workspace(workspace_root)
        self._timeout = timeout_seconds
        self._memory_mb = memory_mb
        self._max_output = max_output_bytes

    async def run_tests(
        self, patch_path: str | None = None, patch_content: str | None = None
    ) -> SandboxResult:
        return await asyncio.to_thread(self._run_sync, patch_path, patch_content)

    def _run_sync(self, patch_path: str | None, patch_content: str | None) -> SandboxResult:
        started = time.monotonic()

        def done(status: Status, output: str, code: int | None = None) -> SandboxResult:
            elapsed = round(time.monotonic() - started, 2)
            return SandboxResult(
                status=status, exit_code=code, output=output, duration_seconds=elapsed
            )

        launcher = _launcher()
        if launcher is None:
            return done(
                "unavailable",
                "No Linux sandbox is available (needs WSL on Windows, or bash and "
                "bubblewrap on Linux). Nothing was executed.",
            )
        tmp = Path(tempfile.mkdtemp(prefix="sf_sbx_"))
        try:
            try:
                _prepare_copy(self._workspace, tmp / "ws", patch_path, patch_content)
                source = _to_wsl_path(tmp / "ws") if os.name == "nt" else str(tmp / "ws")
            except (SandboxError, WorkspaceError, OSError) as exc:
                reason = getattr(exc, "message", str(exc))
                return done("error", f"Sandbox preparation failed: {reason}")
            marker = f"__SF_EXIT_{secrets.token_hex(8)}__"
            limits = [str(self._timeout), str(self._memory_mb * 1024 * 1024), str(self._max_output)]
            try:
                proc = subprocess.run(  # noqa: S603
                    [*launcher, _SCRIPT, "bash", source, marker, *limits],
                    capture_output=True,
                    timeout=self._timeout + 30,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return done("timeout", "The sandbox did not finish in time and was stopped.")
            except OSError:
                return done(
                    "unavailable", "The sandbox could not be started. Nothing was executed."
                )
            text = (proc.stdout + proc.stderr).replace(b"\x00", b"").decode("utf-8", "replace")
            parsed = _parse_output(text, marker)
            if parsed is None:
                return done(
                    "unavailable",
                    "The sandbox could not start (is bubblewrap installed and permitted?). "
                    "Nothing was executed.\n" + text[-500:].strip(),
                )
            body, code = parsed
            if len(body.encode("utf-8")) >= self._max_output:
                body += "\n[output truncated]"
            return done(_classify(code, body), body, code)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
