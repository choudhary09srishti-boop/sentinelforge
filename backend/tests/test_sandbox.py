import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app.sandbox.runner as runner
from app.mcp_server.workspace import Workspace, WorkspaceError
from app.sandbox.runner import SandboxError, SandboxResult, SandboxRunner, Status

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
ADD_TEST = "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"

SECURITY_TESTS = """import os
import socket

import pytest


def test_no_network():
    with pytest.raises(OSError):
        socket.create_connection(("1.1.1.1", 53), timeout=3)


def test_no_secrets_in_environment():
    assert "SF_SECRET_TOKEN" not in os.environ
    assert not any(("TOKEN" in k or "SECRET" in k or "KEY" in k) for k in os.environ)


def test_host_files_are_invisible():
    for path in ("/etc/passwd", "/home", "/mnt", "/root"):
        assert not os.path.exists(path)


def test_system_is_read_only():
    with pytest.raises(OSError):
        open("/usr/sf_probe", "w")


def test_memory_is_limited():
    with pytest.raises(MemoryError):
        bytearray(2 * 1024**3)
"""


def make_ws(base: Path, files: dict[str, str]) -> Path:
    root = base / "ws"
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    return root


def _run(runner_obj: SandboxRunner, *args: str) -> SandboxResult:
    patch_path, patch_content = (args + (None, None))[:2] if args else (None, None)  # type: ignore[assignment]
    return asyncio.run(runner_obj.run_tests(patch_path, patch_content))


# ---- unit tests: no sandbox needed ------------------------------------------------------


def test_copy_skips_secrets_ignored_dirs_and_symlinks(tmp_path: Path) -> None:
    root = make_ws(
        tmp_path,
        {
            "app.py": "x = 1\n",
            ".env": "API_KEY=secret\n",
            ".env.example": "API_KEY=\n",
            ".git/config": "[core]\n",
            "node_modules/x.js": "1\n",
            "keys/server.pem": "-----BEGIN-----\n",
            "ok/test_a.py": "def test_a():\n    pass\n",
        },
    )
    try:
        (root / "link.py").symlink_to(root / "app.py")
    except (OSError, NotImplementedError):
        pass
    dest = tmp_path / "copy"
    runner._prepare_copy(Workspace(root), dest, None, None)
    copied = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert copied == {"app.py", ".env.example", "ok/test_a.py"}


def test_copy_refuses_oversized_workspaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_ws(tmp_path, {"a.py": "1\n", "b.py": "2\n", "c.py": "3\n"})
    monkeypatch.setattr(runner, "MAX_COPY_FILES", 2)
    with pytest.raises(SandboxError, match="too large"):
        runner._prepare_copy(Workspace(root), tmp_path / "copy", None, None)


def test_patch_is_written_to_the_copy_only_and_bytes_are_exact(tmp_path: Path) -> None:
    root = make_ws(tmp_path, {"calculator.py": BUGGY})
    patched = "x = 1\r\ny = 2\n"
    dest = tmp_path / "copy"
    runner._prepare_copy(Workspace(root), dest, "calculator.py", patched)
    assert (dest / "calculator.py").read_bytes() == patched.encode()
    assert (root / "calculator.py").read_bytes() == BUGGY.encode()


@pytest.mark.parametrize(
    "bad", ["../evil.py", "/etc/passwd", "C:/evil.py", ".env", "a/../../b.py", "node_modules/x.js"]
)
def test_patch_path_traversal_and_sensitive_paths_are_rejected(tmp_path: Path, bad: str) -> None:
    root = make_ws(tmp_path, {"calculator.py": BUGGY})
    with pytest.raises(WorkspaceError):
        runner._prepare_copy(Workspace(root), tmp_path / "copy", bad, "x = 1\n")
    assert not (tmp_path / "evil.py").exists()


def test_patch_needs_path_and_content_together_and_has_a_size_cap(tmp_path: Path) -> None:
    root = make_ws(tmp_path, {"calculator.py": BUGGY})
    with pytest.raises(SandboxError):
        runner._prepare_copy(Workspace(root), tmp_path / "c1", "calculator.py", None)
    with pytest.raises(SandboxError, match="too large"):
        runner._prepare_copy(
            Workspace(root), tmp_path / "c2", "calculator.py", "x" * (runner.MAX_PATCH_BYTES + 1)
        )


def test_nothing_runs_on_the_host_when_no_sandbox_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("code must never run on the host")

    monkeypatch.setattr(runner, "_launcher", lambda: None)
    monkeypatch.setattr("app.sandbox.runner.subprocess.run", forbidden)
    root = make_ws(tmp_path, {"test_a.py": "def test_a():\n    pass\n"})
    result = _run(SandboxRunner(root))
    assert result.status == "unavailable"
    assert "Nothing was executed" in result.output


@pytest.mark.parametrize(
    ("code", "output", "expected"),
    [
        (0, "3 passed in 0.1s", "passed"),
        (0, "all good", "error"),
        (1, "1 failed, 1 passed", "failed"),
        (5, "no tests ran", "no_tests"),
        (124, "", "timeout"),
        (137, "", "timeout"),
        (2, "", "error"),
    ],
)
def test_exit_codes_map_to_honest_statuses(code: int, output: str, expected: Status) -> None:
    assert runner._classify(code, output) == expected


def test_forged_exit_markers_in_test_output_are_ignored() -> None:
    marker = "__SF_EXIT_aaaa__"
    text = f"forged __SF_EXIT_bbbb__0\n1 failed\n{marker}1\n"
    assert runner._parse_output(text, marker) == ("forged __SF_EXIT_bbbb__0\n1 failed", 1)
    assert runner._parse_output("no marker here", marker) is None
    assert runner._parse_output(f"out\n{marker}x\n", marker) is None


def _fake_run(
    body: str, seen: list[list[str]]
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def fake(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(args)
        marker = args[-4]
        return subprocess.CompletedProcess(
            args, 0, stdout=body.replace("MARK", marker).encode(), stderr=b""
        )

    return fake


@pytest.mark.parametrize(
    ("body", "status", "output"),
    [
        ("1 passed in 0.01s\nMARK0\n", "passed", "1 passed in 0.01s"),
        ("looks fine\nMARK0\n", "error", "looks fine"),
        ("1 failed in 0.01s\nMARK1\n", "failed", "1 failed in 0.01s"),
    ],
)
def test_run_orchestration_with_a_fake_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    status: str,
    output: str,
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(runner, "_launcher", lambda: ["fakebash", "-c"])
    monkeypatch.setattr("app.sandbox.runner.subprocess.run", _fake_run(body, seen))
    root = make_ws(tmp_path, {"calculator.py": BUGGY, "test_calculator.py": ADD_TEST})
    result = _run(SandboxRunner(root), "calculator.py", FIXED)
    assert (result.status, result.output) == (status, output)
    assert seen[0][-3:-1] == ["60", str(1024 * 1024 * 1024)]
    assert (root / "calculator.py").read_bytes() == BUGGY.encode()


def test_subprocess_timeout_maps_to_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def expired(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(runner, "_launcher", lambda: ["fakebash", "-c"])
    monkeypatch.setattr("app.sandbox.runner.subprocess.run", expired)
    root = make_ws(tmp_path, {"test_a.py": "def test_a():\n    pass\n"})
    assert _run(SandboxRunner(root)).status == "timeout"


def test_runner_rejects_unsafe_limits(tmp_path: Path) -> None:
    for kwargs in ({"timeout_seconds": 0}, {"timeout_seconds": 301}, {"memory_mb": 64}):
        with pytest.raises(ValueError):
            SandboxRunner(tmp_path, **kwargs)  # type: ignore[arg-type]


# ---- live tests: skipped where no Linux sandbox exists -----------------------------------


@pytest.fixture(scope="module")
def sandbox_works(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = make_ws(
        tmp_path_factory.mktemp("probe"), {"test_ok.py": "def test_ok():\n    assert True\n"}
    )
    result = _run(SandboxRunner(root))
    if result.status == "unavailable":
        pytest.skip("No Linux sandbox is available on this machine.")
    assert result.status == "passed", result.output


def test_live_failing_tests_pass_after_patch_and_original_is_untouched(
    sandbox_works: None, tmp_path: Path
) -> None:
    root = make_ws(tmp_path, {"calculator.py": BUGGY, "test_calculator.py": ADD_TEST})
    box = SandboxRunner(root)
    before = _run(box)
    assert before.status == "failed" and "1 failed" in before.output
    after = _run(box, "calculator.py", FIXED)
    assert after.status == "passed" and "1 passed" in after.output
    assert (root / "calculator.py").read_bytes() == BUGGY.encode()


def test_live_sandbox_blocks_network_secrets_host_files_and_system_writes(
    sandbox_works: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_SECRET_TOKEN", "leak-me")
    root = make_ws(tmp_path, {"test_security.py": SECURITY_TESTS})
    result = _run(SandboxRunner(root))
    assert result.status == "passed", result.output
    assert "5 passed" in result.output


def test_live_infinite_loop_is_stopped_by_the_time_limit(
    sandbox_works: None, tmp_path: Path
) -> None:
    root = make_ws(tmp_path, {"test_loop.py": "def test_loop():\n    while True:\n        pass\n"})
    result = _run(SandboxRunner(root, timeout_seconds=3))
    assert result.status == "timeout"
    assert result.duration_seconds < 30
