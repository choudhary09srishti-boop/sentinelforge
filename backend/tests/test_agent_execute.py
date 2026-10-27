import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import MAX_TEST_OUTPUT_CHARS, SandboxPort, build_graph
from app.agent.state import AgentState, new_state
from app.providers.base import ChatMessage
from app.sandbox.runner import SandboxResult, SandboxRunner, Status

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
ADD_TEST = "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
ANALYZE = '{"summary": "Fix add", "search_terms": ["add"]}'
PLAN = '{"plan": ["Fix the add function", "Run the tests"]}'
REPLY = f"```python\n{FIXED}```"
READ = {
    "path": "calculator.py",
    "start_line": 1,
    "end_line": 2,
    "total_lines": 2,
    "content": BUGGY.rstrip("\n"),
    "truncated": False,
}
DIFF = "--- a/calculator.py\n+++ b/calculator.py\n@@\n-    return a - b\n+    return a + b\n"


class Tools:
    def __init__(self, has_changes: bool = True) -> None:
        self.has_changes = has_changes

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if tool == "search_repo":
            return {"matches": [{"path": "calculator.py", "line": 1, "text": "x"}]}
        if tool == "read_repo_file":
            return READ
        if tool == "propose_patch":
            return {"has_changes": self.has_changes, "diff": DIFF, "patch_id": "pid"}
        return {"hits": []}


class Provider:
    name = "fake"
    external = False

    def __init__(self) -> None:
        self.replies = [ANALYZE, PLAN, REPLY]
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class FakeSandbox:
    def __init__(self, status: Status, output: str = "") -> None:
        self.status = status
        self.output = output
        self.calls: list[tuple[str | None, str | None]] = []

    async def run_tests(
        self, patch_path: str | None = None, patch_content: str | None = None
    ) -> SandboxResult:
        self.calls.append((patch_path, patch_content))
        return SandboxResult(status=self.status, exit_code=0, output=self.output)


def _run(sandbox: SandboxPort | None, *, has_changes: bool = True) -> AgentState:
    graph = build_graph(Tools(has_changes), Provider(), 1, sandbox)
    final: AgentState = asyncio.run(graph.ainvoke(new_state("fix add")))
    return final


def test_passing_tests_are_reported_with_the_patch_that_was_tested() -> None:
    box = FakeSandbox("passed", "1 passed in 0.01s")
    final = _run(box)
    assert box.calls == [("calculator.py", FIXED)]
    assert final["test_status"] == "passed" and final["test_output"] == "1 passed in 0.01s"
    assert "Tests passed in the isolated sandbox." in final["report"]
    assert not any("did not complete" in risk for risk in final["risks"])


def test_failing_tests_are_reported_as_failed() -> None:
    final = _run(FakeSandbox("failed", "1 failed in 0.01s"))
    assert final["test_status"] == "failed"
    assert "Tests failed in the isolated sandbox." in final["report"]


def test_unavailable_sandbox_is_never_reported_as_a_pass() -> None:
    final = _run(FakeSandbox("unavailable", "Nothing was executed."))
    assert final["test_status"] == "unavailable"
    assert any("unavailable" in risk for risk in final["risks"])
    assert "no isolated sandbox is available" in final["report"]
    assert "passed" not in final["report"].lower()


def test_no_patch_means_the_sandbox_is_never_called() -> None:
    box = FakeSandbox("passed", "1 passed")
    final = _run(box, has_changes=False)
    assert box.calls == []
    assert final["test_status"] == "not_run"
    assert "nothing to test" in final["test_output"]


def test_without_a_sandbox_nothing_is_executed() -> None:
    final = _run(None)
    assert final["test_status"] == "not_run"
    assert "No sandbox is configured" in final["test_output"]
    assert "Tests were not executed." in final["report"]


def test_long_test_output_is_capped() -> None:
    final = _run(FakeSandbox("failed", "x" * (MAX_TEST_OUTPUT_CHARS * 3)))
    assert len(final["test_output"]) == MAX_TEST_OUTPUT_CHARS


def test_live_agent_tests_its_patch_in_the_real_sandbox(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "calculator.py").write_bytes(BUGGY.encode())
    (root / "test_calculator.py").write_bytes(ADD_TEST.encode())
    runner = SandboxRunner(root)
    if asyncio.run(runner.run_tests()).status == "unavailable":
        pytest.skip("No Linux sandbox is available on this machine.")
    final = _run(runner)
    assert final["test_status"] == "passed" and "1 passed" in final["test_output"]
    assert (root / "calculator.py").read_bytes() == BUGGY.encode()
