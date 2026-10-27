"""The single state object every agent step reads from and writes to."""

from typing import Literal, TypedDict

TestStatus = Literal["not_run", "passed", "failed", "no_tests", "error", "timeout", "unavailable"]


class AgentState(TypedDict):
    task: str
    analysis: str
    search_terms: list[str]
    plan: list[str]
    evidence: list[str]
    inspected_files: list[str]
    diff: str
    patch_path: str
    patch_id: str
    patch_content: str
    test_status: TestStatus
    test_output: str
    feedback: str
    risks: list[str]
    attempts: int
    review_ok: bool
    report: str


def new_state(task: str) -> AgentState:
    return AgentState(
        task=task,
        analysis="",
        search_terms=[],
        plan=[],
        evidence=[],
        inspected_files=[],
        diff="",
        patch_path="",
        patch_id="",
        patch_content="",
        test_status="not_run",
        test_output="",
        feedback="",
        risks=[],
        attempts=0,
        review_ok=False,
        report="",
    )
