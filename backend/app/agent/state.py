"""The single state object every agent step reads from and writes to."""

from typing import Literal, TypedDict

TestStatus = Literal["not_run", "passed", "failed"]


class AgentState(TypedDict):
    task: str
    analysis: str
    search_terms: list[str]
    plan: list[str]
    evidence: list[str]
    inspected_files: list[str]
    diff: str
    test_status: TestStatus
    test_output: str
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
        test_status="not_run",
        test_output="",
        risks=[],
        attempts=0,
        review_ok=False,
        report="",
    )
