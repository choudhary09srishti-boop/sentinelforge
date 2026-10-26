import asyncio

from app.agent.graph import build_graph
from app.agent.state import AgentState, new_state


def _path(max_attempts: int) -> list[str]:
    async def go() -> list[str]:
        graph = build_graph(max_attempts)
        return [
            name
            async for chunk in graph.astream(new_state("fix bug"), stream_mode="updates")
            for name in chunk
        ]

    return asyncio.run(go())


def _final(max_attempts: int) -> AgentState:
    final: AgentState = asyncio.run(build_graph(max_attempts).ainvoke(new_state("fix bug")))
    return final


def test_graph_runs_all_steps_and_retries_once() -> None:
    path = _path(2)
    assert path[0] == "analyze" and path[-1] == "report"
    assert path.count("patch") == 2 and path.count("review") == 2
    assert path[:5] == ["analyze", "plan", "retrieve", "inspect", "patch"]


def test_retry_loop_is_capped_by_max_attempts() -> None:
    path = _path(1)
    assert path.count("patch") == 1
    final = _final(1)
    assert final["review_ok"] is False
    assert "unresolved" in final["report"]


def test_report_never_claims_tests_passed_when_not_run() -> None:
    final = _final(2)
    assert final["test_status"] == "not_run"
    assert "Tests were not executed." in final["report"]
    assert "passed" not in final["report"].lower()
