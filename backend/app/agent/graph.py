"""Agent graph skeleton. Every node is a stub for now; later steps replace them one by one.

Nodes return only the state fields they change. Test results are never invented:
until the sandbox exists, test_status stays "not_run".
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState

DEFAULT_MAX_ATTEMPTS = 2


async def analyze(state: AgentState) -> dict[str, Any]:
    return {"analysis": f"[stub] analysis of: {state['task']}"}


async def plan(state: AgentState) -> dict[str, Any]:
    return {"plan": ["[stub] inspect repository", "[stub] propose a patch", "[stub] run checks"]}


async def retrieve(state: AgentState) -> dict[str, Any]:
    return {"evidence": []}


async def inspect_repo(state: AgentState) -> dict[str, Any]:
    return {"inspected_files": []}


async def patch(state: AgentState) -> dict[str, Any]:
    return {"diff": "", "attempts": state["attempts"] + 1}


async def execute(state: AgentState) -> dict[str, Any]:
    return {
        "test_status": "not_run",
        "test_output": "Sandbox not available yet; nothing was executed.",
    }


async def review(state: AgentState) -> dict[str, Any]:
    # Stub: pretend the first attempt needs a second pass so the retry loop is visible.
    return {"review_ok": state["attempts"] >= 2}


async def report(state: AgentState) -> dict[str, Any]:
    lines = [f"Task: {state['task']}", f"Attempts: {state['attempts']}"]
    if state["test_status"] == "not_run":
        lines.append("Tests were not executed.")
    else:
        lines.append(f"Tests {state['test_status']}.")
    if not state["review_ok"]:
        lines.append("Self-review did not pass; the proposal is unresolved.")
    return {"report": "\n".join(lines)}


def build_graph(max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Any:
    def after_review(state: AgentState) -> str:
        if state["review_ok"] or state["attempts"] >= max_attempts:
            return "report"
        return "patch"

    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("inspect", inspect_repo)
    graph.add_node("patch", patch)
    graph.add_node("execute", execute)
    graph.add_node("review", review)
    graph.add_node("report", report)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "inspect")
    graph.add_edge("inspect", "patch")
    graph.add_edge("patch", "execute")
    graph.add_edge("execute", "review")
    graph.add_conditional_edges("review", after_review, {"patch": "patch", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile()
