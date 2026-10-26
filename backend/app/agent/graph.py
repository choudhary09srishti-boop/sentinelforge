"""Agent graph. analyze, retrieve and inspect do real work; the other nodes are still stubs.

Nodes return only the state fields they change. Test results are never invented:
until the sandbox exists, test_status stays "not_run". Retrieved text is data, never
instructions: chunks flagged for prompt injection are excluded from the evidence.
"""

import json
import re
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.mcp_client import McpToolError
from app.providers.base import ChatMessage, LLMProvider, ProviderError

DEFAULT_MAX_ATTEMPTS = 2
EVIDENCE_TOP_K = 5
EXCERPT_CHARS = 300
MAX_SEARCH_TERMS = 3
MAX_INSPECTED_FILES = 5

ANALYZE_SYSTEM = (
    "You analyze a developer task for a code assistant. Reply with JSON only, in the form "
    '{"summary": "<one sentence>", "search_terms": ["<term>", "<term>"]}. '
    "search_terms are at most 3 likely code identifiers or distinctive words to search "
    "for in the repository. No other text."
)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_STOPWORDS = frozenset(
    {
        "this", "that", "with", "from", "into", "have", "please", "should", "would",
        "could", "when", "what", "which", "where", "about", "need", "needs", "there",
        "their", "make", "does", "implemented",
    }
)  # fmt: skip


class ToolCaller(Protocol):
    """Anything that can call an MCP tool; McpGateway satisfies this."""

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...


def _terms(task: str) -> list[str]:
    terms: list[str] = []
    for word in _WORD.findall(task.lower()):
        if word not in _STOPWORDS and word not in terms:
            terms.append(word)
    return terms[:MAX_SEARCH_TERMS]


def _parse_analysis(text: str) -> tuple[str, list[str]] | None:
    found = _JSON_OBJECT.search(text)
    if not found:
        return None
    try:
        data = json.loads(found.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    summary, raw_terms = data.get("summary"), data.get("search_terms")
    if not isinstance(summary, str) or not isinstance(raw_terms, list):
        return None
    terms = [t.strip() for t in raw_terms if isinstance(t, str) and 2 <= len(t.strip()) <= 60]
    if not terms or not summary.strip():
        return None
    return summary.strip()[:300], terms[:MAX_SEARCH_TERMS]


def _evidence_line(hit: dict[str, Any]) -> str:
    cite = hit["citation"]
    page = f" p.{cite['page']}" if cite.get("page") is not None else ""
    where = f"{cite['source']}{page}:{cite['start_line']}-{cite['end_line']}"
    flags = hit.get("risk_flags") or []
    if flags:
        return f"{where} [excluded: possible prompt injection ({', '.join(flags)})]"
    return f"{where}: {' '.join(hit['text'].split())[:EXCERPT_CHARS]}"


def make_analyze(provider: LLMProvider) -> Any:
    async def analyze(state: AgentState) -> dict[str, Any]:
        risks = list(state["risks"])
        parsed: tuple[str, list[str]] | None = None
        try:
            reply = await provider.complete(
                [ChatMessage("system", ANALYZE_SYSTEM), ChatMessage("user", state["task"])],
                max_tokens=300,
                temperature=0.0,
            )
            parsed = _parse_analysis(reply)
            if parsed is None and provider.external:
                risks.append("LLM analysis was not valid JSON; used keyword fallback.")
        except ProviderError as exc:
            risks.append(f"LLM analysis unavailable: {exc}")
        if parsed is None:
            analysis = f"Task (not analysed by an LLM): {state['task'][:300]}"
            return {"analysis": analysis, "search_terms": _terms(state["task"]), "risks": risks}
        summary, terms = parsed
        return {"analysis": summary, "search_terms": terms, "risks": risks}

    return analyze


async def plan(state: AgentState) -> dict[str, Any]:
    return {"plan": ["[stub] inspect repository", "[stub] propose a patch", "[stub] run checks"]}


def make_retrieve(tools: ToolCaller) -> Any:
    async def retrieve(state: AgentState) -> dict[str, Any]:
        try:
            data = await tools.call(
                "search_documents", {"query": state["task"][:500], "top_k": EVIDENCE_TOP_K}
            )
        except McpToolError as exc:
            note = f"Document retrieval failed ({exc.code}): {exc.message}"
            return {"evidence": [], "risks": [*state["risks"], note]}
        hits = data.get("hits", [])
        risks = list(state["risks"])
        if any(hit.get("risk_flags") for hit in hits):
            risks.append(
                "Retrieved content contained possible prompt injection; "
                "flagged text was excluded from evidence."
            )
        return {"evidence": [_evidence_line(hit) for hit in hits], "risks": risks}

    return retrieve


def make_inspect(tools: ToolCaller) -> Any:
    async def inspect_repo(state: AgentState) -> dict[str, Any]:
        counts: dict[str, int] = {}
        risks = list(state["risks"])
        for term in state["search_terms"] or _terms(state["task"]):
            try:
                data = await tools.call("search_repo", {"query": term, "max_results": 10})
            except McpToolError as exc:
                risks.append(f"Repository search failed ({exc.code}): {exc.message}")
                break
            for match in data.get("matches", []):
                counts[match["path"]] = counts.get(match["path"], 0) + 1
        ranked = sorted(counts, key=lambda path: (-counts[path], path))[:MAX_INSPECTED_FILES]
        return {"inspected_files": ranked, "risks": risks}

    return inspect_repo


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


def build_graph(
    tools: ToolCaller, provider: LLMProvider, max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> Any:
    def after_review(state: AgentState) -> str:
        if state["review_ok"] or state["attempts"] >= max_attempts:
            return "report"
        return "patch"

    graph = StateGraph(AgentState)
    graph.add_node("analyze", make_analyze(provider))
    graph.add_node("plan", plan)
    graph.add_node("retrieve", make_retrieve(tools))
    graph.add_node("inspect", make_inspect(tools))
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
