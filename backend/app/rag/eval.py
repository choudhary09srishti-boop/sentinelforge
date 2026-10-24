"""Retrieval evaluation: eight questions with the source each answer should come from."""

from dataclasses import dataclass

from pydantic import BaseModel

from app.rag.service import Mode, RagService


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_source: str
    expected_page: int | None = None
    expected_section: str | None = None


EVAL_QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        "How do I authenticate requests to the API?", "api_guide.md", None, "Authentication"
    ),
    EvalQuestion(
        "What happens if I upload a file that is too large?", "api_guide.md", None, "Upload"
    ),
    EvalQuestion(
        "Who has to approve a patch before it is applied?", "security_policy.md", None, "approval"
    ),
    EvalQuestion("What should I do first when the service goes down?", "incident_runbook.pdf", 1),
    EvalQuestion("How do I respond to a leaked credential?", "incident_runbook.pdf", 2),
    EvalQuestion("What is the maximum time a tool call may run?", "settings.json"),
    EvalQuestion("Which release added support for TypeScript files?", "release_notes.txt"),
    EvalQuestion("Function that retries a failed request with growing delays", "retry.ts"),
)


class EvalResult(BaseModel):
    question: str
    expected_source: str
    found_rank: int | None
    top_sources: list[str]


class EvalReport(BaseModel):
    mode: str
    top_k: int
    results: list[EvalResult]
    hit_at_1: float
    hit_at_3: float
    mrr: float


def _matches(question: EvalQuestion, source: str, page: int | None, section: str | None) -> bool:
    if source != question.expected_source:
        return False
    if question.expected_page is not None and page != question.expected_page:
        return False
    if question.expected_section is not None:
        return question.expected_section.lower() in (section or "").lower()
    return True


def evaluate(service: RagService, top_k: int = 3, mode: Mode = "hybrid") -> EvalReport:
    results: list[EvalResult] = []
    for item in EVAL_QUESTIONS:
        hits = service.search(item.question, top_k=top_k, mode=mode).hits
        rank = next(
            (
                i
                for i, h in enumerate(hits, start=1)
                if _matches(item, h.citation.source, h.citation.page, h.citation.section)
            ),
            None,
        )
        results.append(
            EvalResult(
                question=item.question,
                expected_source=item.expected_source,
                found_rank=rank,
                top_sources=[h.citation.source for h in hits],
            )
        )
    total = len(results)
    return EvalReport(
        mode=mode,
        top_k=top_k,
        results=results,
        hit_at_1=sum(1 for r in results if r.found_rank == 1) / total,
        hit_at_3=sum(1 for r in results if r.found_rank is not None) / total,
        mrr=sum(1 / r.found_rank for r in results if r.found_rank) / total,
    )
