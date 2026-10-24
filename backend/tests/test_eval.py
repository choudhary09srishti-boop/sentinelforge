from pathlib import Path

from app.rag.eval import EVAL_QUESTIONS, evaluate
from app.rag.service import RagService

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "docs"


def test_there_are_eight_questions_and_every_expected_source_exists() -> None:
    assert len(EVAL_QUESTIONS) == 8
    for question in EVAL_QUESTIONS:
        assert (SAMPLES / question.expected_source).is_file(), question.expected_source


def test_harness_reports_ranks_and_metrics_over_the_sample_corpus(rag_service: RagService) -> None:
    for path in sorted(SAMPLES.iterdir()):
        rag_service.ingest(path.name, path.read_bytes())
    report = evaluate(rag_service, top_k=3, mode="hybrid")
    assert len(report.results) == 8
    assert 0.0 <= report.hit_at_1 <= report.hit_at_3 <= 1.0
    assert all(len(r.top_sources) <= 3 for r in report.results)
