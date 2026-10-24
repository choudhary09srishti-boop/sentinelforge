from pathlib import Path

import pytest

from app.rag.errors import RagError
from app.rag.parsers import parse_document

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "docs"


def test_markdown_lines_carry_heading_sections() -> None:
    lines = parse_document(".md", b"# Title\nintro\n## Part A\nbody a\n")
    assert [(ln.line_no, ln.section) for ln in lines] == [
        (1, "Title"),
        (2, "Title"),
        (3, "Part A"),
        (4, "Part A"),
    ]


def test_python_lines_carry_definition_names() -> None:
    lines = parse_document(
        ".py", b"import os\n\ndef alpha():\n    pass\n\nclass Beta:\n    x = 1\n"
    )
    assert lines[0].section is None
    assert lines[3].section == "alpha"
    assert lines[6].section == "Beta"


def test_typescript_and_javascript_definitions() -> None:
    ts = parse_document(".ts", b"export async function load() {\n  return 1;\n}\n")
    js = parse_document(".js", b"class Bucket {\n}\nconst helper = () => 1;\n")
    assert ts[1].section == "load"
    assert [ln.section for ln in js] == ["Bucket", "Bucket", "helper"]


def test_json_is_normalised_and_keyed_by_top_level_key() -> None:
    lines = parse_document(".json", b'{"alpha":1,"beta":{"x":2}}')
    sections = {ln.section for ln in lines}
    assert {"alpha", "beta"} <= sections
    assert lines[0].text == "{"


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(RagError) as exc:
        parse_document(".json", b"{not json")
    assert exc.value.code == "parse_error"


def test_plain_text_has_no_sections() -> None:
    lines = parse_document(".txt", b"one\ntwo\n")
    assert [ln.section for ln in lines] == [None, None]


def test_pdf_lines_carry_page_numbers() -> None:
    lines = parse_document(".pdf", (SAMPLES / "incident_runbook.pdf").read_bytes())
    assert {ln.page for ln in lines} == {1, 2}
    assert any("outage" in ln.text.lower() for ln in lines if ln.page == 1)
    assert any("leaked" in ln.text.lower() for ln in lines if ln.page == 2)


@pytest.mark.parametrize("data", [b"\x00\x01binary\x00", b"\xff\xfe\xfa" * 200])
def test_binary_or_invalid_text_is_rejected(data: bytes) -> None:
    with pytest.raises(RagError) as exc:
        parse_document(".txt", data)
    assert exc.value.code == "parse_error"


def test_corrupt_pdf_is_rejected_without_leaking_details() -> None:
    with pytest.raises(RagError) as exc:
        parse_document(".pdf", b"%PDF-1.4 this is not really a pdf")
    assert exc.value.code == "parse_error"
