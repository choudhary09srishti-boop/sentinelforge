from app.rag.chunker import chunk_lines
from app.rag.parsers import SourceLine


def make(count: int, width: int = 50, page: int | None = None, section: str | None = None):
    return [
        SourceLine(text=("x" * width), line_no=i, page=page, section=section)
        for i in range(1, count + 1)
    ]


def test_chunks_respect_the_character_budget() -> None:
    chunks = chunk_lines(make(100), max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_line_ranges_are_valid_and_cover_the_document() -> None:
    chunks = chunk_lines(make(60), max_chars=300)
    assert all(1 <= c.start_line <= c.end_line <= 60 for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 60


def test_pages_never_share_a_chunk() -> None:
    lines = make(3, page=1) + [SourceLine(text="y" * 50, line_no=i, page=2) for i in range(1, 4)]
    chunks = chunk_lines(lines, max_chars=1000)
    assert [c.page for c in chunks] == [1, 2]


def test_section_change_starts_a_new_chunk_once_big_enough() -> None:
    lines = make(5, width=60, section="A") + make(5, width=60, section="B")
    chunks = chunk_lines(lines, max_chars=1000, min_chars=100)
    assert [c.section for c in chunks] == ["A", "B"]


def test_a_very_long_line_is_split_and_keeps_its_line_number() -> None:
    chunks = chunk_lines([SourceLine(text="z" * 2500, line_no=7)], max_chars=1000)
    assert len(chunks) == 3
    assert {(c.start_line, c.end_line) for c in chunks} == {(7, 7)}


def test_blank_input_produces_no_chunks() -> None:
    blanks = [SourceLine(text="  ", line_no=i) for i in range(1, 4)]
    assert chunk_lines(blanks) == []
