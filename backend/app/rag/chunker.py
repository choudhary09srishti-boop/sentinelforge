"""Pack source lines into chunks that keep exact line, page and section citations."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from app.rag.parsers import SourceLine


@dataclass(frozen=True)
class ChunkDraft:
    text: str
    start_line: int
    end_line: int
    page: int | None
    section: str | None


def _split_long(line: SourceLine, max_chars: int) -> Iterator[SourceLine]:
    if len(line.text) <= max_chars:
        yield line
        return
    for start in range(0, len(line.text), max_chars):
        piece = line.text[start : start + max_chars]
        yield SourceLine(text=piece, line_no=line.line_no, page=line.page, section=line.section)


def chunk_lines(
    lines: Sequence[SourceLine], max_chars: int = 1000, min_chars: int = 150
) -> list[ChunkDraft]:
    """New chunk on page change, on section change (once big enough), or when over budget.

    A chunk is labelled with the section of its last line, where most of its text lies.
    """
    chunks: list[ChunkDraft] = []
    buf: list[SourceLine] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        text = "\n".join(line.text for line in buf).strip()
        if buf and text:
            first, last = buf[0], buf[-1]
            chunks.append(ChunkDraft(text, first.line_no, last.line_no, first.page, last.section))
        buf, size = [], 0

    for original in lines:
        for piece in _split_long(original, max_chars):
            if buf:
                boundary = piece.page != buf[0].page or (
                    piece.section != buf[-1].section and size >= min_chars
                )
                if boundary or size + len(piece.text) + 1 > max_chars:
                    carry = buf[-1] if not boundary and len(buf) > 1 else None
                    flush()
                    if carry and len(carry.text) < 200:  # one line of overlap for context
                        buf, size = [carry], len(carry.text) + 1
            buf.append(piece)
            size += len(piece.text) + 1
    flush()
    return chunks
