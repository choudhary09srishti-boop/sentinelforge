"""Turn raw file bytes into numbered source lines carrying page and section metadata."""

import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from pypdf import PdfReader

from app.rag.errors import RagError

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt", ".json", ".py", ".js", ".ts"})
WORKSPACE_TEXT_EXTENSIONS = SUPPORTED_EXTENSIONS - {".pdf"}
MAX_PDF_PAGES = 300

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_PY_DEF = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")
_JS_DEF = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|type)\s+(\w+)"
    r"|^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*="
)
_JSON_KEY = re.compile(r'^  "([^"]+)"\s*:')


@dataclass(frozen=True)
class SourceLine:
    text: str
    line_no: int
    page: int | None = None
    section: str | None = None


def _first_group(pattern: re.Pattern[str]) -> Callable[[str], str | None]:
    def match(line: str) -> str | None:
        found = pattern.match(line)
        return next((g for g in found.groups() if g), None) if found else None

    return match


def _section_matcher(extension: str) -> Callable[[str], str | None] | None:
    if extension in (".md", ".markdown"):
        return _first_group(_MD_HEADING)
    if extension == ".py":
        return _first_group(_PY_DEF)
    if extension in (".js", ".ts"):
        return _first_group(_JS_DEF)
    if extension == ".json":
        return _first_group(_JSON_KEY)
    return None


def _decode(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if "\x00" in text or text.count("\ufffd") > max(10, len(text) // 50):
        raise RagError("parse_error", "File is not valid UTF-8 text.")
    return text


def _pretty_json(text: str) -> str:
    """JSON is normalised to indent=2, so cited line numbers refer to that layout."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (ValueError, RecursionError):
        raise RagError("parse_error", "File is not valid JSON.") from None


def _number_lines(text: str, matcher: Callable[[str], str | None] | None) -> list[SourceLine]:
    lines: list[SourceLine] = []
    section: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if matcher and (found := matcher(line)):
            section = found
        lines.append(SourceLine(text=line, line_no=number, section=section))
    return lines


def _parse_pdf(data: bytes) -> list[SourceLine]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise RagError("parse_error", "Encrypted PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise RagError("too_large", f"PDF has more than {MAX_PDF_PAGES} pages.")
        lines: list[SourceLine] = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            for number, line in enumerate(text.splitlines(), start=1):
                lines.append(SourceLine(text=line, line_no=number, page=page_no))
        return lines
    except RagError:
        raise
    except Exception:
        raise RagError("parse_error", "PDF could not be read.") from None


def parse_document(extension: str, data: bytes) -> list[SourceLine]:
    if extension == ".pdf":
        return _parse_pdf(data)
    text = _decode(data)
    if extension == ".json":
        text = _pretty_json(text)
    return _number_lines(text, _section_matcher(extension))
