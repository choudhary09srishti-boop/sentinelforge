"""Small text helpers used in examples."""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def slugify(title: str) -> str:
    """Turn a title into a lowercase, dash separated URL slug."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


class WordCounter:
    """Counts how often each word appears in a text."""

    def count(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for word in re.findall(r"\w+", text.lower()):
            counts[word] = counts.get(word, 0) + 1
        return counts
