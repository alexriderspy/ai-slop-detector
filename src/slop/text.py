"""Text cleaning shared by every stage, so human and AI text are cleaned exactly the same way."""

import html
import re

TAG = re.compile(r"<[^>]+>")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def clean(raw: str) -> str:
    """HN comments are HTML: <p> starts a paragraph, &#x27; is an apostrophe."""
    text = raw.replace("<p>", "\n\n")
    text = html.unescape(TAG.sub("", text))
    paragraphs = (" ".join(p.split()) for p in text.split("\n\n"))
    return "\n\n".join(p for p in paragraphs if p)


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.replace("\n\n", " ")) if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())
