"""Slop patterns: phrases LLMs overuse. Each hit keeps its position, so the app can highlight it."""

import re
from dataclasses import dataclass

STOCK_WORDS = (
    "delve|crucial|pivotal|tapestry|landscape|robust|seamless(?:ly)?|leverag(?:e|es|ing)|"
    "testament|showcas(?:e|es|ing)|underscor(?:e|es|ing)|foster(?:s|ing)?|vibrant|intricate|"
    "meticulous(?:ly)?|game[- ]changer|realm|elevate|holistic|endeavou?r|paramount|nuanced?|"
    "ever[- ]evolving|go-to|invaluable|commendable|insightful"
)

RULES: dict[str, re.Pattern[str]] = {
    "stock_word": re.compile(rf"\b(?:{STOCK_WORDS})\b", re.I),
    "not_x_but_y": re.compile(
        r"\bnot (?:just|only|merely)\b[^.!?]{1,80}?\bbut\b|\bit'?s not\b[^.!?]{1,60}?, it'?s\b",
        re.I,
    ),
    "em_dash": re.compile(r"\s?[—–]\s?|\s--\s"),
    "triad": re.compile(r"\b\w+(?: \w+)?, \w+(?: \w+)?,? and \w+(?: \w+)?\b"),
    "stock_opener": re.compile(
        r"^(?:great|interesting|fascinating|impressive|absolutely|indeed)\b[^.!?]{0,40}[.!?,]",
        re.I | re.M,
    ),
    "stock_closer": re.compile(
        r"\b(?:in conclusion|in summary|overall|ultimately|all in all)\b,", re.I
    ),
    "signposting": re.compile(
        r"\b(?:it'?s (?:important|worth) (?:to note|noting)|it'?s clear that"
        r"|let'?s dive|here'?s the thing)\b",
        re.I,
    ),
    "emoji": re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]"),
    "exclamation": re.compile(r"!"),
}


@dataclass(frozen=True, slots=True)
class Hit:
    rule: str
    start: int
    end: int
    text: str


def find_hits(text: str) -> list[Hit]:
    hits = [
        Hit(name, m.start(), m.end(), m.group())
        for name, rx in RULES.items()
        for m in rx.finditer(text)
    ]
    return sorted(hits, key=lambda h: h.start)
