"""Turn a text into numbers a classifier can use: slop-pattern rates, rhythm and perplexity."""

import statistics

import numpy as np
from numpy.typing import NDArray

from slop.perplexity import surprise
from slop.rules import RULES, find_hits
from slop.text import sentences, word_count

RATE_NAMES = [f"{rule}_per_100_words" for rule in RULES]
STYLE_NAMES = [*RATE_NAMES, "sentence_length", "sentence_burstiness"]
SURPRISE_NAMES = ["log_perplexity", "surprise_spread"]


def style_features(text: str) -> list[float]:
    words = max(word_count(text), 1)
    counts = dict.fromkeys(RULES, 0)
    for hit in find_hits(text):
        counts[hit.rule] += 1
    rates = [100 * counts[rule] / words for rule in RULES]
    lengths = [word_count(s) for s in sentences(text)] or [words]
    mean = statistics.mean(lengths)
    burstiness = statistics.pstdev(lengths) / mean  # people mix short and long sentences
    return [*rates, mean, burstiness]


def surprise_features(text: str) -> list[float]:
    s = surprise(text)
    return [s.log_ppl, s.spread]


def style_matrix(texts: list[str]) -> NDArray[np.float64]:
    return np.array([style_features(t) for t in texts])


def surprise_matrix(texts: list[str]) -> NDArray[np.float64]:
    return np.array([surprise_features(t) for t in texts])


def all_matrix(texts: list[str]) -> NDArray[np.float64]:
    return np.hstack([style_matrix(texts), surprise_matrix(texts)])
