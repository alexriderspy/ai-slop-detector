"""Chapter 10: what did the 'sound human' rewrite change, and what did it miss?"""

import json
from pathlib import Path

from slop.features import STYLE_NAMES, style_matrix

SHOW = [
    "stock_word_per_100_words",
    "em_dash_per_100_words",
    "sentence_length",
    "sentence_burstiness",
]


def texts(path: str, source: str | None = None) -> list[str]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [r["text"] for r in rows if source is None or r["source"] == source]


groups = {
    "human (test)": texts("data/test.jsonl", "human"),
    "AI (test)": texts("data/raw/hn_ai_test.jsonl"),
    "AI rewritten": texts("data/raw/hn_ai_rewritten.jsonl"),
}
print(f"{'average':<26}" + "".join(f"{name:>14}" for name in groups))
means = {name: style_matrix(group).mean(axis=0) for name, group in groups.items()}
for feature in SHOW:
    i = STYLE_NAMES.index(feature)
    print(f"{feature:<26}" + "".join(f"{float(means[name][i]):>14.2f}" for name in groups))
