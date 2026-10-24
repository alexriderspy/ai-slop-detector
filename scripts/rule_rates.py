"""Chapter 7: how often each slop pattern shows up, per 1,000 words, in each kind of text."""

import json
from collections import Counter
from pathlib import Path

from slop.rules import RULES, find_hits

FILES = {
    "HN human": "data/raw/hn_human.jsonl",
    "HN AI": "data/raw/hn_ai.jsonl",
    "Wiki human": "data/raw/wiki_human.jsonl",
}

rates: dict[str, dict[str, float]] = {}
for name, path in FILES.items():
    texts = [json.loads(line)["text"] for line in Path(path).read_text().splitlines()]
    words = sum(len(t.split()) for t in texts)
    counts = Counter(hit.rule for t in texts for hit in find_hits(t))
    rates[name] = {rule: 1000 * counts[rule] / words for rule in RULES}

print(f"{'per 1,000 words':<16}" + "".join(f"{name:>12}" for name in FILES))
for rule in RULES:
    print(f"{rule:<16}" + "".join(f"{rates[name][rule]:>12.2f}" for name in FILES))
