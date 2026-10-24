"""Chapter 5: before training anything, check the AI text for give-aways that aren't about style."""

import json
import re
import statistics
import sys
from pathlib import Path

EMOJI = re.compile("[\U0001f300-\U0001faff]")
HASHTAG = re.compile(r"(?:^|\s)#\w")


def share(flags: list[bool]) -> str:
    return f"{sum(flags) / len(flags):6.1%}"


def describe(name: str, texts: list[str]) -> None:
    words = [len(t.split()) for t in texts]
    short = share([w < 40 for w in words])
    tags = share([bool(HASHTAG.search(t)) for t in texts])
    emoji = share([bool(EMOJI.search(t)) for t in texts])
    print(f"{name:<10} {len(texts):>5} {statistics.median(words):>13.0f} {short} {tags} {emoji}")


def texts(path: str) -> list[str]:
    return [json.loads(line)["text"] for line in Path(path).read_text().splitlines()]


print(f"{'':<10} {'texts':>5} {'median words':>13} {'<40 w':>6} {'#tags':>6} {'emoji':>6}")
describe("human", texts("data/raw/hn_human.jsonl"))
describe("AI", texts(sys.argv[1]))
