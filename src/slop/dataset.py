"""Load a JSONL split as texts and labels: 1 means written by AI, 0 means written by a person."""

import json
from pathlib import Path


def load_split(path: Path | str) -> tuple[list[str], list[int]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [r["text"] for r in rows], [int(r["source"] == "ai") for r in rows]
