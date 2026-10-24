"""Pair every human comment with the AI comment on the same story, then split by story."""

import json
import random
from pathlib import Path
from typing import Any

RAW = Path("data/raw")


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["story_id"]): r for r in map(json.loads, path.read_text().splitlines())}


human, ai = load(RAW / "hn_human.jsonl"), load(RAW / "hn_ai.jsonl")
stories = sorted(human.keys() & ai.keys())  # only stories where we have both
random.Random(42).shuffle(stories)
cut = int(len(stories) * 0.8)
splits = {"train": stories[:cut], "test": stories[cut:]}  # a story is in train OR test, never both

for name, ids in splits.items():
    rows = [r for sid in ids for r in (human[sid], ai[sid])]
    Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{name}: {len(ids)} stories -> {len(rows)} texts (half human, half AI)")
test_ai = [ai[sid] for sid in splits["test"]]
(RAW / "hn_ai_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_ai))
