"""Chapter 5: the human comment and the AI comment for the same story, side by side."""

import json
import sys
import textwrap
from pathlib import Path
from typing import Any


def by_story(path: str) -> dict[str, dict[str, Any]]:
    return {str(r["story_id"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


human, ai = by_story("data/raw/hn_human.jsonl"), by_story("data/raw/hn_ai.jsonl")
story_id = sys.argv[1]
print(f"story: {human[story_id]['title']} ({human[story_id]['year']})\n")
for label, row in (("HUMAN", human[story_id]), ("AI", ai[story_id])):
    print(f"--- {label}, {len(row['text'].split())} words ---")
    print(textwrap.fill(row["text"], 100), "\n")
