"""AI text from the LLM in .env: same topic and about the same length as each human text.

python data/generate_ai.py            v2: AI comments for every Hacker News story (the one we use)
python data/generate_ai.py hn_v1      v1: my first prompt, kept to show what went wrong
python data/generate_ai.py wiki       AI Wikipedia openings, for the cross-domain test
python data/generate_ai.py rewrite    test-set AI comments, rewritten to sound human: the attack
"""

import json
import sys
from pathlib import Path
from typing import Any

from slop.config import load_settings
from slop.llm import complete
from slop.text import clean, word_count

HN_V1 = (
    "You are reading Hacker News. Write a comment of about {words} words replying to the story "
    'titled "{title}". Reply with only the comment.'
)
HN = (  # v2: LLMs can't count words, but they're decent at counting sentences
    'You are reading Hacker News. Write a comment replying to the story titled "{title}". '
    "Write {sentences} sentences{paragraphs}. Write it the way people comment on Hacker News: "
    "plain text, no hashtags, no emojis. Reply with only the comment."
)
WIKI = (
    'Write the opening of a Wikipedia article about "{title}": {sentences} sentences{paragraphs}. '
    "Plain text, no headings. Reply with only the text."
)
REWRITE = (
    "Rewrite this Hacker News comment so it reads like a real person typed it: plain words, no "
    "buzzwords, no em dashes, no exclamation marks, uneven sentence lengths. Keep the meaning. "
    "Reply with only the comment.\n\n{text}"
)
# task: (source file, output file, prompt, retry if shorter than this share of the target length)
TASKS = {
    "hn": ("data/raw/hn_human.jsonl", "data/raw/hn_ai.jsonl", HN, 0.7),
    "hn_v1": ("data/raw/hn_human.jsonl", "data/raw/hn_ai_v1.jsonl", HN_V1, 0.0),
    "wiki": ("data/raw/wiki_human.jsonl", "data/raw/wiki_ai.jsonl", WIKI, 0.7),
    "rewrite": ("data/raw/hn_ai_test.jsonl", "data/raw/hn_ai_rewritten.jsonl", REWRITE, 0.0),
}


def write_one(prompt: str, row: dict[str, Any], min_share: float) -> str:
    """Ask for about as much text as the human wrote; retry up to 3 times if it comes back short."""
    words = word_count(row["text"])
    length = {
        "words": round(words, -1),  # used by the v1 prompt
        "sentences": max(3, round(words / 12)),  # about 12 words per sentence
        "paragraphs": " in two paragraphs" if words >= 100 else "",
    }
    best = ""
    for _ in range(3):
        reply = clean(complete(prompt.format(**length, **row)).strip('"'))
        best = max(best, reply, key=word_count)
        if word_count(best) >= min_share * words:
            break
    return best


def generate(task: str) -> None:
    src, out, prompt, min_share = TASKS[task]
    lines = Path(out).read_text().splitlines() if Path(out).exists() else []
    done = {json.loads(line)["story_id"] for line in lines}  # resume where a previous run stopped
    rows = [json.loads(line) for line in Path(src).read_text().splitlines()]
    model = load_settings().llm_model
    with Path(out).open("a") as f:  # append, so a crash halfway doesn't lose finished rows
        for i, row in enumerate(rows, 1):
            if row["story_id"] in done:
                continue
            text = write_one(prompt, row, min_share)
            f.write(json.dumps({**row, "text": text, "source": "ai", "model": model}) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(rows):
                print(f"{i}/{len(rows)}  {word_count(text)} words  {row['title'][:60]}")


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "hn")
