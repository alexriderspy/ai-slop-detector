"""Human text: one top-level Hacker News comment per story, all written before ChatGPT existed."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from slop.text import clean, word_count

API = "https://hn.algolia.com/api/v1"
# ChatGPT launched on 30 November 2022: nothing written before that was written by an LLM
CHATGPT_LAUNCH = int(datetime(2022, 11, 30, tzinfo=UTC).timestamp())
YEARS = range(2013, 2022)
STORIES_PER_YEAR = 45
OUT = Path("data/raw/hn_human.jsonl")


async def top_stories(client: httpx.AsyncClient, year: int) -> list[dict]:
    start, end = (int(datetime(y, 1, 1, tzinfo=UTC).timestamp()) for y in (year, year + 1))
    filters = f"created_at_i>={start},created_at_i<{end},points>250"
    params = {"tags": "story", "numericFilters": filters, "hitsPerPage": STORIES_PER_YEAR}
    response = await client.get(f"{API}/search", params=params)
    response.raise_for_status()
    return response.json()["hits"]


def pick_comment(story: dict) -> str | None:
    """The first top-level comment that is plain prose: 60 to 200 words, no links, no quotes."""
    for child in story.get("children", []):
        raw = child.get("text") or ""
        if child.get("created_at_i", CHATGPT_LAUNCH) >= CHATGPT_LAUNCH:
            continue
        text = clean(raw)
        if 60 <= word_count(text) <= 200 and "http" not in text and not text.startswith(">"):
            return text
    return None


async def main() -> None:
    semaphore = asyncio.Semaphore(5)  # be polite: 5 requests at a time, like in my fetcher video

    async def story_with_comments(client: httpx.AsyncClient, story_id: str) -> dict:
        async with semaphore:
            response = await client.get(f"{API}/items/{story_id}")
            response.raise_for_status()
            return response.json()

    async with httpx.AsyncClient(timeout=30) as client:
        hits = [h for year in YEARS for h in await top_stories(client, year)]
        stories = await asyncio.gather(*(story_with_comments(client, h["objectID"]) for h in hits))
    rows = []
    for story in stories:
        comment = pick_comment(story)
        if comment and story.get("title"):
            year = datetime.fromtimestamp(story["created_at_i"], UTC).year
            row = {"story_id": story["id"], "title": story["title"], "year": year, "text": comment}
            rows.append({**row, "source": "human"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(hits)} stories from {YEARS.start} to {YEARS.stop - 1}", end=" ")
    print(f"-> {len(rows)} usable comments -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
