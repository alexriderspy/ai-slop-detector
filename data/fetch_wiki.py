"""Cross-domain human text: the opening of Wikipedia articles as they were on 1 January 2022."""

import json
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from slop.text import word_count

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "sreemanti-bot/0.1 (+https://github.com/alexriderspy/ai-slop-detector)"}
OUT = Path("data/raw/wiki_human.jsonl")


def lead_before_2022(client: httpx.Client, title: str) -> str:
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvstart": "2022-01-01T00:00:00Z",
        "rvdir": "older",
        "rvprop": "ids",
        "format": "json",
        "redirects": 1,
    }  # follow redirects, e.g. "Electric car" may point elsewhere
    page = next(iter(client.get(API, params=params).json()["query"]["pages"].values()))
    revid = page["revisions"][0]["revid"]  # the last edit before 2022
    params = {"action": "parse", "oldid": revid, "prop": "text", "section": 0, "format": "json"}
    soup = BeautifulSoup(client.get(API, params=params).json()["parse"]["text"]["*"], "html.parser")
    for tag in soup.select("sup, .mw-empty-elt, style"):  # [1] citation markers, empty paragraphs
        tag.decompose()
    words: list[str] = []
    for p in soup.find_all("p"):
        if len(words) >= 60:
            break
        words += p.get_text().split()  # no separator, so "word ," never appears
    return " ".join(words[:200])


titles = [t.strip() for t in Path("data/wiki_titles.txt").read_text().splitlines() if t.strip()]
with httpx.Client(headers=HEADERS, timeout=30) as client:
    rows = [
        {
            "story_id": f"wiki:{t}",
            "title": t,
            "text": lead_before_2022(client, t),
            "source": "human",
        }
        for t in titles
    ]
rows = [r for r in rows if word_count(r["text"]) >= 40]
OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(titles)} articles -> {len(rows)} openings from before 2022 -> {OUT}")
