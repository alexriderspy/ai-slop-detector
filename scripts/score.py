"""Chapter 11: send a text file to the running API and print what comes back."""

import sys
from pathlib import Path

import httpx

url, path = sys.argv[1], Path(sys.argv[2])
response = httpx.post(f"{url}/score", json={"text": path.read_text()}, timeout=60)
print(f"HTTP {response.status_code}")
body = response.json()
if response.is_success:
    print(f"verdict: {body['verdict']}   P(AI) = {body['ai_probability']}   words: {body['words']}")
    for hit in body["highlights"]:
        print(f"  {hit['rule']:<14} {hit['text']!r}")
else:
    print(body["detail"])
