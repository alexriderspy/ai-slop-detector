# ai-slop-detector

Is this text written by a person, or churned out by an LLM? A detector built three ways (slop-pattern
rules, perplexity from a small language model, and both combined), trained on Hacker News comments
written before ChatGPT existed versus LLM comments on the same stories, then tested honestly: false
positives, a domain it never saw, and AI text rewritten to sound human.

Built in my YouTube video "Build an AI slop detector" (@sreemantidey).

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -e .
cp .env.example .env && set -a && source .env && set +a   # local Ollama by default

python data/fetch_hn.py              # human comments from before 30 Nov 2022
python data/generate_ai.py           # the LLM writes a comment on every same story
python data/build_dataset.py         # pair them and split by story
python train.py                      # four detectors compared, the best saved to models/
python data/fetch_wiki.py && python data/generate_ai.py wiki && python data/generate_ai.py rewrite
python evaluate.py                   # false positives, another domain, the rewrite attack
uvicorn slop.api:app --port 8000     # open http://localhost:8000
```

Generating the AI text needs an LLM. By default it's `qwen2.5:7b` through Ollama's OpenAI-compatible
API (`ollama pull qwen2.5:7b`). To use a cloud model instead, change `LLM_BASE_URL`, `LLM_MODEL` and
`LLM_API_KEY` in `.env`. Perplexity uses `unsloth/Llama-3.2-1B-Instruct` (about 2.5 GB, downloaded on
first use; set `SCORER_MODEL` to change it).

## What's inside

```
src/slop/
  config.py       settings from environment variables
  llm.py          one OpenAI-compatible call (Ollama or any cloud provider)
  text.py         cleaning, sentences, word counts
  rules.py        the slop patterns, with positions for highlighting
  perplexity.py   token-by-token surprise from a small language model
  features.py     rules + rhythm + perplexity as numbers
  api.py          FastAPI: POST /score, and a page at /
data/             fetch_hn.py, generate_ai.py, build_dataset.py, fetch_wiki.py
train.py          four detectors compared on held-out stories
evaluate.py       false positives, cross-domain, the rewrite attack
scripts/          small demos used in the video
tests/            pytest, with a fake detector so the API tests need no model
```

## A warning

No AI-text detector is reliable enough to accuse a person on its own. This one flags some human text
as AI and can be fooled by a simple rewrite; `evaluate.py` shows by how much. Treat the score as a
signal to look closer, never as proof.
