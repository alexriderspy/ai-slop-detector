"""The detector as a web service: POST /score with some text, or open / in a browser."""

from functools import cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from slop.rules import find_hits
from slop.text import word_count

MIN_WORDS = 40  # below this, any verdict is a guess, so we refuse to give one
MODEL_PATH = Path("models/rules_perplexity.joblib")  # chosen in chapter 10
app = FastAPI(title="AI slop detector")


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class Highlight(BaseModel):
    rule: str
    start: int
    end: int
    text: str


class ScoreResponse(BaseModel):
    ai_probability: float
    verdict: str
    words: int
    highlights: list[Highlight]


@cache
def detector() -> Pipeline:
    return joblib.load(MODEL_PATH)


def verdict(p_ai: float) -> str:
    if p_ai >= 0.8:
        return "likely AI"
    if p_ai <= 0.2:
        return "likely human"
    return "can't tell"  # the honest answer for the middle band


@app.post("/score")
def score(request: ScoreRequest) -> ScoreResponse:
    words = word_count(request.text)
    if words < MIN_WORDS:
        raise HTTPException(422, f"Need at least {MIN_WORDS} words to judge, got {words}.")
    p_ai = float(detector().predict_proba([request.text])[0, 1])
    hits = [
        Highlight(rule=h.rule, start=h.start, end=h.end, text=h.text)
        for h in find_hits(request.text)
    ]
    return ScoreResponse(
        ai_probability=round(p_ai, 3), verdict=verdict(p_ai), words=words, highlights=hits
    )


@app.get("/")
def page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
