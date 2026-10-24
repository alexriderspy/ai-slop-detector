from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from slop import api

SLOP = (
    "Great point! This is a crucial and pivotal moment for the tech landscape. "
    "It's not just a tool, but a testament to innovation, collaboration, and vision. "
) * 3


class FakeDetector:
    """Stands in for the real model, so the API tests don't need a 1B language model."""

    def __init__(self, p_ai: float) -> None:
        self.p_ai = p_ai

    def predict_proba(self, texts: list[str]) -> NDArray[np.float64]:
        return np.array([[1 - self.p_ai, self.p_ai] for _ in texts])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "detector", lambda: FakeDetector(0.93))
    yield TestClient(api.app)


def test_score_returns_verdict_and_highlights(client: TestClient) -> None:
    response = client.post("/score", json={"text": SLOP})
    assert response.status_code == 200
    body = response.json()
    assert (body["ai_probability"], body["verdict"]) == (0.93, "likely AI")
    assert {"stock_word", "not_x_but_y", "triad", "stock_opener"} <= {
        h["rule"] for h in body["highlights"]
    }


def test_short_text_is_refused(client: TestClient) -> None:
    response = client.post("/score", json={"text": "Too short to judge."})
    assert response.status_code == 422
    assert "at least 40 words" in response.json()["detail"]


@pytest.mark.parametrize(
    ("p_ai", "expected"), [(0.95, "likely AI"), (0.5, "can't tell"), (0.1, "likely human")]
)
def test_verdict_bands(p_ai: float, expected: str) -> None:
    assert api.verdict(p_ai) == expected
