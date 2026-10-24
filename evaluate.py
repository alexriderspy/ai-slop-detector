"""How good are the detectors, really? The home test, a domain they never saw, and an attack."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from slop.dataset import load_split

THRESHOLDS = (0.5, 0.8)  # 0.5: the default cut-off; 0.8: what the API needs to say "likely AI"
NAMES = ["slop_rules", "tfidf_words", "perplexity", "rules_perplexity"]


def rows(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_sets() -> dict[str, tuple[list[str], list[int]]]:
    wiki = rows("data/raw/wiki_human.jsonl") + rows("data/raw/wiki_ai.jsonl")
    rewritten = [r["text"] for r in rows("data/raw/hn_ai_rewritten.jsonl")]
    return {
        "HN test (home)": load_split("data/test.jsonl"),
        "Wikipedia (never seen)": (
            [r["text"] for r in wiki],
            [int(r["source"] == "ai") for r in wiki],
        ),
        "AI rewritten (attack)": (rewritten, [1] * len(rewritten)),
    }


def share(flags: NDArray[np.bool_]) -> str:
    return f"{flags.mean():.1%}" if len(flags) else "-"


def main() -> None:
    detectors = {name: joblib.load(f"models/{name}.joblib") for name in NAMES}
    for set_name, (texts, labels) in test_sets().items():
        y = np.array(labels)
        print(f"\n== {set_name}: {int((y == 0).sum())} human, {int((y == 1).sum())} AI ==")
        print(f"{'detector':<18} {'humans flagged':>16} {'AI caught':>16} {'ROC AUC':>8}")
        print(f"{'':<18} {'at 0.5 / at 0.8':>16} {'at 0.5 / at 0.8':>16}")
        for name, detector in detectors.items():
            p_ai = detector.predict_proba(texts)[:, 1]
            false_alarms = " / ".join(share(p_ai[y == 0] >= t) for t in THRESHOLDS)
            caught = " / ".join(share(p_ai[y == 1] >= t) for t in THRESHOLDS)
            auc = f"{roc_auc_score(y, p_ai):.3f}" if len(set(labels)) == 2 else "-"
            print(f"{name:<18} {false_alarms:>16} {caught:>16} {auc:>8}")


if __name__ == "__main__":
    main()
