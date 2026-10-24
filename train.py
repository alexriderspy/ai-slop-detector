"""Train four detectors on the same data and compare them on stories none of them have seen."""

from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from slop.dataset import load_split
from slop.features import STYLE_NAMES, SURPRISE_NAMES, all_matrix, style_matrix, surprise_matrix

Featurizer = Callable[[list[str]], NDArray[np.float64]]


def numeric(featurize: Featurizer) -> Pipeline:
    """Text -> features -> rescale each feature to mean 0, std 1 -> logistic regression."""
    return make_pipeline(
        FunctionTransformer(featurize), StandardScaler(), LogisticRegression(max_iter=1000)
    )


def detectors() -> dict[str, Pipeline]:
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "slop_rules": numeric(style_matrix),
        "tfidf_words": make_pipeline(words, LogisticRegression(max_iter=2000, C=5)),
        "perplexity": numeric(surprise_matrix),
        "rules_perplexity": numeric(all_matrix),
    }


def show_weights(names: list[str], weights: NDArray[np.float64], top: int) -> None:
    order = np.argsort(weights)  # most negative first
    human = ", ".join(f"{names[i]} {weights[i]:+.2f}" for i in order[:top])
    ai = ", ".join(f"{names[i]} {weights[i]:+.2f}" for i in order[::-1][:top])
    print(f"  pushes towards HUMAN: {human}\n  pushes towards AI:    {ai}")


def main() -> None:
    x_train, y_train = load_split("data/train.jsonl")
    x_test, y_test = load_split("data/test.jsonl")
    print(f"train {len(x_train)} texts, test {len(x_test)} texts\n")
    print(f"{'detector':<18} {'accuracy':>8} {'ROC AUC':>8}")
    Path("models").mkdir(exist_ok=True)
    trained = detectors()
    for name, detector in trained.items():
        detector.fit(x_train, y_train)
        p_ai = detector.predict_proba(x_test)[:, 1]
        accuracy = accuracy_score(y_test, p_ai >= 0.5)
        print(f"{name:<18} {accuracy:>8.3f} {roc_auc_score(y_test, p_ai):>8.3f}")
        joblib.dump(detector, f"models/{name}.joblib")

    print("\nwhat tf-idf learned (the words with the biggest weights):")
    vocabulary = list(trained["tfidf_words"][0].get_feature_names_out())
    show_weights(vocabulary, trained["tfidf_words"][-1].coef_[0], top=8)
    print("\nwhat rules + perplexity learned (weight per feature, after rescaling):")
    show_weights(STYLE_NAMES + SURPRISE_NAMES, trained["rules_perplexity"][-1].coef_[0], top=4)


if __name__ == "__main__":
    main()
