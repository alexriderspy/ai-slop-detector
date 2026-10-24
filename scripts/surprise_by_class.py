"""Chapter 8: one sentence proves nothing, so compare every test comment: human vs AI."""

import math
import statistics

from slop.dataset import load_split
from slop.perplexity import surprise

texts, labels = load_split("data/test.jsonl")
scores = [surprise(t) for t in texts]
for label, name in ((0, "human"), (1, "AI")):
    group = [s for s, y in zip(scores, labels, strict=True) if y == label]
    mean = statistics.mean(s.log_ppl for s in group)
    spread = statistics.mean(s.spread for s in group)
    print(f"{name:>5}: {len(group)} comments, average surprise {mean:.2f} ", end="")
    print(f"(perplexity {math.exp(mean):.1f}), spread {spread:.2f}")
human_median = statistics.median(s.log_ppl for s, y in zip(scores, labels, strict=True) if y == 0)
below = [s.log_ppl < human_median for s, y in zip(scores, labels, strict=True) if y == 1]
print(f"AI comments less surprising than the median human comment: {sum(below)}/{len(below)}")
