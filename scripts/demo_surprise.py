"""Chapter 8: watch the scorer model's surprise, token by token, on a short sentence."""

import sys

import torch

from slop.perplexity import load_scorer, perplexity, surprise

text = sys.argv[1]
tokenizer, model, device = load_scorer()
ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    log_probs = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
for position, token_id in enumerate(ids[0, 1:].tolist()):
    token = tokenizer.decode([token_id])
    print(f"{token!r:>16}  surprise {-log_probs[position, token_id].item():5.2f}")
s = surprise(text)
print(f"\naverage surprise {s.log_ppl:.2f} -> perplexity {perplexity(text):.1f}")
print(f"spread of surprise {s.spread:.2f}")
