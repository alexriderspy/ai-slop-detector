"""How surprised is a small language model by each word? LLM text tends to be unsurprising."""

import math
from dataclasses import dataclass
from functools import cache, lru_cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from slop.config import load_settings

MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Surprise:
    log_ppl: float  # average surprise per token: low means the text is very predictable
    spread: float  # how much the surprise jumps around from token to token


@cache
def load_scorer() -> tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    name = load_settings().scorer_model
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)  # already in eval mode
    model.to(device)  # type: ignore[arg-type]  # transformers' type hints for .to() confuse mypy
    return tokenizer, model, device


@lru_cache(maxsize=4096)  # the same text is often scored by several detectors
@torch.no_grad()
def surprise(text: str) -> Surprise:
    tokenizer, model, device = load_scorer()
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids
    ids = ids.to(device)
    logits = model(ids).logits[0, :-1].float()  # the prediction at position i is for token i + 1
    log_probs = torch.log_softmax(logits, dim=-1)  # scores -> log probabilities over the vocabulary
    next_tokens = ids[0, 1:].unsqueeze(1)  # the tokens that really came next
    surprises = -log_probs.gather(1, next_tokens).squeeze(1)  # minus log probability of each one
    return Surprise(log_ppl=surprises.mean().item(), spread=surprises.std().item())


def perplexity(text: str) -> float:
    return math.exp(surprise(text).log_ppl)
