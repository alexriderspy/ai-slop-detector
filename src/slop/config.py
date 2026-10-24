"""Settings come from environment variables, so swapping in a cloud LLM is a one-line change."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    llm_base_url: str  # any OpenAI-compatible API: Ollama locally, or a cloud provider
    llm_model: str
    llm_api_key: str
    scorer_model: str  # the small language model we use to measure perplexity


def load_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        llm_api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
        scorer_model=os.getenv("SCORER_MODEL", "unsloth/Llama-3.2-1B-Instruct"),
    )
