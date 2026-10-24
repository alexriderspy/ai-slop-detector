"""One function to call an LLM. Ollama speaks the OpenAI API, so a cloud model is a config swap."""

from openai import OpenAI

from slop.config import load_settings


def complete(prompt: str, temperature: float = 0.8, max_tokens: int = 400) -> str:
    settings = load_settings()
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
