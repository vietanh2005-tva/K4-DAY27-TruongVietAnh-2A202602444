"""LLM factory supporting OpenAI-compatible gateways and OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    provider = os.environ.get("AI_PROVIDER", "openrouter").strip().lower()
    if provider not in {'openai', 'openrouter'}:
        raise ValueError('AI_PROVIDER must be openai or openrouter')
    key_name = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"{key_name} is not set — add it to .env")
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-6-luna") if provider == "openai" else os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") if provider == "openai" else os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
        timeout=float(os.environ.get('LLM_TIMEOUT_SECONDS', '120')),
        max_retries=1,
    )
