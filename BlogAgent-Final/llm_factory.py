"""
llm_factory.py  –  unified LLM provider for BWA 4.0
Supports:  Groq (cloud)  |  Ollama (local)
"""

from __future__ import annotations
import os
from typing import Literal

ProviderType = Literal["groq", "ollama"]

# Groq model catalogue
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (recommended)",
    "llama-3.1-8b-instant":    "Llama 3.1 8B Instant (fast)",
    "mixtral-8x7b-32768":      "Mixtral 8×7B 32K (long context)",
    "gemma2-9b-it":            "Gemma 2 9B IT",
}

# Ollama model catalogue
OLLAMA_MODELS = {
    "llama3.1":       "Llama 3.1 8B (recommended)",
    "llama3.2":       "Llama 3.2 3B (lightweight)",
    "mistral":        "Mistral 7B",
    "phi3":           "Phi-3 Mini",
    "gemma2":         "Gemma 2 9B",
    "deepseek-r1:7b": "DeepSeek R1 7B",
}


def get_llm(provider: ProviderType, model_name: str, temperature: float = 0.3):
    """
    Return a LangChain-compatible chat LLM for the given provider/model.
    Raises ValueError if the provider is unknown or env vars are missing.
    """
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature)

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'groq' or 'ollama'.")


def get_structured_llm(provider: ProviderType, model_name: str, schema, temperature: float = 0.1):
    """Return an LLM with structured output bound to the given Pydantic schema."""
    llm = get_llm(provider, model_name, temperature)
    return llm.with_structured_output(schema)