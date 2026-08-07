"""Factory for selecting the configured LLM provider.

Reads `LLM_PROVIDER` from config (sourced from `.env`) and returns the
matching adapter. Add new providers here as they are implemented under
`src/llm/providers/`.
"""

from src.config import LLM_PROVIDER
from src.llm.base import BaseLLM
from src.llm.providers.gemini import GeminiProvider


def get_llm() -> BaseLLM:
    """Return an instance of the configured LLM provider."""
    if LLM_PROVIDER == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
