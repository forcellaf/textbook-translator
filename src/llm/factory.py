"""Factory for selecting the configured LLM provider.

Reads `LLM_PROVIDER` from config (sourced from `.env`) and returns the
matching adapter. Add new providers here as they are implemented under
`src/llm/providers/`.
"""

from src import config
from src.llm.base import BaseLLM, VisionLLM
from src.llm.providers.deepseek import DeepSeekProvider
from src.llm.providers.gemini import GeminiProvider


def get_llm() -> BaseLLM:
    """Return an instance of the configured LLM provider."""
    if config.LLM_PROVIDER == "deepseek":
        return DeepSeekProvider()
    if config.LLM_PROVIDER == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")


def get_vision_llm() -> VisionLLM:
    """Return the image-capable model, independent of `LLM_PROVIDER`.

    Vision is DeepSeek's `DEEPSEEK_VISION_MODEL`, on the existing
    `DEEPSEEK_API_KEY`. It is checked here rather than in `src.config`
    because only figure classification needs it.
    """
    if not config.DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is missing; figure classification uses DeepSeek's "
            f"vision model ({config.DEEPSEEK_VISION_MODEL}). Set it in .env, or "
            "pass --no-classify."
        )
    return DeepSeekProvider()
