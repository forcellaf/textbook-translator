"""Provider-agnostic LLM interface.

Every concrete provider (Gemini, OpenAI, etc.) must implement this interface.
Pipeline stages (meta_profiler.py, translator.py, renderer.py) depend only on
`BaseLLM`, never on a specific provider SDK.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM provider adapters."""

    @abstractmethod
    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        """Generate text from the LLM.

        Args:
            system_prompt: The system/instruction prompt.
            user_text: The user-provided content to process.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """
        raise NotImplementedError
