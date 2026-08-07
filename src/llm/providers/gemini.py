"""Gemini provider implementation of BaseLLM using the `google-genai` SDK."""

# Import config first: this sets HTTP_PROXY/HTTPS_PROXY/HF_ENDPOINT in
# os.environ (if configured) before the google-genai SDK is imported, so any
# network calls it makes respect the proxy settings.
from src import config
from google import genai
from google.genai import types

from src.llm.base import BaseLLM


class GeminiProvider(BaseLLM):
    """LLM adapter for Google's Gemini models via the `google-genai` SDK."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._model = config.GEMINI_MODEL

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            raise RuntimeError(f"Gemini API call failed ({self._model}): {exc}") from exc

        if not response.text:
            raise RuntimeError(f"Gemini API returned an empty response ({self._model}).")

        return response.text
