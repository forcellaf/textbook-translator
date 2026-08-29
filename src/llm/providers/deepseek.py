"""DeepSeek provider implementation of BaseLLM using the OpenAI-compatible API.

DeepSeek exposes an OpenAI-compatible `/chat/completions` endpoint, so the
`openai` SDK is used as the transport with `base_url` pointed at DeepSeek.
"""

# Import config first: this sets HTTP_PROXY/HTTPS_PROXY/HF_ENDPOINT in
# os.environ (if configured) before the openai SDK is imported, so any network
# calls it makes respect the proxy settings.
from src import config
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    OpenAIError,
)

from src.llm.base import BaseLLM

# DeepSeek turns chain-of-thought on by default at "high" effort. It is
# disabled here for two reasons:
#   1. Reasoning tokens bill as output tokens, and this task (transcribe and
#      translate) needs no reasoning -- it would be the most expensive line
#      item in a book-sized run for no gain.
#   2. Thinking mode silently ignores `temperature` (and top_p / the penalty
#      parameters): per the API docs they "will not trigger an error but will
#      also have no effect". The pipeline passes temperature=0.3 for
#      translation and 0.2 for profiling and depends on that determinism, so
#      leaving thinking on would break it with no error to notice.
# `reasoning_effort` must not be sent alongside a disabled `thinking` block.
_THINKING_DISABLED = {"thinking": {"type": "disabled"}}

# Failures worth another attempt: rate limiting and anything server-side.
# Everything else in the 4xx range is a configuration or request problem that
# will fail identically on every retry.
_RETRYABLE_STATUS = 429

# Non-retryable statuses, and what to tell the user about each. 402 is the one
# people hit mid-run, so it names the fix.
_FATAL_HINTS = {
    400: "the request format was rejected",
    401: "the API key was rejected -- check DEEPSEEK_API_KEY",
    402: "the account is out of credit (insufficient balance) -- top up at "
    "https://platform.deepseek.com/top_up",
    422: "the request parameters were invalid",
}


class DeepSeekProvider(BaseLLM):
    """LLM adapter for DeepSeek's chat models via the `openai` SDK.

    Error contract, which `src/translator.py` depends on: transient failures
    (429, 5xx, connection/timeout, empty response) raise `RuntimeError` so
    tenacity retries them; permanent ones (400, 401, 402, 422) raise
    `ValueError` so the run fails fast instead of burning a five-attempt
    backoff schedule on a dead key or an empty balance.
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._model = config.DEEPSEEK_MODEL

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                extra_body=_THINKING_DISABLED,
            )
        except APIStatusError as exc:
            raise self._from_status(exc) from exc
        except APIConnectionError as exc:  # includes APITimeoutError
            raise RuntimeError(
                f"DeepSeek API call failed ({self._model}): could not reach the API: {exc}"
            ) from exc
        except OpenAIError as exc:  # any other SDK-level failure
            raise RuntimeError(f"DeepSeek API call failed ({self._model}): {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            # Transient server-side oddity rather than a config problem, so
            # this is retryable.
            raise RuntimeError(f"DeepSeek API returned an empty response ({self._model}).")

        return content

    def _from_status(self, exc: APIStatusError) -> Exception:
        """Map an HTTP status onto this provider's retryable/fatal contract.

        The status code is read off the exception rather than matched in its
        message, which varies with the error and the deployment.
        """
        status = exc.status_code
        hint = _FATAL_HINTS.get(status)
        if hint is not None:
            return ValueError(f"DeepSeek API call failed ({self._model}, HTTP {status}): {hint}.")
        if status == _RETRYABLE_STATUS or status >= 500:
            return RuntimeError(
                f"DeepSeek API call failed ({self._model}, HTTP {status}): {exc}"
            )
        # An unlisted 4xx (403, 404, ...) is still a request the server will
        # reject every time; retrying it only delays the report.
        return ValueError(f"DeepSeek API call failed ({self._model}, HTTP {status}): {exc}")
