"""Claude provider: `BaseLLM` and `VisionLLM` over the Anthropic Messages API.

Used in API mode (``LLM_BACKEND=anthropic``). No-API mode answers the same
calls through `src.llm.exchange` instead, and both build the *same* request:
`build_request` is the one place a role's model, effort, system prompt and
content become API parameters, so the exchange's request files and the batch
translator (`scripts/translate_kit.py`) describe exactly what this provider
would send.

Decisions worth knowing before editing
--------------------------------------
* **Roles, not one model.** Every call site names a role (profile,
  structure, classify, translate) and `src.config.role_config` maps it to a
  model, an effort and a max_tokens. The provider is constructed per role.
* **Effort, adaptive thinking.** Effort goes in ``output_config.effort``;
  thinking is adaptive on every model this project uses. Only the text
  blocks come back -- thinking blocks are dropped, never returned as part of
  an answer.
* **No temperature.** The pipeline used to pass 0.0-0.3 and relied on it.
  Claude Opus 5.5 rejects ``temperature`` outright and Claude Sonnet 5.5 /
  Haiku 5.5 reject any non-default value (HTTP 400), with thinking always
  on. The argument is still accepted, so callers need not change, but it is
  never sent. Determinism now rests on the on-disk caches
  (``book_profile.json``, ``structure.json``, ``figures.json``, the exchange's
  responses): an answer is asked for once and then reused.
* **Truncation is a failure.** Thinking counts toward ``max_tokens``, so a
  reply cut off at the cap may be missing its whole tail. ``stop_reason ==
  "max_tokens"`` raises `TruncatedReplyError`; a partial answer is never
  returned as success.
* **Streaming.** Every call streams and reads the final message. A
  translation can run past the SDK's ten-minute non-streaming guard, and one
  code path is easier to reason about than two.
* **Prompt caching.** ``cache_control`` sits on the system prompt. For
  translation that is the long shared ``system_prompt.txt``; the short
  profile and structure prompts fall under the minimum cacheable size, which
  is harmless.
* **Refusal fallback.** On the Sonnet/Opus roles the server-side fallback is
  on by default (``CLAUDE_REFUSAL_FALLBACK``): a request a safety classifier
  declines is re-run on a fallback model inside the same call. Haiku has no
  server-side fallback. A refusal that survives raises `RefusalError`.

Error contract (``src/translator.py`` and ``src/figures.py`` depend on it):
transient failures -- 408, 409, 429, 5xx, connection/timeout, an empty
reply, a truncated reply, a refusal -- raise ``RuntimeError`` (worth
retrying); permanent ones -- 400, 401, 403, 404, 413, no credit, no
credentials -- raise ``ValueError`` with an actionable hint, so a dead key
fails fast instead of burning a backoff schedule.
"""

from __future__ import annotations

import base64

# Import config first: it sets HTTP_PROXY/HTTPS_PROXY in os.environ (if
# configured) before the SDK is imported.
from src import config

import anthropic

from src.config import RoleConfig, role_config
from src.llm.base import BaseLLM, VisionLLM, image_mime_type

# The `fallbacks: "default"` form routes by refusal category, so there is no
# fallback model list to maintain here.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Statuses a later attempt can plausibly get past.
_TRANSIENT_STATUSES = frozenset({408, 409, 429})

_FATAL_HINTS = {
    401: "the API key was rejected -- check ANTHROPIC_API_KEY (or run without one: "
    "LLM_BACKEND=exchange)",
    403: "the key is not allowed to use this model or feature",
    404: "the model was not found -- check CLAUDE_MODEL_<ROLE>",
    413: "the request is too large -- lower KIT_CHUNK_CHARS or the sample size",
}


class TruncatedReplyError(RuntimeError):
    """The reply hit ``max_tokens``. Retryable: thinking length varies, so
    the same request can fit on a later attempt; raise
    ``CLAUDE_MAX_TOKENS_<ROLE>`` if it keeps happening."""


class RefusalError(RuntimeError):
    """A safety classifier declined the request (after any fallback)."""


def uses_fallback(cfg: RoleConfig) -> bool:
    """Whether this role's calls carry the server-side refusal fallback."""
    return config.CLAUDE_REFUSAL_FALLBACK != "off" and "haiku" not in cfg.model


def build_request(cfg: RoleConfig, system_prompt: str, content: str | list[dict]) -> dict:
    """The Messages API parameters for one call in role ``cfg``.

    Shared by the live provider, the batch translator and the exchange, so
    the three can never drift apart. No temperature: see the module
    docstring.
    """
    return {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": cfg.effort},
    }


def image_blocks(images: list[tuple[bytes, str]]) -> list[dict]:
    """Image content blocks; the MIME type is read from the bytes, because
    MinerU names every image ``.jpg``."""
    blocks: list[dict] = []
    for data, mime in images:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime_type(data) or mime,
                    "data": base64.standard_b64encode(data).decode("ascii"),
                },
            }
        )
    return blocks


def reply_text(message: object, model: str) -> str:
    """The text of a finished message, or the exception its stop reason
    calls for. Thinking blocks (and fallback markers) are dropped."""
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise TruncatedReplyError(
            f"Claude reply hit max_tokens ({model}); the answer is incomplete. Raise "
            "CLAUDE_MAX_TOKENS_<ROLE> if this keeps happening."
        )
    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise RefusalError(f"Claude declined the request ({model}, category {category}).")

    text = "".join(
        block.text for block in getattr(message, "content", []) or [] if block.type == "text"
    )
    if not text.strip():
        raise RuntimeError(f"Claude returned an empty reply ({model}).")
    return text


class AnthropicProvider(BaseLLM, VisionLLM):
    """One role's Claude model. See the module docstring for the contract."""

    def __init__(
        self,
        role: str,
        *,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.config = role_config(role)
        self._client = client or anthropic.Anthropic(max_retries=config.ANTHROPIC_MAX_RETRIES)

    @property
    def role(self) -> str:
        return self.config.role

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.0) -> str:
        return self._complete(system_prompt, user_text)

    def generate_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[tuple[bytes, str]],
        temperature: float = 0.0,
    ) -> str:
        # Images first, then the text that refers to them.
        content = image_blocks(images) + [{"type": "text", "text": user_text}]
        return self._complete(system_prompt, content)

    def _complete(self, system_prompt: str, content: str | list[dict]) -> str:
        cfg = self.config
        params = build_request(cfg, system_prompt, content)
        try:
            if uses_fallback(cfg):
                with self._client.beta.messages.stream(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                ) as stream:
                    message = stream.get_final_message()
            else:
                with self._client.messages.stream(**params) as stream:
                    message = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise _from_status(exc, cfg.model) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise RuntimeError(
                f"Claude API call failed ({cfg.model}): could not reach the API: {exc}"
            ) from exc
        except anthropic.AnthropicError as exc:
            # No credentials at all is the common one here.
            raise ValueError(
                f"Claude API call failed ({cfg.model}): {exc}. Set ANTHROPIC_API_KEY, "
                "or run in no-API mode with LLM_BACKEND=exchange."
            ) from exc
        return reply_text(message, cfg.model)


def _from_status(exc: anthropic.APIStatusError, model: str) -> Exception:
    """Map an HTTP status onto the retryable/permanent contract.

    Read off the status code, not the message -- except for "out of credit",
    which the API reports as a 400 whose message is the only distinguishing
    feature.
    """
    status = exc.status_code
    if status in _TRANSIENT_STATUSES or status >= 500:
        return RuntimeError(f"Claude API call failed ({model}, HTTP {status}): {exc}")
    if status == 400 and "credit" in str(exc).lower():
        return ValueError(
            f"Claude API call failed ({model}, HTTP 400): the account is out of credit "
            "-- add credit in the Console, or run in no-API mode (LLM_BACKEND=exchange)."
        )
    hint = _FATAL_HINTS.get(status, "the request was rejected")
    return ValueError(f"Claude API call failed ({model}, HTTP {status}): {hint}. ({exc})")


__all__ = [
    "FALLBACK_BETA",
    "AnthropicProvider",
    "RefusalError",
    "TruncatedReplyError",
    "build_request",
    "image_blocks",
    "reply_text",
    "uses_fallback",
]
