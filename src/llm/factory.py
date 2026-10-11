"""Factory: the right backend for one role.

Every LLM call in the pipeline names a role -- ``profile``, ``structure``,
``classify`` or ``translate`` (see `src.config.role_config`) -- and
``LLM_BACKEND`` decides who answers it:

``anthropic``
    `src.llm.providers.anthropic.AnthropicProvider` calls the Messages API.
    API mode, and the `src.main` automated path.
``exchange``
    `src.llm.exchange.ExchangeLLM` writes a request file and returns the
    answer an agent has written, or records the request as pending. No-API
    mode; the default when ``ANTHROPIC_API_KEY`` is unset.

Translation is the exception: in no-API mode it never goes through the
exchange, because the kit already is one (``kit/chunks/NNN.md`` in,
``kit/translated/NNN.tex`` out). Asking for a ``translate`` LLM on the
exchange backend is therefore an error with directions.
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.llm.base import BaseLLM, VisionLLM


def get_llm(role: str, *, work_dir: Path | None = None) -> BaseLLM:
    """The text model for ``role`` on the configured backend.

    ``work_dir`` is the book's ``data/work/<book>/``; the exchange keeps its
    request and response files there, so it is required on that backend.
    """
    config.role_config(role)  # validates the role and its settings up front
    backend = config.llm_backend()
    if backend == "exchange":
        if role == config.ROLE_TRANSLATE:
            raise ValueError(
                "LLM_BACKEND=exchange does not translate through Python: the kit is the "
                "exchange. Translate kit/chunks/NNN.md into kit/translated/NNN.tex with the "
                "chunk-translator agent (see .claude/skills/translate-book/SKILL.md), or "
                "set ANTHROPIC_API_KEY and LLM_BACKEND=anthropic for the automated path."
            )
        if work_dir is None:
            raise ValueError(f"the exchange backend needs the book's work dir (role {role})")
        from src.llm.exchange import ExchangeLLM

        return ExchangeLLM(role, work_dir)

    from src.llm.providers.anthropic import AnthropicProvider

    return AnthropicProvider(role)


def get_vision_llm(*, work_dir: Path | None = None) -> VisionLLM:
    """The figure classifier (role ``classify``) on the configured backend."""
    llm = get_llm(config.ROLE_CLASSIFY, work_dir=work_dir)
    assert isinstance(llm, VisionLLM)
    return llm
