"""Tests for src.llm.providers.deepseek and the provider selection around it.

The `openai` client is a fake throughout: the suite must never make a real API
call. Use `scripts/test_api.py` for that.

The point of most of these tests is the *error contract*, not the happy path.
`src/translator.py` retries `RuntimeError` five times with exponential backoff
and treats everything else as fatal, so which exception type this provider
raises for a given HTTP status decides whether a dead key costs one second or
several minutes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src import config
from src.llm.factory import get_llm
from src.llm.providers.deepseek import DeepSeekProvider
from src.llm.providers.gemini import GeminiProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Fakes ───────────────────────────────────────────────────────────────────


class FakeCompletions:
    """Records `create` calls; returns scripted content or raises."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome: object, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.chat = type("_Chat", (), {"completions": FakeCompletions(outcome)})()


def _response(content: str | None) -> Any:
    """Minimal stand-in for an openai ChatCompletion object."""
    message = type("_Message", (), {"content": content})()
    choice = type("_Choice", (), {"message": message})()
    return type("_Completion", (), {"choices": [choice]})()


def _make_provider(monkeypatch: pytest.MonkeyPatch, outcome: object) -> DeepSeekProvider:
    """A provider wired to a fake client that yields ``outcome``."""
    clients: list[FakeClient] = []

    def _factory(**kwargs: Any) -> FakeClient:
        client = FakeClient(outcome, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("src.llm.providers.deepseek.OpenAI", _factory)
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "fake-key")
    monkeypatch.setattr(config, "DEEPSEEK_MODEL", "deepseek-v4-pro")

    provider = DeepSeekProvider()
    provider._clients = clients  # type: ignore[attr-defined]  # for assertions
    return provider


def _calls(provider: DeepSeekProvider) -> list[dict[str, Any]]:
    return provider._client.chat.completions.calls  # type: ignore[attr-defined]


def _status_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return APIStatusError(
        f"HTTP {status} from the API",
        response=httpx.Response(status, request=request),
        body=None,
    )


# ── Request shape ───────────────────────────────────────────────────────────


def test_generate_sends_system_and_user_messages_with_the_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider(monkeypatch, _response("Hello, world."))

    assert provider.generate("You translate.", "你好，世界。") == "Hello, world."

    (call,) = _calls(provider)
    assert call["model"] == "deepseek-v4-pro"
    assert call["messages"] == [
        {"role": "system", "content": "You translate."},
        {"role": "user", "content": "你好，世界。"},
    ]


def test_thinking_is_disabled_and_reasoning_effort_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking mode silently ignores temperature and bills its own tokens."""
    provider = _make_provider(monkeypatch, _response("ok"))

    provider.generate("system", "user")

    (call,) = _calls(provider)
    assert call["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in call
    assert "reasoning_effort" not in call["extra_body"]


@pytest.mark.parametrize("temperature", [0.0, 0.2, 0.3])
def test_temperature_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, temperature: float
) -> None:
    provider = _make_provider(monkeypatch, _response("ok"))

    provider.generate("system", "user", temperature)

    assert _calls(provider)[0]["temperature"] == temperature


def test_client_is_constructed_once_not_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(monkeypatch, _response("ok"))

    provider.generate("system", "user")
    provider.generate("system", "user")

    assert len(provider._clients) == 1  # type: ignore[attr-defined]
    assert len(_calls(provider)) == 2
    assert provider._clients[0].init_kwargs["api_key"] == "fake-key"  # type: ignore[attr-defined]
    assert provider._clients[0].init_kwargs["base_url"] == config.DEEPSEEK_BASE_URL  # type: ignore[attr-defined]


# ── Error contract: retryable ───────────────────────────────────────────────


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_statuses_raise_runtime_error(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """RuntimeError is what tenacity in src/translator.py retries."""
    provider = _make_provider(monkeypatch, _status_error(status))

    with pytest.raises(RuntimeError, match=str(status)):
        provider.generate("system", "user")


def test_connection_and_timeout_errors_raise_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    for error in (
        APIConnectionError(request=request),
        APITimeoutError(request=request),
    ):
        provider = _make_provider(monkeypatch, error)
        with pytest.raises(RuntimeError):
            provider.generate("system", "user")


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_content_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    """A blank completion is a server-side hiccup, so it is worth retrying."""
    provider = _make_provider(monkeypatch, _response(content))

    with pytest.raises(RuntimeError, match="empty response"):
        provider.generate("system", "user")


# ── Error contract: fail fast ───────────────────────────────────────────────


@pytest.mark.parametrize("status", [400, 401, 402, 422])
def test_permanent_statuses_do_not_raise_runtime_error(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Retrying a bad key or an empty balance wastes minutes and tells you nothing."""
    provider = _make_provider(monkeypatch, _status_error(status))

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the type is the assertion
        provider.generate("system", "user")

    assert not isinstance(excinfo.value, RuntimeError)
    assert isinstance(excinfo.value, ValueError)
    assert str(status) in str(excinfo.value)


def test_402_message_names_the_empty_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """402 is the one people hit mid-run; the message has to say why."""
    provider = _make_provider(monkeypatch, _status_error(402))

    with pytest.raises(ValueError) as excinfo:
        provider.generate("system", "user")

    message = str(excinfo.value).lower()
    assert "balance" in message or "credit" in message


# ── Provider selection ──────────────────────────────────────────────────────


def test_factory_returns_deepseek_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "fake-key")

    assert isinstance(get_llm(), DeepSeekProvider)


def test_factory_still_returns_gemini_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")

    assert isinstance(get_llm(), GeminiProvider)


def test_factory_rejects_an_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "llamafile")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm()


# ── Config credential validation ────────────────────────────────────────────


def _import_config(**env_overrides: str | None) -> subprocess.CompletedProcess[str]:
    """Import src.config in a subprocess under a controlled environment.

    A subprocess (rather than importlib.reload) keeps the already-imported
    config module in this process untouched, and `load_dotenv` is stubbed out
    so the developer's own `.env` cannot supply a key the test is trying to
    withhold.
    """
    env = {key: value for key, value in os.environ.items()}
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    code = (
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: False;"
        "import src.config as c;"
        "print(c.LLM_PROVIDER, c.DEEPSEEK_MODEL)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_config_imports_with_only_a_deepseek_key() -> None:
    """A DeepSeek user with no Gemini key must not be blocked at import time."""
    result = _import_config(
        LLM_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="fake-key",
        DEEPSEEK_MODEL=None,
        GEMINI_API_KEY=None,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["deepseek", "deepseek-v4-pro"]


def test_config_still_requires_the_active_providers_key() -> None:
    result = _import_config(
        LLM_PROVIDER="gemini",
        DEEPSEEK_API_KEY="fake-key",
        GEMINI_API_KEY=None,
    )

    assert result.returncode != 0
    assert "GEMINI_API_KEY is missing" in result.stderr


def test_deepseek_is_the_default_provider() -> None:
    result = _import_config(
        LLM_PROVIDER=None,
        DEEPSEEK_API_KEY="fake-key",
        GEMINI_API_KEY=None,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == "deepseek"
