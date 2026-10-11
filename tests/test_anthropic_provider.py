"""Tests for src.llm.providers.anthropic.

Every request is served by an ``httpx2.MockTransport`` (the SDK runs on
``httpx2``, the maintained fork of ``httpx``), the same pattern
``test_mineru_api.py`` uses for MinerU. Nothing here reaches the network.

The fake speaks the Messages API's server-sent events, because the provider
always streams: thinking blocks arrive in the stream exactly as they would
from the API, so "thinking is stripped" is tested against the real wire
shape rather than a hand-built object.
"""

from __future__ import annotations

import base64
import json

import anthropic
import httpx2
import pytest

from src.llm.providers.anthropic import (
    FALLBACK_BETA,
    AnthropicProvider,
    RefusalError,
    TruncatedReplyError,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for role in ("PROFILE", "STRUCTURE", "CLASSIFY", "TRANSLATE"):
        for kind in ("MODEL", "EFFORT", "MAX_TOKENS"):
            monkeypatch.delenv(f"CLAUDE_{kind}_{role}", raising=False)
    monkeypatch.setattr("src.config.CLAUDE_REFUSAL_FALLBACK", "default")


def _sse(blocks: list[dict], stop_reason: str = "end_turn", stop_details: dict | None = None) -> str:
    """A complete Messages API event stream for ``blocks``."""
    events: list[tuple[str, dict]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        )
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": "sig"},
            ]
        else:
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        events.append(
            (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
        )
        for delta in deltas:
            events.append(
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    delta: dict = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events.append(
        ("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 5}})
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class Server:
    """Records each request and answers with a scripted response."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def _stream_reply(blocks: list[dict], **kwargs) -> Server:
    text = _sse(blocks, **kwargs)
    return Server(
        lambda _request: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=text.encode()
        )
    )


def _error_reply(status: int, error_type: str, message: str) -> Server:
    payload = {"type": "error", "error": {"type": error_type, "message": message}}
    return Server(lambda _request: httpx2.Response(status, json=payload))


def _provider(role: str, server: Server) -> AnthropicProvider:
    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )
    return AnthropicProvider(role, client=client)


# ── Replies ─────────────────────────────────────────────────────────────────


def test_only_text_comes_back_and_thinking_is_stripped() -> None:
    server = _stream_reply(
        [
            {"type": "thinking", "thinking": "private chain of thought"},
            {"type": "text", "text": "Hello, "},
            {"type": "text", "text": "world."},
        ]
    )
    reply = _provider("translate", server).generate("system", "你好，世界。", 0.3)

    assert reply == "Hello, world."
    assert "private" not in reply


def test_request_shape_effort_adaptive_thinking_cached_system_no_temperature() -> None:
    server = _stream_reply([{"type": "text", "text": "ok"}])
    _provider("translate", server).generate("the system prompt", "the chunk", 0.3)

    body = server.body
    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"] == {"effort": "medium"}
    assert body["thinking"] == {"type": "adaptive"}
    assert body["stream"] is True
    assert "temperature" not in body
    assert body["system"] == [
        {"type": "text", "text": "the system prompt", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "the chunk"}]
    assert body["max_tokens"] >= 64_000  # thinking plus a ~36k-token translation


@pytest.mark.parametrize(
    ("role", "model", "effort"),
    [
        ("profile", "claude-sonnet-5-5", "medium"),
        ("structure", "claude-sonnet-5-5", "medium"),
        ("classify", "claude-haiku-5-5", "low"),
        ("translate", "claude-haiku-5-5", "medium"),
    ],
)
def test_model_and_effort_are_chosen_per_role(role: str, model: str, effort: str) -> None:
    server = _stream_reply([{"type": "text", "text": "ok"}])
    _provider(role, server).generate("s", "u")

    assert server.body["model"] == model
    assert server.body["output_config"]["effort"] == effort


def test_role_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_MODEL_STRUCTURE", "claude-opus-5-5")
    monkeypatch.setenv("CLAUDE_EFFORT_STRUCTURE", "high")
    monkeypatch.setenv("CLAUDE_MAX_TOKENS_STRUCTURE", "12345")
    server = _stream_reply([{"type": "text", "text": "ok"}])
    _provider("structure", server).generate("s", "u")

    assert server.body["model"] == "claude-opus-5-5"
    assert server.body["output_config"]["effort"] == "high"
    assert server.body["max_tokens"] == 12345


def test_an_invalid_effort_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_EFFORT_PROFILE", "extreme")
    with pytest.raises(ValueError, match="CLAUDE_EFFORT_PROFILE"):
        _provider("profile", _stream_reply([{"type": "text", "text": "ok"}]))


def test_sonnet_roles_carry_the_refusal_fallback_and_haiku_roles_do_not() -> None:
    sonnet = _stream_reply([{"type": "text", "text": "ok"}])
    _provider("profile", sonnet).generate("s", "u")
    assert sonnet.body["fallbacks"] == "default"
    assert FALLBACK_BETA in sonnet.requests[-1].headers["anthropic-beta"]

    haiku = _stream_reply([{"type": "text", "text": "ok"}])
    _provider("classify", haiku).generate("s", "u")
    assert "fallbacks" not in haiku.body
    assert "anthropic-beta" not in haiku.requests[-1].headers


def test_images_travel_in_the_user_message_before_the_text() -> None:
    server = _stream_reply([{"type": "text", "text": '{"label": "figure"}'}])
    _provider("classify", server).generate_with_images(
        "classify it", "context", [(PNG, "image/jpeg")]
    )

    content = server.body["messages"][0]["content"]
    assert content[0]["type"] == "image"
    # The type is read from the bytes, not the caller's (MinerU's .jpg) guess.
    assert content[0]["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(content[0]["source"]["data"]) == PNG
    assert content[1] == {"type": "text", "text": "context"}
    assert isinstance(server.body["system"], list)  # instructions stay out of the user turn


# ── Failures ────────────────────────────────────────────────────────────────


def test_a_reply_cut_off_at_max_tokens_is_a_failure_not_an_answer() -> None:
    server = _stream_reply([{"type": "text", "text": "\\chapter{Half a ch"}], stop_reason="max_tokens")
    with pytest.raises(TruncatedReplyError, match="max_tokens"):
        _provider("translate", server).generate("s", "u")
    assert issubclass(TruncatedReplyError, RuntimeError)  # retryable


def test_an_empty_reply_is_transient() -> None:
    server = _stream_reply([{"type": "thinking", "thinking": "hmm"}])
    with pytest.raises(RuntimeError, match="empty"):
        _provider("translate", server).generate("s", "u")


def test_a_refusal_raises() -> None:
    server = _stream_reply(
        [], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}
    )
    with pytest.raises(RefusalError):
        _provider("classify", server).generate("s", "u")


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (429, "rate_limit_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_transient_statuses_raise_runtime_error(status: int, error_type: str) -> None:
    with pytest.raises(RuntimeError) as caught:
        _provider("translate", _error_reply(status, error_type, "try later")).generate("s", "u")
    assert not isinstance(caught.value, ValueError)
    assert f"HTTP {status}" in str(caught.value)


@pytest.mark.parametrize(
    ("status", "error_type", "message", "hint"),
    [
        (400, "invalid_request_error", "messages: bad", "rejected"),
        (
            400,
            "invalid_request_error",
            "Your credit balance is too low to access the Anthropic API.",
            "out of credit",
        ),
        (401, "authentication_error", "invalid x-api-key", "ANTHROPIC_API_KEY"),
        (403, "permission_error", "nope", "not allowed"),
        (404, "not_found_error", "model: claude-nope", "CLAUDE_MODEL"),
    ],
)
def test_permanent_statuses_raise_value_error_with_a_hint(
    status: int, error_type: str, message: str, hint: str
) -> None:
    with pytest.raises(ValueError, match=hint):
        _provider("translate", _error_reply(status, error_type, message)).generate("s", "u")


def test_a_connection_failure_is_transient() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(RuntimeError, match="could not reach"):
        _provider("translate", Server(refuse)).generate("s", "u")
