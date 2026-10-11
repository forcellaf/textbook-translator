"""Tests for src.llm.exchange: no-API mode's request/response files.

Nothing here calls a model. "The agent" is a function in this file that
reads request files and writes response files, which is exactly the contract
the real subagents follow.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.figures import FLAG_MALFORMED, FLAG_PENDING, build_figures
from src.llm import exchange
from src.llm.exchange import (
    EXIT_PENDING,
    ExchangeLLM,
    ExchangePending,
    pending_requests,
    write_pending,
)
from src.profiler import _sample_text, profile_book
from src.structure import _row, find_divisions

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

BOOK = (
    "# 第篇 电磁学\n\n本篇讲述宏观电磁现象的规律。\n\n"
    "# 静电场\n\n"
    "## 12.1 电荷\n\n正文。\n\n"
    "![](images/f1.png)\n图12.1 质子和中子的电荷分布\n\n"
    "![](images/glyph.png)\n\n"
    "$$\nF = k q_1 q_2 / r^2 \\tag{12.1}\n$$\n\n"
    "## 12.2 电场\n\n正文。\n\n"
    "![](images/loose.png)\n\n"
    "# 第13章 电势\n\n"
    "## 13.1 静电场的保守性\n\n正文。\n"
)


@pytest.fixture(autouse=True)
def _fresh_round(monkeypatch: pytest.MonkeyPatch) -> None:
    exchange.reset()
    monkeypatch.setenv("LLM_BACKEND", "exchange")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield
    exchange.reset()


def _work(tmp_path: Path, text: str = BOOK) -> Path:
    work = tmp_path / "book"
    images = work / "images"
    images.mkdir(parents=True)
    for name in ("f1.png", "glyph.png", "loose.png"):
        (images / name).write_bytes(PNG_HEADER + name.encode())
    (work / "merged.md").write_text(text, encoding="utf-8")
    return work


def _requests(work: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((work / "llm" / "requests").glob("*.json"))
    }


def _answer(work: Path, request_id: str, reply: str) -> None:
    path = work / "llm" / "responses" / f"{request_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reply, encoding="utf-8")


# ── The protocol ────────────────────────────────────────────────────────────


def test_a_missing_answer_writes_a_request_and_is_recorded_as_pending(tmp_path: Path) -> None:
    work = _work(tmp_path)
    llm = ExchangeLLM("structure", work)

    with pytest.raises(ExchangePending) as caught:
        llm.generate("the system prompt", "the user text", temperature=0.0)

    rid = caught.value.request_id
    request = json.loads((work / "llm" / "requests" / f"{rid}.json").read_text(encoding="utf-8"))
    assert request["id"] == rid and len(rid) == 16
    assert request["role"] == "structure"
    assert request["model"] == "claude-sonnet-5-5"
    assert request["effort"] == "medium"
    assert request["system"] == "the system prompt"
    assert request["user"] == "the user text"
    assert request["images"] == []
    assert request["reply_format"] == "json"
    assert request["created"].endswith("Z")

    assert write_pending(work) == 1
    manifest = json.loads((work / "llm" / "pending.json").read_text(encoding="utf-8"))
    assert manifest["by_role"] == {"structure": [rid]}


def test_pending_is_not_an_api_error_so_retry_loops_leave_it_alone() -> None:
    assert not issubclass(ExchangePending, (RuntimeError, ValueError))


def test_a_written_answer_is_returned_verbatim(tmp_path: Path) -> None:
    work = _work(tmp_path)
    llm = ExchangeLLM("profile", work)
    with pytest.raises(ExchangePending) as caught:
        llm.generate("s", "u")
    _answer(work, caught.value.request_id, '{"subject": "physics"}')

    exchange.reset()
    assert llm.generate("s", "u") == '{"subject": "physics"}'
    assert write_pending(work) == 0
    assert not (work / "llm" / "pending.json").exists()  # a stale manifest is cleared


def test_classify_requests_carry_their_image_as_a_readable_file(tmp_path: Path) -> None:
    work = _work(tmp_path)
    data = PNG_HEADER + b"pixels"
    with pytest.raises(ExchangePending) as caught:
        ExchangeLLM("classify", work).generate_with_images("s", "u", [(data, "image/jpeg")])

    request = _requests(work)[caught.value.request_id]
    assert request["model"] == "claude-haiku-5-5" and request["effort"] == "low"
    (relative,) = request["images"]
    assert relative.startswith("llm/images/") and relative.endswith(".png")
    assert (work / relative).read_bytes() == data


def test_ids_are_stable_across_runs_and_change_with_the_prompt(tmp_path: Path) -> None:
    work = _work(tmp_path)
    ids = []
    for _ in range(2):
        exchange.reset()
        with pytest.raises(ExchangePending) as caught:
            ExchangeLLM("profile", work).generate("s", "u")
        ids.append(caught.value.request_id)
    assert ids[0] == ids[1]

    with pytest.raises(ExchangePending) as caught:
        ExchangeLLM("profile", work).generate("s", "u, edited")
    assert caught.value.request_id != ids[0]


def test_the_prompts_behind_the_ids_are_deterministic() -> None:
    big = "# 标题\n\n" + "\n\n".join(f"## {i}.1 节\n\n" + "正文。" * 400 for i in range(40))
    assert _sample_text(big) == _sample_text(big)
    tables = [
        "\n".join(_row(i, d) for i, d in enumerate(find_divisions(BOOK), start=1))
        for _ in range(2)
    ]
    assert tables[0] == tables[1]


def test_profile_requests_are_collected_together_and_nothing_is_cached(tmp_path: Path) -> None:
    work = _work(tmp_path)
    path = work / "merged.md"
    first = profile_book(path, work)
    assert first.glossary == ()
    assert not (work / "book_profile.json").exists()  # a provisional profile is never cached
    # The profile call and the heading call, in one round.
    assert len(pending_requests(work)["profile"]) == 2

    exchange.reset()
    profile_book(path, work)
    assert len(pending_requests(work)["profile"]) == 2  # same ids again: nothing new


# ── Figures ─────────────────────────────────────────────────────────────────


def test_every_figure_request_is_emitted_in_one_pass(tmp_path: Path) -> None:
    work = _work(tmp_path)
    report = build_figures(
        BOOK, images_dir=work / "images", vision=ExchangeLLM("classify", work)
    )

    candidates = [name for name in ("glyph.png", "loose.png")]
    assert len(pending_requests(work)["classify"]) == len(candidates)
    assert report.pending == len(candidates)
    for name in candidates:
        record = report.records[name]
        assert record.keep and FLAG_PENDING in record.flags


def test_a_malformed_answer_keeps_the_figure_and_is_never_re_requested(tmp_path: Path) -> None:
    work = _work(tmp_path)
    vision = ExchangeLLM("classify", work)
    build_figures(BOOK, images_dir=work / "images", vision=vision)
    for rid in pending_requests(work)["classify"]:
        _answer(work, rid, "I think it is probably a figure?")

    for _ in range(2):
        exchange.reset()
        report = build_figures(
            BOOK, images_dir=work / "images", vision=vision, existing=None
        )
        assert pending_requests(work) == {}
        for name in ("glyph.png", "loose.png"):
            assert report.records[name].keep
            assert FLAG_MALFORMED in report.records[name].flags


# ── prepare_kit.py, round by round ──────────────────────────────────────────


def _agent(work: Path) -> int:
    """Answer every pending request the way the subagents would."""
    manifest = json.loads((work / "llm" / "pending.json").read_text(encoding="utf-8"))
    answered = 0
    for role, ids in manifest["by_role"].items():
        for rid in ids:
            request = _requests(work)[rid]
            if role == "classify":
                reply = {"label": "figure", "confidence": 0.95, "reason": "a diagram"}
            elif role == "profile" and "heading_levels" in request["system"]:
                reply = {"heading_levels": {}, "non_headings": [], "fragments": []}
            elif role == "profile":
                reply = {"subject": "physics", "glossary": [{"source": "电荷", "target": "charge"}]}
            elif "The row to re-check" in request["user"]:
                reply = {"kind": "chapter", "number": None, "reason": "kept"}
            else:
                reply = {
                    "rows": [
                        {"row": 1, "kind": "part", "number": None, "title": "电磁学", "reason": "r"},
                        {"row": 2, "kind": "chapter", "number": 12, "title": "静电场", "reason": "r"},
                        {"row": 3, "kind": "chapter", "number": 13, "title": "电势", "reason": "r"},
                    ]
                }
            _answer(work, rid, json.dumps(reply, ensure_ascii=False))
            answered += 1
    return answered


def test_prepare_kit_exits_with_the_pending_code_until_every_answer_is_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    prepare_kit = importlib.import_module("prepare_kit")
    work = _work(tmp_path)

    codes: list[int] = []
    for _round in range(8):
        exchange.reset()
        code = prepare_kit.main([str(work)])
        codes.append(code)
        if code != EXIT_PENDING:
            break
        assert _agent(work) > 0

    assert codes[0] == EXIT_PENDING
    assert codes[-1] == 0, capsys.readouterr().out
    assert len(codes) <= 6  # the stage2-preparer's round limit
    assert not (work / "llm" / "pending.json").exists()
    assert (work / "kit" / "chunks" / "001.md").exists()
    profile = json.loads((work / "book_profile.json").read_text(encoding="utf-8"))
    assert profile["subject"] == "physics"
    structure = json.loads((work / "structure.json").read_text(encoding="utf-8"))
    assert structure["headings"]["静电场"]["by"] in ("llm", "follow-up")

    # A further run is free: every answer is cached.
    exchange.reset()
    assert prepare_kit.main([str(work)]) == 0
