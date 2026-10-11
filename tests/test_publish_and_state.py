"""Tests for publishing to data/output/<book>/, run state, the API batch
translator and the --json outputs. Nothing here calls an API or pdflatex."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import anthropic
import httpx2
import pytest

from src.batch_translate import run as run_batch
from src.lint import SENTINEL
from src.publish import publish
from src.run_state import load

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(monkeypatch: pytest.MonkeyPatch, name: str):  # noqa: ANN202
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module(name)


# ── Publishing ──────────────────────────────────────────────────────────────


def _built_book(tmp_path: Path, *, succeeded: bool) -> Path:
    work = tmp_path / "work" / "physics"
    kit = work / "kit"
    kit.mkdir(parents=True)
    (kit / "translated_book.tex").write_text("\\documentclass{book}", encoding="utf-8")
    time.sleep(0.01)
    (kit / "translated_book.pdf").write_bytes(b"%PDF-1.5")
    (kit / "build_result.json").write_text(
        json.dumps({"succeeded": succeeded, "message": "x"}), encoding="utf-8"
    )
    (work / "run_log.md").write_text("# physics -- translation run log\n", encoding="utf-8")
    return work


def test_a_built_book_publishes_pdf_tex_and_log(tmp_path: Path) -> None:
    work = _built_book(tmp_path, succeeded=True)
    result = publish(work, output_root=tmp_path / "output")

    out = tmp_path / "output" / "physics"
    assert result.built
    assert sorted(p.name for p in out.iterdir()) == [
        "physics_translated.pdf", "physics_translated.tex", "run_log.md"
    ]
    assert (work / "kit" / "translated_book.pdf").exists()  # copied, not moved


def test_a_failed_build_publishes_the_log_only(tmp_path: Path) -> None:
    work = _built_book(tmp_path, succeeded=False)
    out = tmp_path / "output" / "physics"
    out.mkdir(parents=True)
    (out / "physics_translated.pdf").write_bytes(b"%PDF from an earlier run")

    result = publish(work, output_root=tmp_path / "output")

    assert not result.built
    names = sorted(p.name for p in out.iterdir())
    assert "physics_translated.pdf" not in names
    assert names == ["physics_translated.stale.pdf", "run_log.md"]
    assert any("not publishing the PDF" in p for p in result.problems)


def test_a_pdf_older_than_its_tex_is_not_published(tmp_path: Path) -> None:
    work = _built_book(tmp_path, succeeded=True)
    tex = work / "kit" / "translated_book.tex"
    later = time.time() + 60
    os.utime(tex, (later, later))
    assert not publish(work, output_root=tmp_path / "output").built


def test_the_publish_script_exits_non_zero_without_a_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish_script = _script(monkeypatch, "publish")
    monkeypatch.setattr("publish.publish", lambda w: publish(w, output_root=tmp_path / "out"))
    work = _built_book(tmp_path, succeeded=False)
    assert publish_script.main([str(work)]) == 1


# ── Run state ───────────────────────────────────────────────────────────────


def test_run_state_tracks_stage_chunks_builds_and_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_state = _script(monkeypatch, "run_state")
    work = tmp_path / "book"
    (work / "kit" / "chunks").mkdir(parents=True)
    (work / "kit" / "translated").mkdir()
    for n in ("001", "002", "003"):
        (work / "kit" / "chunks" / f"{n}.md").write_text("x", encoding="utf-8")
    (work / "kit" / "translated" / "001.tex").write_text("x", encoding="utf-8")

    assert run_state.main([str(work), "sync-chunks"]) == 0
    assert run_state.main([str(work), "set", "stage", "translate"]) == 0
    assert run_state.main([str(work), "set", "chunk", "2", "3", "failed", "--error", "lost IMG"]) == 0
    assert run_state.main([str(work), "set", "chunk", "002", "translated"]) == 0
    assert run_state.main([str(work), "build", "failed", "--message", "008.tex:39"]) == 0
    assert run_state.main([str(work), "decide", "start p15", "--evidence", "第12章 on p16"]) == 0

    state = load(work)
    assert state.stage == "translate"
    assert state.chunks["001"].status == "translated" and state.chunks["001"].attempts == 0
    assert (state.chunks["002"].status, state.chunks["002"].attempts) == ("translated", 2)
    assert (state.chunks["003"].status, state.chunks["003"].last_error) == ("failed", "lost IMG")
    assert state.build_attempts[0]["succeeded"] is False
    assert state.decisions[0]["evidence"] == "第12章 on p16"


def test_run_state_rejects_an_unknown_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_state = _script(monkeypatch, "run_state")
    (tmp_path / "book").mkdir()
    assert run_state.main([str(tmp_path / "book"), "set", "stage", "dancing"]) == 1
    assert not (tmp_path / "book" / "run_state.json").exists()


# ── The API batch translator ────────────────────────────────────────────────


def _kit(tmp_path: Path) -> Path:
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir()
    (kit / "system_prompt.txt").write_text("Translate.", encoding="utf-8")
    for n in ("001", "002", "003"):
        (kit / "chunks" / f"{n}.md").write_text(f"第{n}块", encoding="utf-8")
    return kit


def _message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class BatchServer:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.created: list[dict] = []
        self.polls = 0

    def _batch(self, status: str) -> dict:
        return {
            "id": "msgbatch_1", "type": "message_batch", "processing_status": status,
            "request_counts": {"processing": 0, "succeeded": 2, "errored": 0, "canceled": 0,
                               "expired": 0},
            "created_at": "2026-10-11T00:00:00Z", "expires_at": "2026-10-12T00:00:00Z",
            "ended_at": None, "cancel_initiated_at": None, "archived_at": None,
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results",
        }

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if request.method == "POST":
            self.created.append(json.loads(request.content))
            return httpx2.Response(200, json=self._batch("in_progress"))
        if path.endswith("/results"):
            lines = [
                {"custom_id": "001", "result": {"type": "succeeded",
                                                "message": _message(f"Chunk one.\n{SENTINEL}")}},
                {"custom_id": "002", "result": {"type": "succeeded",
                                                "message": _message("Chunk two, cut", "max_tokens")}},
                {"custom_id": "003", "result": {"type": "succeeded",
                                                "message": _message("No sentinel here.")}},
            ]
            return httpx2.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        return httpx2.Response(200, json=self._batch(status))


def _client(server: BatchServer) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="k", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(server))
    )


def test_the_batch_sends_every_untranslated_chunk_with_a_cached_system_prompt(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    (kit / "translated" / "003.tex").write_text("done", encoding="utf-8")
    server = BatchServer(["ended"])
    run_batch(kit, client=_client(server), sleep=lambda _s: None)

    (body,) = server.created
    assert [r["custom_id"] for r in body["requests"]] == ["001", "002"]
    params = body["requests"][0]["params"]
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"] == {"effort": "medium"}
    assert "temperature" not in params


def test_truncated_or_unsentinelled_replies_fail_and_are_kept_aside(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    outcome = run_batch(kit, client=_client(BatchServer(["ended"])), sleep=lambda _s: None)

    assert outcome.translated == ["001"]
    assert "max_tokens" in outcome.failed["002"]
    assert SENTINEL in outcome.failed["003"]
    assert (kit / "translated" / "001.tex").read_text(encoding="utf-8").endswith(SENTINEL)
    assert not (kit / "translated" / "002.tex").exists()
    assert (kit / "translated" / "002.failed.tex").exists()


def test_a_running_batch_is_resumed_not_resubmitted(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    server = BatchServer(["in_progress", "ended"])
    first = run_batch(kit, client=_client(server), wait_minutes=0, sleep=lambda _s: None)
    assert first.state == "running"

    second = run_batch(kit, client=_client(server), sleep=lambda _s: None)
    assert second.state == "collected"
    assert len(server.created) == 1


# ── --json outputs ──────────────────────────────────────────────────────────


def test_lint_json_prints_a_summary_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lint_translation = _script(monkeypatch, "lint_translation")
    kit = _kit(tmp_path)
    (kit / "translated" / "001.tex").write_text("Chunk one.", encoding="utf-8")

    code = lint_translation.main([str(kit), "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert code == 1 and summary["needs_review"]
    assert any(f["check"] == "missing-translation" and f["file"] == "translated/002.tex"
               for f in summary["review"])
    assert Path(summary["log"]).exists()


def test_build_json_reports_the_lint_gate_and_records_the_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_pdf = _script(monkeypatch, "build_pdf")
    kit = _kit(tmp_path)

    code = build_pdf.main([str(kit), "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert code == 1
    assert summary["stage"] == "lint" and summary["succeeded"] is False
    assert summary["lint"]["counts"]["review"] >= 3
    record = json.loads((kit / "build_result.json").read_text(encoding="utf-8"))
    assert record["succeeded"] is False and record["stage"] == "lint"
