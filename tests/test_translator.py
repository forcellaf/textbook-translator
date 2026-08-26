"""Tests for src.translator (chunking, healing, checkpointing).

Every LLM here is a fake: the suite must never make a real API call. Use
`scripts/test_api.py` for that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tenacity import wait_none

from src.config import API_MAX_RETRIES, MAX_HEAL_ATTEMPTS
from src.llm.base import BaseLLM
from src.profiler import BookProfile
from src.translator import (
    TranslationError,
    chunk_markdown,
    translate_book,
    translate_chunk,
    translate_markdown,
)


class FakeLLM(BaseLLM):
    """Scripted BaseLLM that records every call.

    ``responses`` are returned in order; once exhausted the last one repeats,
    so a test only has to script the responses it cares about. An ``Exception``
    instance in the list is raised instead of returned.

    With no scripted responses the fake echoes its input behind an ``[EN]``
    marker, which keeps every answer proportional to its source and so always
    clears the "implausibly short" heuristic.
    """

    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses: list[object] | None = list(responses) if responses is not None else None
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        if self.responses is None:
            return f"[EN] {user_text}"
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return str(response)

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture(autouse=True)
def _no_backoff_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tenacity's retry schedule but drop its sleeps (~15s otherwise)."""
    monkeypatch.setattr("src.translator._RETRY_WAIT", wait_none())


def _long_text(label: str, repeats: int = 30) -> str:
    return (f"{label} sentence about the subject matter. " * repeats).strip()


# ── chunk_markdown ──────────────────────────────────────────────────────────


def test_empty_and_whitespace_input_produce_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n \t\n") == []


def test_short_document_is_a_single_chunk() -> None:
    text = "# 第一章\n\n第一段。\n\n第二段。"
    chunks = chunk_markdown(text, max_tokens=1000)

    assert len(chunks) == 1
    assert "第一章" in chunks[0]
    assert "第二段。" in chunks[0]


def test_multiple_chunks_split_only_on_paragraph_boundaries() -> None:
    paragraphs = [f"Paragraph {i}: {_long_text('body', repeats=5)}" for i in range(8)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_markdown(text, max_tokens=200)

    assert len(chunks) > 1
    # Every paragraph survives whole, in order, in exactly one chunk.
    rejoined = [p for chunk in chunks for p in chunk.split("\n\n")]
    assert rejoined == paragraphs


def test_oversized_single_paragraph_is_kept_whole() -> None:
    table = "\n".join(f"| cell {i} | value {i} |" for i in range(200))  # one blank-line block
    chunks = chunk_markdown(table, max_tokens=50)

    assert len(chunks) == 1
    assert chunks[0] == table
    assert chunks[0].count("| cell") == 200


def test_table_rows_are_not_split_across_chunks() -> None:
    table = "| a | b |\n| --- | --- |\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(60))
    text = f"{_long_text('intro')}\n\n{table}\n\n{_long_text('outro')}"

    chunks = chunk_markdown(text, max_tokens=100)

    assert sum(1 for chunk in chunks if "| --- |" in chunk) == 1
    holder = next(chunk for chunk in chunks if "| --- |" in chunk)
    assert table in holder


# ── translate_chunk: healing ────────────────────────────────────────────────


def test_empty_response_triggers_heal_naming_the_problem() -> None:
    source = _long_text("source")
    good = _long_text("translated")
    llm = FakeLLM(["", good])

    result = translate_chunk(source, llm=llm)

    assert result == good
    assert llm.call_count == 2
    heal_prompt = llm.calls[1][0]
    assert "empty or implausibly short" in heal_prompt
    assert str(len(source)) in heal_prompt


def test_implausibly_short_response_triggers_heal() -> None:
    source = _long_text("source")  # comfortably over the 200-char floor
    good = _long_text("translated")
    llm = FakeLLM(["too short", good])

    result = translate_chunk(source, llm=llm)

    assert result == good
    assert "9 chars" in llm.calls[1][0]


def test_short_source_is_not_flagged_as_implausibly_short() -> None:
    llm = FakeLLM(["Hi."])

    assert translate_chunk("你好。", llm=llm) == "Hi."
    assert llm.call_count == 1


def test_invalid_latex_triggers_heal_quoting_the_defect() -> None:
    source = _long_text("公式", repeats=10)
    # Long enough to clear the length heuristic, so the LaTeX validation path
    # (not the length check) is what fires.
    broken = "\\begin{equation}\n" + _long_text("x = y +")
    fixed = _long_text("The equation states that") + "\n\\[ x = y \\]"
    llm = FakeLLM([broken, fixed])

    result = translate_chunk(source, llm=llm, output_format="latex")

    assert result == fixed
    assert llm.call_count == 2
    assert "invalid LaTeX" in llm.calls[1][0]
    assert "\\begin{equation} is never closed" in llm.calls[1][0]


def test_latex_mode_accepts_a_sound_fragment_without_healing() -> None:
    source = _long_text("公式", repeats=10)
    sound = _long_text("A well-formed") + "\n\\begin{itemize}\n\\item one\n\\end{itemize}"
    llm = FakeLLM([sound])

    assert translate_chunk(source, llm=llm, output_format="latex") == sound
    assert llm.call_count == 1


def test_last_response_returned_after_heals_are_exhausted(caplog) -> None:
    source = _long_text("source")
    llm = FakeLLM(["", "", "", "", ""])

    with caplog.at_level("ERROR"):
        result = translate_chunk(source, llm=llm)

    assert result == ""
    assert llm.call_count == MAX_HEAL_ATTEMPTS + 1
    assert any("Giving up healing" in record.message for record in caplog.records)


def test_wrapping_code_fence_is_stripped() -> None:
    body = _long_text("translated")
    llm = FakeLLM([f"```markdown\n{body}\n```"])

    assert translate_chunk(_long_text("source"), llm=llm) == body


# ── translate_chunk: transient failures ─────────────────────────────────────


def test_translation_error_after_retries_are_exhausted() -> None:
    llm = FakeLLM([RuntimeError("Gemini API call failed: 401 invalid key")])

    with pytest.raises(TranslationError, match="invalid key"):
        translate_chunk(_long_text("source"), llm=llm)


def test_hard_failure_does_not_re_run_backoff_once_per_heal_attempt() -> None:
    """A dead key must cost one backoff schedule, not one per heal attempt."""
    llm = FakeLLM([RuntimeError("401 invalid key")])

    with pytest.raises(TranslationError):
        translate_chunk(_long_text("source"), llm=llm)

    assert llm.call_count == API_MAX_RETRIES
    assert llm.call_count < API_MAX_RETRIES * (MAX_HEAL_ATTEMPTS + 1)


def test_transient_failure_is_retried_then_succeeds() -> None:
    good = _long_text("translated")
    llm = FakeLLM([RuntimeError("503 temporarily unavailable"), good])

    assert translate_chunk(_long_text("source"), llm=llm) == good
    assert llm.call_count == 2


# ── translate_markdown ──────────────────────────────────────────────────────


def test_translate_markdown_empty_input_makes_no_calls() -> None:
    llm = FakeLLM()

    assert translate_markdown("   ", "English", llm=llm) == ""
    assert llm.call_count == 0


def test_translate_markdown_reports_progress_per_chunk() -> None:
    # Long enough to exceed MAX_CHUNK_TOKENS and split into several chunks.
    text = "\n\n".join(_long_text(f"para{i}", repeats=60) for i in range(6))
    llm = FakeLLM()
    progress: list[tuple[int, int]] = []

    translate_markdown(
        text,
        "English",
        llm=llm,
        on_chunk_done=lambda index, total, _text: progress.append((index, total)),
    )

    assert progress
    assert [index for index, _ in progress] == list(range(1, len(progress) + 1))
    assert all(total == progress[-1][1] for _, total in progress)
    assert llm.call_count == progress[-1][1]


def test_translate_markdown_positional_target_lang_still_works() -> None:
    """src/main.py calls translate_markdown(md, lang) positionally."""
    llm = FakeLLM(["Translated."])

    translate_markdown("你好。", "French", llm=llm)

    assert "French" in llm.calls[0][0]


def test_profile_glossary_reaches_every_chunk_prompt() -> None:
    profile = BookProfile(subject="mathematics", glossary=(("导数", "derivative"),))
    text = "\n\n".join(_long_text(f"para{i}", repeats=60) for i in range(6))
    llm = FakeLLM()

    translate_markdown(text, "English", llm=llm, profile=profile)

    assert llm.call_count > 1
    assert all("导数 -> derivative" in system for system, _, _ in llm.calls)


def test_unknown_output_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="output_format"):
        translate_markdown("你好。", "English", llm=FakeLLM(), output_format="rtf")


# ── translate_book ──────────────────────────────────────────────────────────


def _write_book(tmp_path: Path) -> tuple[Path, Path]:
    work_dir = tmp_path / "book"
    work_dir.mkdir()
    merged = work_dir / "merged.md"
    merged.write_text(
        f"# 第一章 绪论\n\n{_long_text('第一章', repeats=3)}\n\n"
        f"# 第二章 方法\n\n{_long_text('第二章', repeats=3)}\n",
        encoding="utf-8",
    )
    return merged, work_dir


def test_translate_book_markdown_writes_checkpoints_and_merged_output(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    llm = FakeLLM([_long_text("translated", repeats=3)])

    output = translate_book(merged, work_dir, llm=llm, use_profile=False)

    assert output == work_dir / "translated_merged.md"
    assert "translated sentence" in output.read_text(encoding="utf-8")
    checkpoints = sorted((work_dir / "translated_chapters").glob("*"))
    assert len(checkpoints) == 2
    assert all(path.suffix == ".md" for path in checkpoints)


def test_translate_book_latex_mode_uses_tex_checkpoints(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    llm = FakeLLM(["\\chapter{Introduction}\n" + _long_text("Translated", repeats=3)])

    output = translate_book(
        merged, work_dir, llm=llm, use_profile=False, output_format="latex", title="A Textbook"
    )

    assert output == work_dir / "translated_book.tex"
    tex = output.read_text(encoding="utf-8")
    assert tex.count("\\begin{document}") == 1
    assert "\\chapter{Introduction}" in tex
    assert "A Textbook" in tex

    checkpoints = sorted((work_dir / "translated_chapters").glob("*"))
    assert len(checkpoints) == 2
    assert all(path.suffix == ".tex" for path in checkpoints)


def test_switching_output_format_re_translates_instead_of_mixing(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    body = _long_text("Translated", repeats=3)

    translate_book(merged, work_dir, llm=FakeLLM([body]), use_profile=False)
    latex_llm = FakeLLM(["\\chapter{One}\n" + body])
    translate_book(
        merged, work_dir, llm=latex_llm, use_profile=False, output_format="latex"
    )

    assert latex_llm.call_count == 2  # both chapters redone for the new format
    suffixes = {path.suffix for path in (work_dir / "translated_chapters").glob("*")}
    assert suffixes == {".md", ".tex"}


def test_resume_skips_chapters_that_already_have_checkpoints(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    body = _long_text("translated", repeats=3)
    translate_book(merged, work_dir, llm=FakeLLM([body]), use_profile=False)

    checkpoints = sorted((work_dir / "translated_chapters").glob("*.md"))
    checkpoints[0].unlink()  # simulate an interrupted run

    second_llm = FakeLLM([body])
    translate_book(merged, work_dir, llm=second_llm, use_profile=False)

    assert second_llm.call_count == 1
    assert checkpoints[0].exists()


def test_resume_re_translates_when_the_source_chapter_is_newer(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    body = _long_text("translated", repeats=3)
    translate_book(merged, work_dir, llm=FakeLLM([body]), use_profile=False)

    chapter = sorted((work_dir / "chapters").glob("*.md"))[0]
    chapter.write_text(chapter.read_text(encoding="utf-8") + "\n\n新增段落。\n", encoding="utf-8")

    second_llm = FakeLLM([body])
    translate_book(merged, work_dir, llm=second_llm, use_profile=False)

    assert second_llm.call_count == 1


def test_fully_translated_book_costs_zero_llm_calls(tmp_path: Path) -> None:
    """Locks in lazy profiling: nothing pending => no profiling call either."""
    merged, work_dir = _write_book(tmp_path)
    translate_book(
        merged, work_dir, llm=FakeLLM([_long_text("translated", repeats=3)]), use_profile=False
    )

    idle_llm = FakeLLM()
    output = translate_book(merged, work_dir, llm=idle_llm)  # profiling enabled

    assert idle_llm.call_count == 0
    assert not (work_dir / "book_profile.json").exists()
    assert output.exists()


def test_translate_book_rejects_unknown_output_format_before_any_work(tmp_path: Path) -> None:
    merged, work_dir = _write_book(tmp_path)
    llm = FakeLLM()

    with pytest.raises(ValueError, match="output_format"):
        translate_book(merged, work_dir, llm=llm, output_format="epub")

    assert llm.call_count == 0
    assert not (work_dir / "translated_chapters").exists()
