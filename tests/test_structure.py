"""Tests for src.structure (parts, chapters and readings before translation).

The real failure: a part heading and five supplementary readings came out as
numbered chapters, so the PDF printed "Chapter 24" over chapter 22. The
parser had dropped 第N章 from most titles, so the text of a heading cannot
settle it -- the numbering under it, compared across siblings, can.

DeepSeek labels the outline; code only cross-checks. No test here makes a
real API call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.llm.base import BaseLLM
from src.structure import (
    CHAPTER,
    OTHER,
    PART,
    READING,
    analyze_structure,
    apply_structure,
    find_divisions,
    render_outline,
    sequence_problems,
)

BOOK = (
    "# 第篇 电磁学\n\n本篇讲述宏观电磁现象的规律。\n\n"  # row 1: part
    "# 静电场\n\n"  # row 2: chapter 12 -- 第12章 lost by the parser
    "## 12.1 电荷\n\n"
    "### 1. 电荷的种类\n\n正文。\n\n"  # a real subsection: must stay a heading
    "![](IMG_0001)\n图12.1 质子和中子的电荷分布\n\n"
    "$$\nF = k q_1 q_2 / r^2 \\tag{12.1}\n$$\n\n"
    "## 提要\n\n"
    "1. 库仑定律。\n\n"
    "### 2.夫琅禾费衍射\n\n"  # a summary item the parser made a heading
    "### \\* 5. 电容器的充放电\n\n"
    "## 习题\n\n12.1 求电场。\n\n"
    "# 大气电学\n\n"  # row 3: a reading
    "## G.1 晴天大气电场\n\n正文。\n\n"
    "![](IMG_0002)\n图G.2 电场磨\n\n"
    "## 雷暴的电荷和电场\n\n正文。\n\n"
    "# 第13章 电势\n\n"  # row 4: chapter 13, numbered in the title
    "## 13.1 静电场的保守性\n\n正文。\n"
)


def _rows(*rows: tuple[str, int | None, str]) -> str:
    """An outline answer: (kind, number, clean title) per row, in order."""
    return json.dumps(
        {
            "rows": [
                {"row": i, "kind": kind, "number": number, "title": title, "reason": "r"}
                for i, (kind, number, title) in enumerate(rows, start=1)
            ]
        },
        ensure_ascii=False,
    )


CORRECT = _rows(
    (PART, None, "电磁学"), (CHAPTER, 12, "静电场"), (READING, None, "大气电学"), (CHAPTER, 13, "电势")
)


class FakeLLM(BaseLLM):
    """Answers the outline call with ``outline`` and each follow-up with
    ``follow_up(row_number)``; an Exception is raised instead of returned."""

    def __init__(
        self, outline: object, follow_up: Callable[[int], object] | None = None
    ) -> None:
        self.outline = outline
        self.follow_up = follow_up or (lambda _row: RuntimeError("unexpected follow-up"))
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        marker = "The row to re-check: row "
        if marker in user_text:
            row = int(user_text.split(marker, 1)[1].split(".", 1)[0])
            response = self.follow_up(row)
        else:
            response = self.outline
        if isinstance(response, Exception):
            raise response
        return str(response)


def _kinds(report) -> dict[str, str | None]:  # noqa: ANN001
    return {d.title: d.kind for d in report.divisions}


# ── Evidence ────────────────────────────────────────────────────────────────


def test_evidence_counts_sections_captions_and_tags() -> None:
    chapter = find_divisions(BOOK)[1]

    assert chapter.evidence["12"] == 3  # 12.1 section, 图12.1 caption, \tag{12.1}


def test_caption_evidence_does_not_depend_on_the_caption_word() -> None:
    """Every book words its captions differently; the number is what counts."""
    text = "# Electrostatics\n\n![](IMG_0001)\nFigure 12.3 A field\n\n![](IMG_0002)\nFig. 12.4 More\n"

    assert find_divisions(text)[0].evidence["12"] == 2


# ── DeepSeek labels the whole outline in one call ───────────────────────────


def test_one_call_sees_every_row_and_labels_the_book() -> None:
    llm = FakeLLM(CORRECT)

    report = analyze_structure(BOOK, llm=llm)

    assert len(llm.calls) == 1 == report.llm_calls
    _, table, temperature = llm.calls[0]
    assert temperature == 0.0
    assert "row 1: 第篇 电磁学" in table and "row 4: 第13章 电势" in table
    assert "12.x x3" in table and "G.x" in table  # siblings compared on evidence
    assert _kinds(report) == {
        "第篇 电磁学": PART,
        "静电场": CHAPTER,
        "大气电学": READING,
        "第13章 电势": CHAPTER,
    }
    assert all(d.decided_by == "llm" for d in report.divisions)
    assert report.warnings == []


def test_without_an_llm_the_rules_decide_and_the_outline_says_so() -> None:
    report = analyze_structure(BOOK)

    assert _kinds(report)["大气电学"] == READING
    assert all(d.decided_by == "rules" for d in report.divisions)
    assert any("no LLM" in w for w in report.warnings)


def test_a_failed_outline_call_falls_back_to_the_rules() -> None:
    for response in (RuntimeError("timeout"), "not json", '{"rows": "nope"}'):
        report = analyze_structure(BOOK, llm=FakeLLM(response))

        assert _kinds(report)["大气电学"] == READING
        assert any("could not label the outline" in w for w in report.warnings)


def test_a_row_the_model_skipped_takes_the_rules_opinion() -> None:
    partial = json.dumps({"rows": [{"row": 1, "kind": "part", "title": "电磁学"}]})

    report = analyze_structure(BOOK, llm=FakeLLM(partial))

    assert report.divisions[0].decided_by == "llm"
    assert report.divisions[1].decided_by == "rules"
    assert _kinds(report)["静电场"] == CHAPTER


# ── Cross-checks and follow-ups ─────────────────────────────────────────────


# The model calls the reading a chapter, which also breaks the numbering.
MISTAKE = _rows(
    (PART, None, "电磁学"), (CHAPTER, 12, "静电场"), (CHAPTER, 13, "大气电学"), (CHAPTER, 13, "电势")
)


def test_a_conflicting_row_gets_one_follow_up_and_a_helpful_answer_is_kept() -> None:
    answers = {3: json.dumps({"kind": "reading", "reason": "lettered G.x"})}
    llm = FakeLLM(MISTAKE, lambda row: answers.get(row, json.dumps({"kind": "chapter", "number": 13})))

    report = analyze_structure(BOOK, llm=llm)

    assert _kinds(report)["大气电学"] == READING
    assert report.divisions[2].decided_by == "follow-up"
    follow_ups = [text for _, text, _ in llm.calls[1:]]
    assert follow_ups and all("Outline:" in text for text in follow_ups)
    assert any("the numbering says reading" in text for text in follow_ups)
    assert report.warnings == []


def test_a_follow_up_that_does_not_help_is_discarded_and_the_row_flagged() -> None:
    """A second answer that trades one conflict for another is a guess; the
    first was made with the whole outline in view."""
    llm = FakeLLM(MISTAKE, lambda _row: json.dumps({"kind": "other"}))

    report = analyze_structure(BOOK, llm=llm)

    reading = report.divisions[2]
    assert reading.kind == CHAPTER and reading.decided_by == "llm"  # first answer kept
    assert reading.flag
    assert any("check '大气电学'" in w for w in report.warnings)
    assert "CHECK" in render_outline(report)


AMBIGUOUS = (
    "# 电磁感应\n\n## 20.1 法拉第定律\n\n"
    # OCR read the reading's lettered I.1, I.2 as 1.x.
    "# 超导电性\n\n## 1.3 超导体中的电场\n\n![](IMG_0001)\n图 1.2 磁悬浮\n\n"
    "## 1.7 超导的应用\n\n正文。\n\n"
    "# 第21章 麦克斯韦方程组\n\n## 21.1 位移电流\n"
)


def test_an_opinion_the_sequence_rules_out_is_not_a_conflict() -> None:
    """Regression: the rules said "chapter 1" between chapters 20 and 21, the
    follow-up gave in, and a correct "reading" became "chapter 13"."""
    outline = _rows((CHAPTER, 20, "电磁感应"), (READING, None, "超导电性"), (CHAPTER, 21, "麦克斯韦方程组"))
    llm = FakeLLM(outline)

    report = analyze_structure(AMBIGUOUS, llm=llm)

    assert _kinds(report)["超导电性"] == READING
    assert report.llm_calls == 1  # nothing to follow up
    assert report.warnings == []
    assert "rules said chapter" in render_outline(report)


def test_chapter_numbers_that_do_not_run_on_are_found() -> None:
    report = analyze_structure(BOOK, llm=FakeLLM(MISTAKE, lambda _row: RuntimeError("down")))

    assert sequence_problems(report.divisions)
    assert any("next chapter should be 14" in w for w in report.warnings)


def test_a_book_starting_at_another_chapter_than_the_preamble_is_flagged() -> None:
    report = analyze_structure(BOOK, llm=FakeLLM(CORRECT), first_chapter=1)

    assert any("\\setcounter{chapter}{11}" in w for w in report.warnings)
    assert analyze_structure(BOOK, llm=FakeLLM(CORRECT), first_chapter=12).warnings == []


# ── structure.json ──────────────────────────────────────────────────────────


def test_a_manual_entry_overrides_the_model(tmp_path: Path) -> None:
    cache = tmp_path / "structure.json"
    cache.write_text(
        json.dumps({"headings": {"大气电学": {"kind": "other", "by": "manual"}}}),
        encoding="utf-8",
    )

    report = analyze_structure(BOOK, llm=FakeLLM(CORRECT), cache_path=cache)

    assert _kinds(report)["大气电学"] == OTHER
    saved = json.loads(cache.read_text(encoding="utf-8"))["headings"]
    assert saved["大气电学"]["by"] == "manual"  # a human's word survives a re-run
    assert saved["第13章 电势"] == {
        "kind": "chapter", "number": 13, "title": "电势", "by": "llm", "reason": "r"
    }


def test_earlier_labels_are_reused_until_the_outline_changes(tmp_path: Path) -> None:
    cache = tmp_path / "structure.json"
    analyze_structure(BOOK, llm=FakeLLM(CORRECT), cache_path=cache)

    again = FakeLLM(RuntimeError("must not be called"))
    report = analyze_structure(BOOK, llm=again, cache_path=cache)
    assert again.calls == [] and _kinds(report)["大气电学"] == READING

    grown = BOOK + "\n# 附录\n\n常用物理常量。\n"
    relabel = FakeLLM(CORRECT)
    analyze_structure(grown, llm=relabel, cache_path=cache)
    assert len(relabel.calls) == 1  # a new heading means the outline is asked again

    forced = FakeLLM(CORRECT)
    analyze_structure(BOOK, llm=forced, cache_path=cache, force=True)
    assert len(forced.calls) == 1


# ── Rewrite ─────────────────────────────────────────────────────────────────


def test_decided_headings_become_final_latex() -> None:
    text, counts = apply_structure(BOOK, analyze_structure(BOOK, llm=FakeLLM(CORRECT)))
    lines = text.split("\n")

    assert lines[0] == "\\part{电磁学}"
    assert "\\chapter{静电场}" in lines
    assert "\\chapter{电势}" in lines  # the model's clean title: LaTeX numbers it
    assert "\\chapter*{大气电学}" in lines
    assert counts[PART] == 1 and counts[CHAPTER] == 2 and counts[READING] == 1


def test_a_clean_title_that_is_not_part_of_the_heading_is_not_trusted() -> None:
    """The model may only remove a label, never rewrite -- or translate -- a
    title. Otherwise the label is stripped by pattern."""
    invented = CORRECT.replace('"电势"', '"Electric Potential"')

    text, _ = apply_structure(BOOK, analyze_structure(BOOK, llm=FakeLLM(invented)))

    assert "\\chapter{电势}" in text.split("\n")


def test_sections_of_an_unnumbered_chapter_are_starred_and_unnumbered() -> None:
    """A numbered \\section inside \\chapter* would be numbered from the
    chapter before it."""
    text, counts = apply_structure(BOOK, analyze_structure(BOOK, llm=FakeLLM(CORRECT)))

    assert "\\section*{晴天大气电场}" in text
    assert "\\section*{雷暴的电荷和电场}" in text
    assert counts["unnumbered section"] == 2
    assert "## 13.1 静电场的保守性" in text  # a real chapter's section is left alone


def test_summary_items_become_paragraphs_but_real_subsections_stay() -> None:
    text, counts = apply_structure(BOOK, analyze_structure(BOOK))
    lines = text.split("\n")

    assert "2.夫琅禾费衍射" in lines
    assert "*5. 电容器的充放电" in lines
    assert "### 1. 电荷的种类" in lines  # under 12.1, not under 提要
    assert "## 习题" in lines  # the summary ends at the next section
    assert counts["summary item"] == 2


def test_an_undecided_heading_is_left_as_markdown() -> None:
    text = "# 前言\n\n本书是……\n"

    rewritten, _ = apply_structure(text, analyze_structure(text))

    assert rewritten.startswith("# 前言")


def test_the_outline_shows_one_line_per_heading_and_whether_the_rules_agree() -> None:
    outline = render_outline(analyze_structure(BOOK, llm=FakeLLM(CORRECT))).split("\n")

    assert len(outline) == 4
    assert "PART" in outline[0]
    assert "12" in outline[1] and "(rules agree)" in outline[1]
    assert "*" in outline[2]
