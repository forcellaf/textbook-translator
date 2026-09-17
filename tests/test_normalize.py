"""Tests for src.normalize (split-heading repair + diagnostics).

Fixtures are built from the real defect signatures measured on the test book,
so each one fails without its fix. The negative fixtures matter just as much:
``提要`` and ``习题`` appear 19 times each as legitimate two-character
headings, and a naive "merge short adjacent headings" rule would corrupt
every one of them.
"""

from __future__ import annotations

from src.normalize import (
    display_parity,
    dollar_run_histogram,
    find_delimiter_whitespace,
    fix_inline_math_spacing,
    merge_split_headings,
    normalize,
)


def _kinds(changes) -> list[str]:
    return [c.kind for c in changes]


# ── Pattern 1: chapter number torn off its title ────────────────────────────


def test_chapter_number_merges_with_the_title_that_follows_it() -> None:
    text = "## 第13章\n\n## 电势\n\n正文。\n"
    result, changes, _ = merge_split_headings(text)

    assert "## 第13章 电势" in result
    assert "## 电势" not in result.replace("## 第13章 电势", "")
    assert _kinds(changes) == ["chapter-title-split"]


def test_chapter_merge_takes_the_shallower_of_the_two_levels() -> None:
    """`## 第17章` + `# 磁场和它的源` is one heading; keep the prominent level."""
    text = "## 第17章\n\n# 磁场和它的源\n\n正文。\n"
    result, changes, _ = merge_split_headings(text)

    assert "# 第17章 磁场和它的源" in result
    assert "## 第17章" not in result
    assert len(changes) == 1


def test_spaced_chapter_number_also_merges() -> None:
    text = "## 第 13 章\n\n## 电势\n"
    result, changes, _ = merge_split_headings(text)
    assert "## 第 13 章 电势" in result
    assert len(changes) == 1


def test_chapter_number_not_adjacent_to_a_heading_is_left_alone() -> None:
    """The back-matter answers section lists `## 第 12 章` per chapter, each
    followed by pages of answers -- not by a torn-off title."""
    text = "## 第 12 章\n\n12.1 答案是 $x$。\n\n12.2 更多答案。\n\n## 第 13 章\n\n13.1 答案。\n"
    result, changes, _ = merge_split_headings(text)

    assert result == text
    assert changes == []


def test_chapter_number_followed_by_a_numbered_section_is_left_alone() -> None:
    text = "## 第13章\n\n## 13.1 电势能\n"
    result, changes, _ = merge_split_headings(text)

    assert result == text
    assert changes == []


# ── Pattern 2: part title torn into fragments ───────────────────────────────


def test_bare_fragments_merge_into_one_part_title() -> None:
    text = "## 第\n\n## 篇\n\n## 电磁学\n\n本篇讲解电磁学。\n"
    result, changes, _ = merge_split_headings(text)

    assert "## 第篇 电磁学" in result
    assert _kinds(changes) == ["part-title-fragments"]


def test_a_lone_fragment_merges_with_the_title_after_it() -> None:
    text = "## 篇\n\n## 光学\n\n正文。\n"
    result, changes, _ = merge_split_headings(text)

    assert "## 篇 光学" in result
    assert len(changes) == 1


def test_a_fragment_run_with_its_number_intact_merges_correctly() -> None:
    text = "## 第3\n\n## 篇\n\n## 电磁学\n"
    result, _, _ = merge_split_headings(text)
    assert "## 第3篇 电磁学" in result


def test_missing_part_ordinal_is_reported_not_invented() -> None:
    text = "## 第\n\n## 篇\n\n## 电磁学\n"
    _, _, notes = merge_split_headings(text)

    ordinal_notes = [n for n in notes if n.kind == "part-ordinal-missing"]
    assert len(ordinal_notes) == 1
    assert "第篇 电磁学" in ordinal_notes[0].text


def test_a_fragment_with_no_title_after_it_is_reported_not_merged() -> None:
    text = "## 篇\n\n正文，不是标题。\n"
    result, changes, notes = merge_split_headings(text)

    assert result == text
    assert changes == []
    assert any(n.kind == "unmerged-fragment" for n in notes)


# ── The critical negative: legitimate short headings ────────────────────────


def test_recurring_two_character_headings_are_left_completely_unchanged() -> None:
    """`提要` and `习题` appear 19 times each; merging them corrupts the book."""
    text = (
        "## 12.1 电荷\n\n正文一。\n\n"
        "## 提要\n\n本章要点。\n\n"
        "## 习题\n\n12.1 计算 $q$。\n\n"
        "## 13.1 电势\n\n正文二。\n\n"
        "## 提要\n\n本章要点。\n\n"
        "## 习题\n\n13.1 计算 $V$。\n"
    )
    result, changes, _ = merge_split_headings(text)

    assert result == text
    assert changes == []


def test_adjacent_legitimate_short_headings_are_not_merged() -> None:
    """`## 目录` sits immediately before `## CONTENTS`, and `## 提要`
    immediately before `## 1. 相干光`. Adjacency alone is never enough."""
    text = "## 目录\n\n## CONTENTS\n\n## 提要\n\n## 1. 相干光\n"
    result, changes, _ = merge_split_headings(text)

    assert result == text
    assert changes == []


def test_unmerged_short_headings_are_reported_for_review() -> None:
    text = "## 提要\n\n正文。\n\n## 习题\n\n更多正文。\n"
    _, _, notes = merge_split_headings(text)

    short = [n for n in notes if n.kind == "short-heading"]
    assert sorted(n.text for n in short) == ["习题", "提要"]


# ── Idempotency ─────────────────────────────────────────────────────────────


def test_repairs_are_idempotent() -> None:
    text = "## 第\n\n## 篇\n\n## 电磁学\n\n## 第13章\n\n## 电势\n\n## 提要\n\n正文。\n"
    once, first_changes, _ = merge_split_headings(text)
    twice, second_changes, _ = merge_split_headings(once)

    assert len(first_changes) == 2
    assert twice == once
    assert second_changes == []


# ── Diagnostics ─────────────────────────────────────────────────────────────


def test_dollar_run_histogram_detects_welded_delimiters() -> None:
    """The old parser's `$$$$` weld. Gone from the current output; the check
    stays so a parser regression cannot reintroduce it unnoticed."""
    clean = "Inline $x$ and display:\n\n$$\ny = 2\n$$\n"
    assert dollar_run_histogram(clean) == {1: 2, 2: 2}

    welded = "$$$$y = 2$$$$"
    assert 4 in dollar_run_histogram(welded)


def test_display_parity_detects_an_unclosed_block() -> None:
    assert display_parity("$$\nx\n$$\n") == (2, True)
    assert display_parity("$$\nx\n")[1] is False


def test_delimiter_whitespace_is_detected_but_not_repaired_by_default() -> None:
    """Zero of the current parser's 6,641 inline spans hit this, so it is a
    diagnostic. The detector must still find it if a parser regresses."""
    text = r"The span $n S { \mathrm { d } } l $ leaks into text mode."

    assert len(find_delimiter_whitespace(text)) == 1
    assert normalize(text).text == text  # untouched by default
    assert normalize(text).diagnostics["inline_spans_with_padding"] == 1


def test_math_spacing_fix_is_available_behind_the_flag() -> None:
    text = r"The span $ x + y $ is padded."
    repaired, count = fix_inline_math_spacing(text)

    assert count == 1
    assert "$x + y$" in repaired
    assert normalize(text, fix_math_spacing=True).text == repaired


def test_math_spacing_fix_leaves_display_math_alone() -> None:
    """Display math is still character-spaced by the current parser. That is
    cosmetic only -- with no delimiter-adjacent whitespace it compiles
    correctly -- so this pass must not touch it."""
    text = "$$\n e = 1. 6 0 2 \\times 1 0 ^ {- 1 9} \n$$\n"
    repaired, count = fix_inline_math_spacing(text)

    assert repaired == text
    assert count == 0


def test_normalize_reports_the_heading_level_distribution() -> None:
    text = "# A\n\n## B\n\n## C\n"
    assert normalize(text).diagnostics["heading_level_counts"] == {1: 1, 2: 2}
