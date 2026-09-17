"""Tests for src.lint.

Three jobs here.

1. The kit checks -- CJK residue, truncation, image tokens -- each catch a
   failure the others cannot.
2. The LaTeX checks each have to fire on a fixture carrying that one defect
   and stay silent on clean input. Every one of them is a compile failure
   that is invisible by reading the file, which is the whole reason they are
   explicit checks.
3. The historical source diagnostics must still *detect* the defects the
   current parser no longer produces (`$x $` padding, `$$$$` welds,
   unbalanced braces, over-tabbed arrays). Those fixtures are the guard
   against a future parser regression slipping through unnoticed, which is
   why the checks were kept after the normalizer passes were dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.lint import (
    Severity,
    apply_auto_fixes,
    brace_imbalance,
    cjk_ratio,
    lint_chunk_pair,
    lint_kit,
    lint_latex,
    lint_markdown,
)

# Long enough that the length heuristic cannot fire: a truncation check
# comparing this against a same-length source sees a ratio of ~1.0. If this
# were short, the truncation finding would appear and the residue check would
# never be the thing under test.
UNTRANSLATED_CHUNK = (
    "## 12.1 电荷\n\n"
    "人们很早就发现，用丝绸摩擦过的玻璃棒能够吸引轻小的物体，这种现象叫做摩擦起电。"
    "带电的物体所带的电荷有两种，一种叫正电荷，另一种叫负电荷。同种电荷互相排斥，"
    "异种电荷互相吸引。电荷的多少叫做电荷量，在国际单位制中，电荷量的单位是库仑。\n\n"
    "实验证明，在一个与外界没有电荷交换的系统内，正负电荷的代数和在任何物理过程中"
    "始终保持不变，这就是电荷守恒定律。电荷守恒定律是物理学中最基本的定律之一。\n\n"
    "电荷还有一个重要性质，就是它的量子性。所有带电体所带的电荷量都是基本电荷 $e$ "
    "的整数倍。基本电荷的数值是 $e = 1.602 \\times 10^{-19}$ 库仑。\n"
)

TRANSLATED_CHUNK = (
    "\\section{Electric Charge}\n\n"
    "It was discovered long ago that a glass rod rubbed with silk attracts light "
    "objects; this phenomenon is called triboelectric charging. The charge carried "
    "by a charged body comes in two kinds, one called positive charge and the other "
    "negative charge. Like charges repel one another and unlike charges attract. "
    "The amount of charge is called the quantity of electric charge, and in SI its "
    "unit is the coulomb.\n\n"
    "Experiment shows that within a system exchanging no charge with its "
    "surroundings, the algebraic sum of positive and negative charge stays constant "
    "throughout any physical process. This is the law of conservation of charge, one "
    "of the most fundamental laws in physics.\n\n"
    "Charge has another important property: it is quantized. The charge on any "
    "charged body is an integer multiple of the elementary charge $e$, whose value "
    "is $e = 1.602 \\times 10^{-19}$ coulomb.\n"
)


def _checks(findings, severity: Severity | None = None) -> set[str]:
    return {
        f.check for f in findings if severity is None or f.severity is severity
    }


# A clean LaTeX translation: every new check below has to stay silent on it,
# or the check is useless -- a linter that fires on correct output is one the
# user learns to pass --no-lint to.
CLEAN_LATEX = (
    "\\section{Electric Charge}\n\n"
    "Charge is quantized: the charge on any body is an integer multiple of "
    "the elementary charge $e$, whose value is $1.602 \\times 10^{-19}$ C.\n\n"
    "\\begin{equation}\nF = \\frac{q_1 q_2}{4 \\pi \\varepsilon_0 r^2} \\tag{12.1}\n"
    "\\end{equation}\n\n"
    "\\bookfig{images/field.jpg}{Field of a point charge}\n\n"
    "\\begin{example}\n"
    "Two charges of $1\\,\\mathrm{C}$ sit $1\\,\\mathrm{m}$ apart.\n"
    "\\textbf{Solution.} Apply Eq. (12.1).\n"
    "\\end{example}\n\n"
    "\\begin{itemize}\n\\item Like charges repel.\n\\item Unlike charges attract.\n"
    "\\end{itemize}\n"
)


# ── LaTeX checks: each fires on its own defect, and only on it ──────────────


def test_clean_latex_needs_no_review() -> None:
    report = lint_latex(CLEAN_LATEX)

    assert _checks(report.findings, Severity.REVIEW) == set(), report.render()
    assert _checks(report.findings, Severity.AUTO_FIX) == set(), report.render()
    assert report.exit_code == 0


def test_an_unbalanced_environment_is_flagged() -> None:
    report = lint_latex("\\begin{example}\nA worked example.\n")

    assert "environment-balance" in _checks(report.findings, Severity.REVIEW)
    assert report.exit_code == 1


def test_a_crossed_environment_pair_is_flagged() -> None:
    report = lint_latex("\\begin{itemize}\n\\item one\n\\end{enumerate}\n")

    findings = [f for f in report.findings if f.check == "environment-balance"]
    assert len(findings) == 1
    assert "\\end{enumerate}" in findings[0].message


def test_an_undefined_environment_is_flagged_by_name_and_line() -> None:
    """The model invented `solution` during testing. Nothing defines it, so
    the build stops there."""
    report = lint_latex(f"{CLEAN_LATEX}\n\\begin{{solution}}\nBecause.\n\\end{{solution}}\n")

    findings = [f for f in report.findings if f.check == "undefined-environment"]
    assert len(findings) == 1
    assert "solution" in findings[0].message
    assert findings[0].line == CLEAN_LATEX.count("\n") + 2


def test_preamble_leakage_into_a_chunk_is_flagged() -> None:
    report = lint_latex(
        "\\documentclass{book}\n\\usepackage{amsmath}\n"
        "\\begin{document}\n\\section{One}\n\\end{document}\n"
    )

    leakage = [f for f in report.findings if f.check == "preamble-leakage"]
    assert {f.line for f in leakage} == {1, 2, 3, 5}
    assert all(f.severity is Severity.REVIEW for f in leakage)


def test_unbalanced_braces_are_flagged_with_a_line_number() -> None:
    report = lint_latex(f"{CLEAN_LATEX}\n\\section{{Electric Field\n")

    findings = [f for f in report.findings if f.check == "brace-balance"]
    assert len(findings) == 1
    assert "unclosed" in findings[0].message


def test_escaped_braces_do_not_count_toward_the_latex_balance() -> None:
    report = lint_latex("A literal \\{ and \\} in prose.\n")

    assert "brace-balance" not in _checks(report.findings, Severity.REVIEW)


def test_odd_inline_dollar_parity_is_flagged_per_paragraph() -> None:
    report = lint_latex("The value $x is undefined.\n\nA later paragraph.\n")

    findings = [f for f in report.findings if f.check == "inline-parity"]
    assert len(findings) == 1
    assert findings[0].line == 1


def test_a_duplicated_section_number_is_flagged_and_auto_fixable() -> None:
    """Output like "12.1 12.1 Electric Charge": readable, so nothing fails."""
    report = lint_latex("\\section{12.1 Electric Charge}\n")

    findings = [f for f in report.findings if f.check == "duplicated-number"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.AUTO_FIX
    assert report.exit_code == 0  # auto-fixable, so it does not gate the build


def test_a_duplicated_figure_caption_label_is_flagged() -> None:
    report = lint_latex(
        "\\begin{figure}\n\\caption{Figure 12.3 Field distribution}\n\\end{figure}\n"
    )

    findings = [f for f in report.findings if f.check == "duplicated-number"]
    assert len(findings) == 1
    assert "Figure 12.3" in findings[0].message


def test_a_duplicated_table_caption_label_is_flagged() -> None:
    report = lint_latex("\\caption{Table 12.1 Relative permittivity}\n")

    assert [f.check for f in report.findings if f.check == "duplicated-number"] == [
        "duplicated-number"
    ]


def test_a_numbered_subsection_is_flagged() -> None:
    report = lint_latex("\\subsection{1. Types of Charge}\n")

    assert "duplicated-number" in _checks(report.findings, Severity.AUTO_FIX)


def test_titles_that_merely_begin_with_a_digit_are_left_alone() -> None:
    """`3D` is not a section number, and neither is a caption that says
    "Figure" without one."""
    report = lint_latex(
        "\\section{3D Charge Distributions}\n\n"
        "\\caption{Figure of the apparatus}\n\n"
        "\\section{Electric Charge}\n"
    )

    assert "duplicated-number" not in _checks(report.findings)


def test_the_number_auto_fix_strips_it_and_leaves_clean_titles_untouched() -> None:
    fixed, count = apply_auto_fixes(
        "\\section{12.1 Electric Charge}\n"
        "\\section{Electric Charge}\n"
        "\\caption{Figure 12.3 Field distribution}\n"
        "\\subsection{1. Types of Charge}\n"
    )

    assert count == 3
    assert fixed == (
        "\\section{Electric Charge}\n"
        "\\section{Electric Charge}\n"
        "\\caption{Field distribution}\n"
        "\\subsection{Types of Charge}\n"
    )


def test_the_number_auto_fix_keeps_nested_groups_intact() -> None:
    fixed, count = apply_auto_fixes("\\caption{Table 12.1 Values of $E_{x}$ measured}\n")

    assert count == 1
    assert fixed == "\\caption{Values of $E_{x}$ measured}\n"


def test_a_dangling_argument_command_in_latex_math_is_flagged() -> None:
    report = lint_latex("The field is $8.3 \\times 10^{-21} \\mathrm$ T.\n")

    assert "dangling-command" in _checks(report.findings, Severity.REVIEW)


def test_math_inside_a_latex_only_delimiter_is_still_checked() -> None:
    """\\[ ... \\] and equation environments are invisible to the source-side
    dollar scanner, so the math checks would skip most of the book."""
    report = lint_latex("Display math:\n\\[\n\\frobnicate{x}\n\\]\n")

    assert "unknown-command" in _checks(report.findings, Severity.REVIEW)


def test_unknown_commands_in_latex_are_reported_never_fixed() -> None:
    text = "\\begin{equation}\n\\frobnicate{x}\n\\end{equation}\n"
    report = lint_latex(text)

    assert "unknown-command" in _checks(report.findings, Severity.REVIEW)
    assert apply_auto_fixes(text) == (text, 0)


def test_bookfig_paths_are_resolved_on_disk(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "there.jpg").write_bytes(b"\x89PNG")
    text = (
        "\\bookfig{images/there.jpg}{There}\n\n"
        "\\bookfigtwo{images/there.jpg}{(a)}{images/gone.jpg}{(b)}{Both}\n"
    )

    report = lint_latex(text, base_dir=tmp_path)
    missing = [f for f in report.findings if f.check == "image-paths"]

    assert len(missing) == 1
    assert missing[0].severity is Severity.REVIEW
    assert "images/gone.jpg" in missing[0].context


# ── Historical defects: still detected, never repaired ──────────────────────


def test_welded_dollar_delimiters_are_detected() -> None:
    report = lint_markdown("Some text $$$$y = 2x$$$$ more text.\n")
    assert "welded-delimiters" in _checks(report.findings, Severity.REVIEW)


def test_odd_display_delimiters_are_detected() -> None:
    report = lint_markdown("$$\ny = 2x\n\nAnd the block never closes.\n")
    assert "display-parity" in _checks(report.findings, Severity.REVIEW)


def test_delimiter_whitespace_is_detected() -> None:
    report = lint_markdown(r"A span $n S { \mathrm { d } } l $ that does not read as math.")
    assert "delimiter-whitespace" in _checks(report.findings, Severity.REVIEW)


def test_escaped_closing_dollar_is_detected() -> None:
    report = lint_markdown("The value is $x = 2\\$ and then prose.\n")
    assert "escaped-dollar" in _checks(report.findings, Severity.REVIEW)


def test_unbalanced_braces_inside_math_are_detected() -> None:
    report = lint_markdown("$$\n\\frac { a } { b \n$$\n")
    assert "brace-balance" in _checks(report.findings, Severity.REVIEW)


def test_escaped_braces_do_not_count_toward_the_balance() -> None:
    assert brace_imbalance(r"\{ a \}") == 0
    assert brace_imbalance(r"{ a }") == 0
    assert brace_imbalance(r"{ a ") == 1
    assert brace_imbalance(r" a }") == -1


def test_array_rows_with_too_many_tabs_are_detected() -> None:
    text = "$$\n\\begin{array} { r l r l } { { 27 } } & { a } & { b } & { c } & { d }\n\\end{array}\n$$\n"
    report = lint_markdown(text)
    assert "array-columns" in _checks(report.findings, Severity.REVIEW)


def test_ampersands_inside_braces_are_not_counted() -> None:
    text = "$$\n\\begin{array} { r l } { a & b } & { c }\n\\end{array}\n$$\n"
    report = lint_markdown(text)
    assert "array-columns" not in _checks(report.findings, Severity.REVIEW)


def test_multiple_tags_in_one_math_span_are_detected() -> None:
    """amsmath hard-errors on two \\tag commands, so the build fails outright.

    Seen on a real book: the parser read a stray number in the scan margin as
    a second tag and emitted `\\tag {30}\\tag{30.12}`.
    """
    text = "$$\n\\begin{array}{l} E = 180 \\mathrm{MeV} \\end{array} \\tag {30}\\tag{30.12}\n$$\n"
    report = lint_markdown(text)

    findings = [f for f in report.findings if f.check == "multiple-tags"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.REVIEW
    assert "'30'" in findings[0].message and "'30.12'" in findings[0].message
    assert report.exit_code == 1


def test_a_single_tag_is_not_flagged() -> None:
    """570 equations in the reference book carry exactly one tag."""
    report = lint_markdown("$$\ne = 1.602 \\times 10^{-19} \\tag{12.1}\n$$\n")
    assert "multiple-tags" not in _checks(report.findings, Severity.REVIEW)


def test_tags_in_separate_equations_are_not_flagged() -> None:
    """One tag each across two display blocks is correct, not a duplicate."""
    text = "$$\na = 1 \\tag{1.1}\n$$\n\nProse between them.\n\n$$\nb = 2 \\tag{1.2}\n$$\n"
    report = lint_markdown(text)
    assert "multiple-tags" not in _checks(report.findings, Severity.REVIEW)


def test_multiple_tags_are_never_auto_fixed() -> None:
    """Which tag is the real equation number needs a human; only report."""
    text = "$$\nx = 1 \\tag {30}\\tag{30.12}\n$$\n"
    assert apply_auto_fixes(text) == (text, 0)


def test_dangling_argument_command_is_detected() -> None:
    report = lint_markdown("The field is $8.3 { \\times } 10 ^ { - 21 } \\ \\mathrm$ T.\n")
    assert "dangling-command" in _checks(report.findings, Severity.REVIEW)


def test_a_clean_document_reports_none_of_the_historical_defects() -> None:
    text = (
        "# Chapter 1\n\n## 1.1 Section\n\n### 1.1.1 Subsection\n\n"
        "Inline math $10^{-20} \\mathrm{~m}$ and a display block:\n\n"
        "$$\ne = 1.602 \\times 10^{-19} \\tag{12.1}\n$$\n\n"
        "More prose.\n"
    )
    report = lint_markdown(text)

    review = _checks(report.findings, Severity.REVIEW)
    assert review == set()
    assert report.exit_code == 0


# ── Report, never auto-fix ──────────────────────────────────────────────────


def test_unknown_commands_are_reported_not_fixed() -> None:
    text = "$$\n\\frobnicate{x}\n$$\n"
    report = lint_markdown(text)

    assert "unknown-command" in _checks(report.findings, Severity.REVIEW)
    assert apply_auto_fixes(text)[0] == text  # untouched


def test_doubled_known_command_is_the_one_safe_auto_fix() -> None:
    text = "The unit is $1 \\mathrmmathrm{T}$ here.\n"
    report = lint_markdown(text)

    assert "doubled-command" in _checks(report.findings, Severity.AUTO_FIX)
    fixed, count = apply_auto_fixes(text)
    assert count == 1
    assert "\\mathrm{T}" in fixed
    assert "mathrmmathrm" not in fixed


def test_a_doubled_unknown_command_is_not_auto_fixed() -> None:
    text = "$\\frobfrob{x}$"
    assert apply_auto_fixes(text) == (text, 0)


# ── Headings and images ─────────────────────────────────────────────────────


def test_a_flat_outline_is_flagged() -> None:
    """The real defect: 422 of 442 headings at level 2, so nothing tells a
    chapter, a section and a worked example apart."""
    text = "\n\n".join(f"## Section {i}\n\nBody {i}." for i in range(30))
    report = lint_markdown(text)

    flat = [f for f in report.findings if f.check == "heading-levels" and f.severity is Severity.REVIEW]
    assert len(flat) == 1
    assert "flat" in flat[0].message


def test_a_real_hierarchy_is_not_flagged() -> None:
    text = "\n\n".join(
        f"# Chapter {i}\n\n## {i}.1 Section\n\n### {i}.1.1 Sub\n\nBody." for i in range(10)
    )
    report = lint_markdown(text)
    assert "heading-levels" not in _checks(report.findings, Severity.REVIEW)


def test_unresolvable_image_paths_are_flagged(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "there.jpg").write_bytes(b"x")
    text = "![a](images/there.jpg)\n\n![b](images/missing.jpg)\n"

    report = lint_markdown(text, base_dir=tmp_path)
    missing = [f for f in report.findings if f.check == "image-paths" and f.severity is Severity.REVIEW]

    assert len(missing) == 1
    assert "missing.jpg" in missing[0].context


# ── Chunk-level checks ──────────────────────────────────────────────────────


def test_cjk_residue_catches_a_full_length_untranslated_chunk() -> None:
    """An untranslated chunk is *full length*, so only the residue check
    finds it. This fixture is deliberately long enough that the truncation
    heuristic cannot fire and take the credit."""
    findings = lint_chunk_pair("001", UNTRANSLATED_CHUNK, UNTRANSLATED_CHUNK)

    checks = _checks(findings)
    assert "cjk-residue" in checks
    assert "truncated-chunk" not in checks  # the residue check stands alone


def test_a_properly_translated_chunk_is_clean() -> None:
    assert lint_chunk_pair("001", UNTRANSLATED_CHUNK, TRANSLATED_CHUNK) == []


def test_truncation_floor_catches_a_severely_cut_off_reply() -> None:
    """The absolute floor: under 40% of the source's own length."""
    truncated = TRANSLATED_CHUNK[:80]
    findings = lint_chunk_pair("001", UNTRANSLATED_CHUNK, truncated)

    assert "truncated-chunk" in _checks(findings)


def test_expansion_outlier_catches_a_truncation_the_floor_misses(tmp_path: Path) -> None:
    """A reply cut off halfway is still well above 40% of a CJK source's
    length, so only the book's own median expansion ratio finds it."""
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir(parents=True)

    for index in range(1, 6):
        (kit / "chunks" / f"{index:03d}.md").write_text(
            UNTRANSLATED_CHUNK, encoding="utf-8"
        )
        # Chunk 3 comes back at roughly a third of the length the others do,
        # but still at ~100% of its source's length -- invisible to the floor.
        body = TRANSLATED_CHUNK[:300] if index == 3 else TRANSLATED_CHUNK
        (kit / "translated" / f"{index:03d}.tex").write_text(body, encoding="utf-8")

    findings = lint_kit(kit).findings
    truncation = [f for f in findings if f.check == "truncated-chunk"]

    assert len(truncation) == 1
    assert "chunk 003" in truncation[0].message


def test_expansion_outlier_stays_quiet_on_a_consistent_book(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir(parents=True)

    for index in range(1, 6):
        (kit / "chunks" / f"{index:03d}.md").write_text(
            UNTRANSLATED_CHUNK, encoding="utf-8"
        )
        (kit / "translated" / f"{index:03d}.tex").write_text(
            TRANSLATED_CHUNK, encoding="utf-8"
        )

    assert [f for f in lint_kit(kit).findings if f.check == "truncated-chunk"] == []


def test_lost_and_invented_image_tokens_are_both_caught() -> None:
    source = f"{UNTRANSLATED_CHUNK}\n\n![](IMG_0001)\n\n![](IMG_0002)\n"
    translation = f"{TRANSLATED_CHUNK}\n\n![](IMG_0001)\n\n![](IMG_0009)\n"

    findings = lint_chunk_pair("001", source, translation)
    messages = " ".join(f.message for f in findings)

    assert "lost 1 image token" in messages
    assert "invented 1 image token" in messages


def test_cjk_ratio_ignores_mathematics() -> None:
    """A chunk that is 40% formulas would otherwise dilute its own score."""
    mostly_math = "$$\n\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta\n$$\n\n电荷\n"
    assert cjk_ratio(mostly_math) > 0.5


# ── Whole-kit lint ──────────────────────────────────────────────────────────


def _make_kit(tmp_path: Path, translations: dict[str, str]) -> Path:
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir(parents=True)
    (kit / "images").mkdir(parents=True)
    (kit / "chunks" / "001.md").write_text(UNTRANSLATED_CHUNK, encoding="utf-8")
    for name, body in translations.items():
        (kit / "translated" / name).write_text(body, encoding="utf-8")
    return kit


def test_lint_kit_exits_zero_when_clean(tmp_path: Path) -> None:
    kit = _make_kit(tmp_path, {"001.tex": TRANSLATED_CHUNK})
    report = lint_kit(kit)

    assert not report.needs_review, report.render()
    assert report.exit_code == 0


def test_lint_kit_exits_non_zero_when_review_is_needed(tmp_path: Path) -> None:
    kit = _make_kit(tmp_path, {"001.tex": UNTRANSLATED_CHUNK})
    report = lint_kit(kit)

    assert report.needs_review
    assert report.exit_code == 1
    assert "cjk-residue" in _checks(report.findings, Severity.REVIEW)


def test_lint_kit_reports_a_missing_translation(tmp_path: Path) -> None:
    kit = _make_kit(tmp_path, {})
    report = lint_kit(kit)

    assert "missing-translation" in _checks(report.findings, Severity.REVIEW)
    assert report.exit_code == 1


def test_lint_kit_strips_a_wrapping_code_fence(tmp_path: Path) -> None:
    """AI Studio adds ```latex around the reply regardless of the prompt."""
    kit = _make_kit(tmp_path, {"001.tex": f"```latex\n{TRANSLATED_CHUNK}\n```"})
    report = lint_kit(kit)

    assert not report.needs_review, report.render()


def test_lint_kit_flags_images_that_never_appear_in_the_translation(tmp_path: Path) -> None:
    kit = _make_kit(tmp_path, {"001.tex": TRANSLATED_CHUNK})
    (kit / "image_map.json").write_text(
        json.dumps({"IMG_0000": "images/a.jpg", "IMG_0001": "images/b.jpg"}), encoding="utf-8"
    )
    report = lint_kit(kit)

    assert "image-map" in _checks(report.findings, Severity.REVIEW)


def test_lint_kit_resolves_image_tokens_before_checking_paths(tmp_path: Path) -> None:
    """Saved replies still carry IMG_nnnn tokens. Linting them as-is would
    report every token as an image that does not resolve."""
    kit = _make_kit(
        tmp_path, {"001.tex": TRANSLATED_CHUNK + "\n\n\\bookfig{IMG_0000}{Field}\n"}
    )
    # The token must be in the source too, or the image-token check correctly
    # reports it as invented and this test passes for the wrong reason.
    (kit / "chunks" / "001.md").write_text(
        f"{UNTRANSLATED_CHUNK}\n\n![图](IMG_0000)\n", encoding="utf-8"
    )
    (kit / "images" / "real.jpg").write_bytes(b"\x89PNG")
    (kit / "image_map.json").write_text(
        json.dumps({"IMG_0000": "images/real.jpg"}), encoding="utf-8"
    )

    report = lint_kit(kit)
    paths = [f for f in report.findings if f.check == "image-paths"]

    assert all(f.severity is Severity.INFO for f in paths), report.render()
    assert not report.needs_review, report.render()


def test_lint_kit_still_flags_a_token_whose_file_is_absent(tmp_path: Path) -> None:
    kit = _make_kit(
        tmp_path, {"001.tex": TRANSLATED_CHUNK + "\n\n\\bookfig{IMG_0000}{Field}\n"}
    )
    (kit / "chunks" / "001.md").write_text(
        f"{UNTRANSLATED_CHUNK}\n\n![图](IMG_0000)\n", encoding="utf-8"
    )
    (kit / "image_map.json").write_text(
        json.dumps({"IMG_0000": "images/gone.jpg"}), encoding="utf-8"
    )

    report = lint_kit(kit)
    missing = [
        f for f in report.findings if f.check == "image-paths" and f.severity is Severity.REVIEW
    ]

    assert len(missing) == 1
    assert "images/gone.jpg" in missing[0].context


def test_lint_kit_on_a_missing_chunks_dir_is_reported_not_raised(tmp_path: Path) -> None:
    report = lint_kit(tmp_path / "nope")
    assert report.exit_code == 1
    assert "kit-layout" in _checks(report.findings, Severity.REVIEW)
