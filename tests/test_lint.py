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


def test_math_environments_are_not_reported_as_undefined() -> None:
    """Regression: `array` was flagged ~1,000 times on a real book. Maths is
    copied through verbatim, so whatever the source used arrives untouched --
    and array, cases and the matrix environments all compile fine."""
    report = lint_latex(
        "\\[\n\\begin{array}{l} a = b \\\\ c = d \\end{array}\n\\]\n\n"
        "\\begin{equation}\n\\begin{cases} x & y \\\\ z & w \\end{cases}\n\\end{equation}\n\n"
        "\\[\n\\begin{pmatrix} 1 & 0 \\end{pmatrix}\n\\]\n"
    )

    assert "undefined-environment" not in _checks(report.findings), report.render()


def test_an_invented_environment_is_still_flagged_alongside_them() -> None:
    """The list got longer, so this is the guard that it did not get useless:
    amsthm is not loaded, so `theorem` and `proof` do not exist here."""
    for invented in ("solution", "theorem", "proof", "note", "definition"):
        report = lint_latex(f"\\begin{{{invented}}}\nBecause.\n\\end{{{invented}}}\n")
        assert "undefined-environment" in _checks(report.findings, Severity.REVIEW), invented


def test_literal_unicode_that_pdflatex_cannot_typeset_is_flagged() -> None:
    """A real book carried 482 of these. Each one is a fatal error, and
    pdflatex reports one per compile."""
    report = lint_latex("where ε₀ = 8.85 and the angle is θ.\n")

    findings = [f for f in report.findings if f.check == "unicode-character"]
    assert {f.severity for f in findings} == {Severity.REVIEW}
    assert any("U+03B5" in f.message for f in findings)
    assert any("U+03B8" in f.message for f in findings)
    assert report.exit_code == 1


def test_a_stray_cjk_character_is_flagged_even_below_the_residue_threshold() -> None:
    """cjk-residue measures a ratio: six characters in a 700,000-character
    book is 0.01% and invisible to it, and still six dead builds."""
    report = lint_latex(CLEAN_LATEX * 4 + "\nThe value 为 is given.\n")

    assert "cjk-residue" not in _checks(report.findings)  # ratio check is elsewhere
    assert any(
        f.check == "unicode-character" and "U+4E3A" in f.message for f in report.findings
    )


def test_ordinary_latin1_and_real_punctuation_are_left_alone() -> None:
    """inputenc handles these, so flagging them would be noise: 25 °C, ±5%,
    a 2 × 3 grid, an em dash and curly quotes."""
    report = lint_latex(
        "At 25 °C the value is 3.2 ±5%, in a 2 × 3 grid — "
        "the “standard” case…\n"
    )

    assert "unicode-character" not in _checks(report.findings), report.render()


def test_zero_width_characters_are_deleted_not_reviewed() -> None:
    """Invisible in every editor, fatal to pdflatex, and meaningless -- so
    deleting is the only repair there is."""
    text = "The field ​strength E​ is given.\n"
    report = lint_latex(text)

    assert "zero-width-character" in _checks(report.findings, Severity.AUTO_FIX)
    assert "unicode-character" not in _checks(report.findings)

    fixed, count = apply_auto_fixes(text)
    assert count == 2
    assert fixed == "The field strength E is given.\n"


def test_a_display_delimiter_that_lost_its_backslash_is_flagged() -> None:
    """Measured on a real book: 971 blocks opened with `\\[`, 94 with a bare
    `[`. The bracket typesets literally and the maths after it runs in text
    mode, so the first \\frac fails."""
    report = lint_latex("Then\n\n[\n\\begin{array}{l} a = b \\end{array}\n]\n")

    findings = [f for f in report.findings if f.check == "display-delimiter"]
    assert [f.line for f in findings] == [3, 5]
    assert all(f.severity is Severity.AUTO_FIX for f in findings)


def test_correct_display_delimiters_are_not_flagged() -> None:
    report = lint_latex(CLEAN_LATEX + "\n\\[\na = b\n\\]\n\nA bracket [like this] in prose.\n")

    assert "display-delimiter" not in _checks(report.findings), report.render()


def test_the_display_delimiter_fix_restores_both_backslashes() -> None:
    fixed, count = apply_auto_fixes("Then\n\n[\na = b\n]\n\nand\n\n[\nc = d\n]\n")

    assert count == 4
    assert fixed == "Then\n\n\\[\na = b\n\\]\n\nand\n\n\\[\nc = d\n\\]\n"


def test_unpaired_bare_brackets_are_reported_instead_of_repaired() -> None:
    """Which block an odd bracket belongs to is a guess, so it is not one."""
    text = "[\na = b\n]\n\n]\n"
    report = lint_latex(text)

    assert "display-delimiter" in _checks(report.findings, Severity.REVIEW)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_tag_in_unnumbered_display_math_is_flagged_and_promoted() -> None:
    """amsmath allows \\tag only in an equation-like environment. The book
    tags 570 of its equations, so this arrives whenever the model wraps one
    of them in \\[ ... \\] instead."""
    text = "\\[\nE = mc^2 \\tag{12.1}\n\\]\n"
    report = lint_latex(text)

    findings = [f for f in report.findings if f.check == "misplaced-tag"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.AUTO_FIX

    fixed, count = apply_auto_fixes(text)
    assert count == 1
    assert fixed == "\\begin{equation}\nE = mc^2 \\tag{12.1}\n\\end{equation}\n"


def test_the_same_applies_to_dollar_display_math() -> None:
    fixed, count = apply_auto_fixes("$$\nx = y \\tag{3.2}\n$$\n")

    assert count == 1
    assert fixed == "\\begin{equation}\nx = y \\tag{3.2}\n\\end{equation}\n"


def test_untagged_display_math_is_left_in_place() -> None:
    """Most display maths is unnumbered and belongs exactly as it is."""
    text = "\\[\nE = mc^2\n\\]\n\n$$\nx = y\n$$\n"
    report = lint_latex(text)

    assert "misplaced-tag" not in _checks(report.findings)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_block_with_two_tags_is_reported_not_promoted() -> None:
    """Promoting it would not fix it -- amsmath rejects the second tag inside
    `equation` too -- and choosing between them means looking at the book."""
    text = "$$\nx = 1 \\tag {30}\\tag{30.12}\n$$\n"
    report = lint_latex(text)

    assert "multiple-tags" in _checks(report.findings, Severity.REVIEW)
    assert "misplaced-tag" not in _checks(report.findings)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_tag_already_inside_an_equation_is_left_alone() -> None:
    text = "\\begin{equation}\nE = mc^2 \\tag{12.1}\n\\end{equation}\n"

    assert "misplaced-tag" not in _checks(lint_latex(text).findings)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_doubled_structural_command_is_caught_and_repaired() -> None:
    """Regression: `\\sectionsection` stopped a real build five times. It is
    the same defect as `\\mathrmmathrm`, but `section` is not in
    KNOWN_COMMANDS -- that list is a math vocabulary -- so nothing reported
    it and nothing repaired it."""
    text = "\\sectionsection{Selected Answers}\n\\sectionsection*{Problem-Solving Points}\n"
    report = lint_latex(text)

    assert "doubled-command" in _checks(report.findings, Severity.AUTO_FIX)

    fixed, count = apply_auto_fixes(text)
    assert count == 2
    assert fixed == "\\section{Selected Answers}\n\\section*{Problem-Solving Points}\n"


def test_a_doubled_command_is_not_also_reported_as_unknown() -> None:
    """It is unknown by definition, but `--fix` repairs it, and the
    unknown-command advice -- "add it to KNOWN_COMMANDS" -- is exactly wrong
    for it."""
    report = lint_latex("The wavelength is $400 \\mathrmmathrm{~nm}$ here.\n")

    assert "doubled-command" in _checks(report.findings, Severity.AUTO_FIX)
    assert "unknown-command" not in _checks(report.findings, Severity.REVIEW)


def test_a_command_written_twice_with_both_backslashes_is_caught_and_repaired() -> None:
    """Regression: `\\section\\section{...}` stopped a real build with "TeX
    capacity exceeded" -- the `\\sectionsection` check could not see it."""
    text = "\\section\\section{Effect of Magnetic Media}\nA pellet of $1 \\mathrm\\mathrm{mm}$.\n"
    report = lint_latex(text)

    doubled = [f for f in report.findings if f.check == "doubled-command"]
    assert [f.line for f in doubled] == [1, 2]
    assert all(f.severity is Severity.AUTO_FIX for f in doubled)

    fixed, count = apply_auto_fixes(text)
    assert count == 2
    assert fixed == "\\section{Effect of Magnetic Media}\nA pellet of $1 \\mathrm{mm}$.\n"


def test_legitimately_repeated_commands_are_left_alone() -> None:
    text = "$f^{\\prime\\prime}$, $|\\Psi|^2 = \\Psi\\Psi^{*}$, $\\bar\\bar{x}\\quad\\quad y$\n"

    assert "doubled-command" not in _checks(lint_latex(text).findings)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_markdown_heading_left_in_the_latex_is_flagged() -> None:
    """Regression: three `### 2. ...` summary items reached a real build, and
    `#` at the start of a line is TeX's macro-parameter character."""
    report = lint_latex(
        "Some text.\n\n### 2. Electron Spin and Spin-Orbit Coupling\n\nMore text.\n"
    )

    headings = [f for f in report.findings if f.check == "markdown-heading"]
    assert len(headings) == 1
    assert headings[0].line == 3
    assert headings[0].severity is Severity.REVIEW


def test_an_escaped_or_commented_hash_is_not_a_markdown_heading() -> None:
    report = lint_latex("Item \\# 3 in the list.\n% ## a note to self\n\\#1 fan\n")

    assert "markdown-heading" not in _checks(report.findings)


def test_a_starred_heading_with_a_number_is_reported_not_stripped() -> None:
    """LaTeX does not number `\\subsection*`, so the number is not duplicated;
    on a real book these were summary list items parsed as headings."""
    text = "\\subsection*{4. Magnetic Field Intensity Vector}\n"
    report = lint_latex(text)

    assert "duplicated-number" not in _checks(report.findings)
    assert "numbered-starred-heading" in _checks(report.findings, Severity.REVIEW)
    assert apply_auto_fixes(text) == (text, 0)


def test_a_command_whose_package_is_not_loaded_is_flagged() -> None:
    """Regression: `\\multirow` in a translated table stopped a real build;
    the preamble never loaded the package."""
    text = "\\begin{tabular}{ll}\n\\multirow{2}{*}{Diamagnetic} & Bismuth \\\\\n\\end{tabular}\n"
    # The commented-out line must not count as loading it.
    preamble = "\\usepackage{amsmath, booktabs}\n% \\usepackage{multirow}\n"

    missing = [
        f for f in lint_latex(text, preamble=preamble).findings if f.check == "missing-package"
    ]
    assert len(missing) == 1
    assert missing[0].line == 2
    assert "\\usepackage{multirow}" in missing[0].message

    loaded = "\\usepackage[table]{array,multirow}\n"
    assert "missing-package" not in _checks(lint_latex(text, preamble=loaded).findings)
    # A lone fragment has no preamble to check against.
    assert "missing-package" not in _checks(lint_latex(text).findings)


def test_a_command_the_preamble_defines_itself_needs_no_package() -> None:
    preamble = "\\newcommand{\\degree}{\\ensuremath{^\\circ}}\n"

    report = lint_latex("It is $30\\degree$ warm.\n", preamble=preamble)
    assert "missing-package" not in _checks(report.findings)


def test_a_text_mode_command_inside_math_is_flagged() -> None:
    """Regression: `^{\\textcircled{1}}` footnote markers inside equations."""
    report = lint_latex("\\[\nE = 10 \\mathrm{nm} ^ {\\textcircled {1}}\n\\]\n")

    assert "text-command-in-math" in _checks(report.findings, Severity.REVIEW)


def test_a_text_mode_command_wrapped_in_text_or_in_prose_is_fine() -> None:
    report = lint_latex(
        "Regions \\textcircled{1} and \\textcircled{2}.\n\n$x^{\\text{\\textcircled{1}}}$\n"
    )

    assert "text-command-in-math" not in _checks(report.findings)


BOOK_PREAMBLE = "\\mainmatter\n\\setcounter{chapter}{11}\n"


def test_a_supplementary_reading_numbered_as_a_chapter_is_caught() -> None:
    """Regression: five readings and a part heading came through as numbered
    chapters, so a real book printed "Chapter 24" over chapter 22."""
    text = (
        "\\chapter{Electrostatic Field}\n\\begin{equation}F\\tag{12.1}\\end{equation}\n"
        "\\chapter{Atmospheric Electricity}\nNo equations here.\n"
        "\\chapter{Dielectrics}\n\\begin{equation}P\\tag{13.1}\\end{equation}\n"
        "\\begin{equation}D\\tag{13.2}\\end{equation}\n"
    )

    drift = [
        f for f in lint_latex(text, preamble=BOOK_PREAMBLE).findings
        if f.check == "chapter-numbering"
    ]
    assert len(drift) == 1
    assert drift[0].line == 5
    assert "chapter 14" in drift[0].message and "13.x" in drift[0].message
    assert "Atmospheric Electricity" in drift[0].message

    fixed = text.replace("\\chapter{Atmospheric", "\\chapter*{Atmospheric")
    report = lint_latex(fixed, preamble=BOOK_PREAMBLE)
    assert "chapter-numbering" not in _checks(report.findings)


def test_chapter_numbering_is_not_checked_without_a_preamble() -> None:
    """A lone fragment does not start at the book's first chapter."""
    report = lint_latex("\\chapter{Dielectrics}\n\\begin{equation}P\\tag{15.1}\\end{equation}\n")

    assert "chapter-numbering" not in _checks(report.findings)


def test_a_genuinely_unknown_command_is_still_reported() -> None:
    report = lint_latex("$\\frobnicate{x}$\n")

    assert "unknown-command" in _checks(report.findings, Severity.REVIEW)


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


def test_expansion_outlier_is_not_fooled_by_a_math_heavy_chunk(tmp_path: Path) -> None:
    """Regression: a real chunk that was mostly formulas and tables expanded
    0.93x against a book median of 2.66x -- math does not grow when prose is
    translated -- and was reported as cut off though it was complete."""
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir(parents=True)

    formulas = "E = \\frac{q}{4 \\pi \\varepsilon_0 r^2} + \\sum_i E_i \\\\\n" * 200
    for index in range(1, 6):
        source, translation = UNTRANSLATED_CHUNK, TRANSLATED_CHUNK
        if index == 3:
            source += f"\n$$\n{formulas}$$\n"
            translation += f"\n\\[\n{formulas}\\]\n"
        (kit / "chunks" / f"{index:03d}.md").write_text(source, encoding="utf-8")
        (kit / "translated" / f"{index:03d}.tex").write_text(translation, encoding="utf-8")

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
