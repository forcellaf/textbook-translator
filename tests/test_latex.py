"""Tests for src.latex (assets, fragment scanning, document assembly).

The two assets are treated as the specification here: these tests assert that
the code loads them and substitutes into them, never that they contain any
particular formatting decision. Editing assets/preamble.tex to change how
captions look must not break this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.latex import (
    GLOSSARY_PLACEHOLDER,
    PREAMBLE_PATH,
    SYSTEM_PROMPT_PATH,
    assemble_document,
    build_system_prompt,
    load_preamble,
    scan_environments,
    validate_fragment,
)
from src.profiler import BookProfile

PHYSICS = BookProfile(
    subject="physics",
    subfield="electromagnetism",
    glossary=(("电荷", "electric charge"), ("电场强度", "electric field strength")),
)


# ── The system prompt asset ─────────────────────────────────────────────────


def test_the_system_prompt_comes_from_the_asset_file() -> None:
    """It lives in a file because the user edits it between books."""
    prompt = build_system_prompt(PHYSICS)
    asset = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    assert "## Figures" in asset  # the asset really is the source of this text
    assert "## Figures" in prompt


def test_the_glossary_placeholder_is_substituted() -> None:
    prompt = build_system_prompt(PHYSICS)

    assert GLOSSARY_PLACEHOLDER not in prompt
    assert "电荷 -> electric charge" in prompt
    assert "电场强度 -> electric field strength" in prompt


def test_a_missing_profile_leaves_the_placeholder_empty_not_literal() -> None:
    """Running without a profile costs terminology consistency, not a build:
    the placeholder must not reach the model as `{{...}}`."""
    prompt = build_system_prompt(None)

    assert GLOSSARY_PLACEHOLDER not in prompt
    assert "BOOK CONTEXT" not in prompt


def test_a_template_without_the_placeholder_is_used_as_is(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    template = tmp_path / "prompt.txt"
    template.write_text("Translate it.", encoding="utf-8")

    with caplog.at_level("WARNING"):
        prompt = build_system_prompt(PHYSICS, path=template)

    assert prompt == "Translate it."
    assert any("placeholder" in record.message for record in caplog.records)


def test_a_missing_asset_is_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tracked asset"):
        build_system_prompt(PHYSICS, path=tmp_path / "gone.txt")
    with pytest.raises(FileNotFoundError, match="tracked asset"):
        load_preamble(tmp_path / "gone.tex")


# ── The preamble asset ──────────────────────────────────────────────────────


def test_the_preamble_opens_the_document_and_does_not_close_it() -> None:
    """assemble_document adds \\end{document} and nothing else, so the asset
    has to carry everything up to and including \\begin{document}."""
    preamble = load_preamble()

    assert "\\begin{document}" in preamble
    assert "\\end{document}" not in preamble
    assert preamble == PREAMBLE_PATH.read_text(encoding="utf-8")


# ── assemble_document ───────────────────────────────────────────────────────


def test_fragments_are_wrapped_in_the_preamble_exactly_once(tmp_path: Path) -> None:
    output = assemble_document(
        ["\\chapter{One}\nBody one.", "\\chapter{Two}\nBody two."],
        tmp_path / "out" / "translated_book.tex",
    )

    tex = output.read_text(encoding="utf-8")

    assert tex.count("\\documentclass") == 1
    assert tex.count("\\begin{document}") == 1
    assert tex.count("\\end{document}") == 1
    assert tex.index("\\begin{document}") < tex.index("\\chapter{One}")
    assert tex.index("\\chapter{One}") < tex.index("\\chapter{Two}")
    assert tex.index("\\chapter{Two}") < tex.index("\\end{document}")


def test_the_chapter_counter_survives_assembly(tmp_path: Path) -> None:
    """This volume starts at chapter 12; without the counter every section
    cross-reference in the prose disagrees with its heading."""
    output = assemble_document(["\\chapter{Electric Charge}"], tmp_path / "book.tex")

    assert "\\setcounter{chapter}{11}" in output.read_text(encoding="utf-8")


def test_preamble_leaked_into_a_body_is_dropped(tmp_path: Path) -> None:
    """A second \\end{document} halfway through would end the book there."""
    output = assemble_document(
        [
            "\\documentclass{book}\n\\usepackage{amsmath}\n"
            "\\begin{document}\n\\chapter{One}\nBody one.\n\\end{document}"
        ],
        tmp_path / "translated_book.tex",
    )

    tex = output.read_text(encoding="utf-8")

    assert tex.count("\\documentclass") == 1
    assert tex.count("\\begin{document}") == 1
    assert tex.count("\\end{document}") == 1
    assert "\\chapter{One}" in tex


def test_assembly_creates_missing_parent_directories(tmp_path: Path) -> None:
    output = assemble_document(["Body."], tmp_path / "a" / "b" / "book.tex")

    assert output.exists()


# ── scan_environments ───────────────────────────────────────────────────────


def test_a_sound_fragment_has_no_environment_issues() -> None:
    assert (
        scan_environments(
            "\\section{Charge}\n"
            "\\begin{example}\nA worked example.\n\\end{example}\n"
            "\\begin{itemize}\n\\item one\n\\end{itemize}\n"
        )
        == []
    )


def test_an_unclosed_environment_is_reported_with_its_line() -> None:
    issues = scan_environments("\\section{Charge}\n\nText.\n\\begin{equation}\nx = y\n")

    assert [(i.kind, i.env, i.line) for i in issues] == [("unclosed", "equation", 4)]


def test_a_crossed_pair_is_reported_as_one_defect() -> None:
    issues = scan_environments("\\begin{itemize}\n\\item one\n\\end{enumerate}\n")

    assert [i.kind for i in issues] == ["crossed"]
    assert "\\begin{itemize}" in issues[0].message
    assert "\\end{enumerate}" in issues[0].message


def test_an_end_with_no_begin_is_reported() -> None:
    issues = scan_environments("Some prose.\n\\end{itemize}\n")

    assert [i.kind for i in issues] == ["stray-end"]


def test_an_invented_environment_is_reported() -> None:
    """The model invented `solution` and `theorem` during testing; neither is
    defined, and either one stops the build."""
    issues = scan_environments("\\begin{solution}\nBecause.\n\\end{solution}\n")

    assert [(i.kind, i.env) for i in issues] == [("undefined", "solution")]


def test_starred_variants_of_defined_environments_are_accepted() -> None:
    assert scan_environments("\\begin{align*}\nx &= y\n\\end{align*}\n") == []


def test_environments_inside_comments_are_ignored() -> None:
    assert scan_environments("% \\begin{itemize} in a comment\nText.\n") == []


# ── validate_fragment (the translator's heal-loop gate) ─────────────────────


def test_validate_fragment_accepts_sound_latex() -> None:
    assert (
        validate_fragment(
            "\\section{Electric Charge}\n\n"
            "Charge is quantized: $q = ne$ for integer $n$.\n\n"
            "\\begin{equation}\nF = \\frac{q_1 q_2}{4\\pi\\varepsilon_0 r^2}\n\\end{equation}\n"
            "\\bookfig{IMG_0042}{Field of a point charge}\n"
        )
        == []
    )


@pytest.mark.parametrize(
    "leaked",
    [
        "\\documentclass{book}\nBody.",
        "\\usepackage{amsmath}\nBody.",
        "\\begin{document}\nBody.\n\\end{document}",
    ],
)
def test_preamble_commands_are_flagged(leaked: str) -> None:
    assert any("body fragment" in issue for issue in validate_fragment(leaked))


def test_unbalanced_braces_are_flagged() -> None:
    assert any("unclosed" in issue for issue in validate_fragment("\\section{Limits"))
    assert any("no matching" in issue for issue in validate_fragment("Limits}"))


def test_escaped_braces_do_not_count_as_unbalanced() -> None:
    assert validate_fragment("A literal \\{ and \\} in prose.") == []


def test_odd_inline_math_delimiter_is_flagged() -> None:
    issues = validate_fragment("The value $x is undefined.")

    assert any("odd number of unescaped '$'" in issue for issue in issues)


def test_escaped_dollars_and_display_math_do_not_count() -> None:
    assert validate_fragment("It costs \\$5 today.") == []
    assert validate_fragment("$$x = y$$ and $z$ inline.") == []


def test_unclosed_display_math_is_flagged() -> None:
    assert any("display math" in issue for issue in validate_fragment("$$x = y\n"))
