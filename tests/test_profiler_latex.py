"""Tests for src.profiler (book profiling) and src.latex (LaTeX output).

No real API calls: the LLM is always a fake that returns scripted JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.latex import (
    DENYLISTED_PACKAGES,
    assemble_document,
    build_preamble,
    compile_pdf,
    escape_latex,
    validate_fragment,
)
from src.llm.base import BaseLLM
from src.profiler import (
    PROFILE_FILENAME,
    BookProfile,
    profile_book,
    profile_to_prompt_block,
)


class FakeLLM(BaseLLM):
    """Returns ``response`` (or raises it, if it is an exception) and counts calls."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        if isinstance(self.response, Exception):
            raise self.response
        return str(self.response)

    @property
    def call_count(self) -> int:
        return len(self.calls)


VALID_PROFILE_JSON = json.dumps(
    {
        "subject": "mathematics",
        "subfield": "real analysis",
        "education_level": "undergraduate",
        "audience": "first-year mathematics students",
        "register": "formal academic prose",
        "notation_notes": "Uses $\\varepsilon$-$\\delta$ definitions throughout.",
        "structural_elements": ["definition", "theorem", "proof", "exercise"],
        "glossary": [
            {"source": "导数", "target": "derivative"},
            {"source": "极限", "target": "limit"},
        ],
        "latex_documentclass": "book",
        "latex_packages": ["amsthm"],
        "summary": "An introduction to real analysis.",
    },
    ensure_ascii=False,
)


def _write_book(tmp_path: Path, text: str = "# 第一章\n\n本章介绍导数的定义。\n") -> Path:
    merged = tmp_path / "merged.md"
    merged.write_text(text, encoding="utf-8")
    return merged


# ── BookProfile.from_dict ───────────────────────────────────────────────────


def test_from_dict_parses_object_style_glossary() -> None:
    profile = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    assert profile.subject == "mathematics"
    assert profile.glossary == (("导数", "derivative"), ("极限", "limit"))
    assert profile.structural_elements == ("definition", "theorem", "proof", "exercise")
    assert profile.latex_packages == ("amsthm",)


def test_from_dict_parses_list_style_glossary() -> None:
    profile = BookProfile.from_dict({"glossary": [["导数", "derivative"], ["极限", "limit"]]})

    assert profile.glossary == (("导数", "derivative"), ("极限", "limit"))


def test_from_dict_skips_junk_glossary_entries() -> None:
    profile = BookProfile.from_dict(
        {
            "glossary": [
                {"source": "导数", "target": "derivative"},
                {"source": "极限"},  # missing target
                ["only-one-element"],
                ["a", "b", "c"],
                "not an entry at all",
                {"source": "", "target": "empty source"},
                42,
                None,
                ["积分", "integral"],
            ]
        }
    )

    assert profile.glossary == (("导数", "derivative"), ("积分", "integral"))


def test_from_dict_tolerates_missing_keys_and_wrong_types() -> None:
    profile = BookProfile.from_dict({"subject": "physics", "glossary": "not a list"})

    assert profile.subject == "physics"
    assert profile.glossary == ()
    assert profile.education_level == BookProfile.generic().education_level
    assert profile.latex_documentclass == "book"


def test_from_dict_coerces_a_bare_string_to_a_one_tuple() -> None:
    profile = BookProfile.from_dict(
        {"latex_packages": "amsthm", "structural_elements": "theorem"}
    )

    assert profile.latex_packages == ("amsthm",)
    assert profile.structural_elements == ("theorem",)


def test_from_dict_of_non_dict_is_generic() -> None:
    assert BookProfile.from_dict(["nope"]) == BookProfile.generic()  # type: ignore[arg-type]


def test_to_json_round_trips_through_from_dict() -> None:
    original = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    assert BookProfile.from_dict(json.loads(original.to_json())) == original


# ── profile_book ────────────────────────────────────────────────────────────


def test_profile_book_parses_and_caches(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    llm = FakeLLM(VALID_PROFILE_JSON)

    profile = profile_book(merged, tmp_path, llm=llm)

    assert profile.subject == "mathematics"
    assert llm.call_count == 1
    assert (tmp_path / PROFILE_FILENAME).exists()


def test_second_call_uses_the_cache_without_hitting_the_llm(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    second_llm = FakeLLM(VALID_PROFILE_JSON)
    cached = profile_book(merged, tmp_path, llm=second_llm)

    assert second_llm.call_count == 0
    assert cached.glossary == (("导数", "derivative"), ("极限", "limit"))


def test_force_re_profiles_despite_the_cache(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    second_llm = FakeLLM(VALID_PROFILE_JSON)
    profile_book(merged, tmp_path, llm=second_llm, force=True)

    assert second_llm.call_count == 1


def test_fenced_and_prefixed_json_is_still_parsed(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    llm = FakeLLM(f"Here is the profile:\n```json\n{VALID_PROFILE_JSON}\n```\nHope that helps!")

    profile = profile_book(merged, tmp_path, llm=llm)

    assert profile.subject == "mathematics"


def test_unparseable_json_falls_back_to_generic_and_is_not_cached(
    tmp_path: Path, caplog
) -> None:
    merged = _write_book(tmp_path)

    with caplog.at_level("WARNING"):
        profile = profile_book(merged, tmp_path, llm=FakeLLM("I'm afraid I can't do that."))

    assert profile == BookProfile.generic()
    assert not (tmp_path / PROFILE_FILENAME).exists()
    assert any("profiling failed" in record.message.lower() for record in caplog.records)


def test_raising_llm_falls_back_to_generic_without_raising(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)

    profile = profile_book(merged, tmp_path, llm=FakeLLM(RuntimeError("API down")))

    assert profile == BookProfile.generic()
    assert not (tmp_path / PROFILE_FILENAME).exists()


def test_missing_source_file_falls_back_to_generic(tmp_path: Path) -> None:
    profile = profile_book(tmp_path / "nope.md", tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    assert profile == BookProfile.generic()


def test_a_failed_profile_does_not_poison_the_next_run(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(RuntimeError("API down")))

    recovered = profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    assert recovered.subject == "mathematics"


def test_large_books_are_sampled_not_sent_whole(tmp_path: Path) -> None:
    body = "\n\n".join(f"## 第{i}节\n\n本节内容重复出现。" * 40 for i in range(200))
    merged = _write_book(tmp_path, body)
    llm = FakeLLM(VALID_PROFILE_JSON)

    profile_book(merged, tmp_path, llm=llm)

    sample = llm.calls[0][1]
    assert len(sample) < len(body)
    assert len(sample) <= 30_000


# ── profile_to_prompt_block ─────────────────────────────────────────────────


def test_prompt_block_lists_required_terminology() -> None:
    profile = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    block = profile_to_prompt_block(profile)

    assert "Required terminology" in block
    assert "导数 -> derivative" in block
    assert "极限 -> limit" in block
    assert "undergraduate" in block


def test_prompt_block_caps_the_glossary_and_stays_compact() -> None:
    profile = BookProfile(glossary=tuple((f"术语{i}", f"term{i}") for i in range(100)))

    block = profile_to_prompt_block(profile, max_glossary=5)

    assert "术语4 -> term4" in block
    assert "术语5 -> term5" not in block
    assert len(block.splitlines()) < 15


def test_prompt_block_of_a_generic_profile_has_no_terminology_section() -> None:
    block = profile_to_prompt_block(BookProfile.generic())

    assert "Required terminology" not in block
    assert "BOOK CONTEXT" in block


# ── escape_latex ────────────────────────────────────────────────────────────


def test_escape_latex_escapes_special_characters() -> None:
    assert escape_latex("50% of A&B costs $5_x #1") == (
        r"50\% of A\&B costs \$5\_x \#1"
    )


# ── validate_fragment ───────────────────────────────────────────────────────


def test_sound_fragment_has_no_issues() -> None:
    tex = (
        "\\chapter{Limits}\n"
        "The limit of $f(x)$ as $x \\to 0$ is 50\\% of the value.\n"
        "\\begin{itemize}\n\\item First\n\\item Second\n\\end{itemize}\n"
        "\\begin{equation}\n\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1\n\\end{equation}\n"
        "A price of \\$5 and a literal brace \\{ are fine.\n"
        "% a comment with an unmatched { and $ in it\n"
    )

    assert validate_fragment(tex) == []


@pytest.mark.parametrize(
    "leaked",
    [
        "\\documentclass{book}\n\\chapter{One}",
        "\\usepackage{amsmath}\n\\chapter{One}",
        "\\begin{document}\n\\chapter{One}\n\\end{document}",
    ],
)
def test_preamble_commands_are_flagged(leaked: str) -> None:
    issues = validate_fragment(leaked)

    assert any("body fragment" in issue for issue in issues)


def test_unclosed_environment_is_flagged() -> None:
    issues = validate_fragment("\\begin{equation}\nx = y\n")

    assert issues == ["\\begin{equation} is never closed"]


def test_crossed_environments_are_flagged() -> None:
    issues = validate_fragment("\\begin{itemize}\n\\item one\n\\end{enumerate}\n")

    assert any(
        "\\begin{itemize} is closed by \\end{enumerate}" == issue for issue in issues
    )


def test_stray_end_without_begin_is_flagged() -> None:
    issues = validate_fragment("Some prose.\n\\end{itemize}\n")

    assert any("has no matching \\begin" in issue for issue in issues)


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
    issues = validate_fragment("$$x = y\n")

    assert any("display math" in issue for issue in issues)


# ── build_preamble ──────────────────────────────────────────────────────────


def test_preamble_includes_baseline_packages_and_cjk_fallback() -> None:
    preamble = build_preamble(BookProfile.generic(), "A Textbook")

    for package in ("amsmath", "amssymb", "graphicx", "booktabs", "longtable", "array"):
        assert f"\\usepackage{{{package}}}" in preamble
    assert "\\usepackage{fontspec}" in preamble
    assert "\\IfFontExistsTF{Noto Sans CJK SC}" in preamble
    assert "\\usepackage{hyperref}" in preamble


def test_preamble_merges_profile_packages() -> None:
    preamble = build_preamble(
        BookProfile(latex_packages=("amsthm", "siunitx")), "A Textbook"
    )

    assert "\\usepackage{amsthm}" in preamble
    assert "\\usepackage{siunitx}" in preamble


def test_preamble_drops_denylisted_and_malformed_package_names() -> None:
    profile = BookProfile(
        latex_packages=(
            "ctex",
            "fontspec",
            "geometry",
            "babel",
            "inputenc",
            "fontenc",
            "xeCJK",
            "amsmath}\\input{/etc/passwd",
            "9lives",
            "with space",
            "",
            "amsthm",
        )
    )

    preamble = build_preamble(profile, "A Textbook")

    # fontspec and geometry are loaded by the fixed preamble itself (which is
    # exactly why a profile may not re-declare them); the rest must be absent.
    for package in DENYLISTED_PACKAGES - {"fontspec", "geometry"}:
        assert f"\\usepackage{{{package}}}" not in preamble
    assert preamble.count("\\usepackage{fontspec}") == 1
    assert preamble.count("geometry") == 1
    assert "\\usepackage[margin=1in]{geometry}" in preamble

    assert "\\input{/etc/passwd" not in preamble
    assert "\\usepackage{9lives}" not in preamble
    assert "\\usepackage{with space}" not in preamble
    assert "\\usepackage{amsthm}" in preamble


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("book", "book"),
        ("report", "report"),
        ("article", "article"),
        ("REPORT", "report"),
        ("memoir", "book"),
        ("", "book"),
        ("book]{\\input|", "book"),
    ],
)
def test_documentclass_is_clamped(requested: str, expected: str) -> None:
    preamble = build_preamble(BookProfile(latex_documentclass=requested), "T")

    assert f"\\documentclass[11pt]{{{expected}}}" in preamble


def test_title_is_escaped_in_the_preamble() -> None:
    preamble = build_preamble(BookProfile.generic(), "Statistics & Probability 100%")

    assert "\\title{Statistics \\& Probability 100\\%}" in preamble


# ── assemble_document ───────────────────────────────────────────────────────


def test_assemble_document_wraps_bodies_exactly_once(tmp_path: Path) -> None:
    output = assemble_document(
        ["\\chapter{One}\nBody one.", "\\chapter{Two}\nBody two."],
        BookProfile.generic(),
        "A Textbook",
        tmp_path / "out" / "translated_book.tex",
    )

    tex = output.read_text(encoding="utf-8")
    assert tex.count("\\begin{document}") == 1
    assert tex.count("\\end{document}") == 1
    assert tex.index("\\begin{document}") < tex.index("\\chapter{One}")
    assert tex.index("\\chapter{One}") < tex.index("\\chapter{Two}")
    assert tex.index("\\chapter{Two}") < tex.index("\\end{document}")
    assert tex.index("\\documentclass") < tex.index("\\begin{document}")


def test_assemble_document_strips_preamble_leaked_into_a_body(tmp_path: Path) -> None:
    output = assemble_document(
        [
            "\\documentclass{book}\n\\usepackage{amsmath}\n"
            "\\begin{document}\n\\chapter{One}\nBody one.\n\\end{document}"
        ],
        BookProfile.generic(),
        "A Textbook",
        tmp_path / "translated_book.tex",
    )

    tex = output.read_text(encoding="utf-8")
    assert tex.count("\\begin{document}") == 1
    assert tex.count("\\end{document}") == 1
    assert tex.count("\\documentclass") == 1
    assert "\\chapter{One}" in tex


def test_assemble_document_creates_missing_parent_directories(tmp_path: Path) -> None:
    output = assemble_document(
        ["Body."], BookProfile.generic(), "T", tmp_path / "a" / "b" / "book.tex"
    )

    assert output.exists()


# ── compile_pdf ─────────────────────────────────────────────────────────────


def test_compile_pdf_reports_a_missing_engine_instead_of_raising(tmp_path: Path) -> None:
    tex_path = tmp_path / "book.tex"
    tex_path.write_text("\\documentclass{book}\\begin{document}x\\end{document}", encoding="utf-8")

    succeeded, message = compile_pdf(tex_path, engine="definitely-not-a-real-engine")

    assert succeeded is False
    assert "not on PATH" in message


def test_compile_pdf_reports_a_missing_source_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.latex.shutil.which", lambda _engine: "/usr/bin/xelatex")

    succeeded, message = compile_pdf(tmp_path / "missing.tex")

    assert succeeded is False
    assert "not found" in message
