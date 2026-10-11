"""Tests for src.fidelity: does the translation say the same mathematics?

Positive fixtures are the drifts the check exists for -- each must fire.
Negative fixtures are formatting-only differences that a correct translation
makes all the time -- each must stay quiet. The source side is MinerU
Markdown, the translation side LaTeX, as in a real kit.
"""

from __future__ import annotations

import pytest

from src.fidelity import check_pair, numbers, split_tags, tokens
from src.findings import Severity

PROSE_SRC = "由库仑定律可知，两个点电荷之间的作用力为\n\n"
PROSE_TGT = "By Coulomb's law, the force between two point charges is\n\n"


def _review(source: str, translation: str) -> list[str]:
    return sorted(
        {f.check for f in check_pair(source, translation) if f.severity is Severity.REVIEW}
    )


# ── Positives: each must fire ───────────────────────────────────────────────


def test_a_wrong_exponent_fires() -> None:
    source = PROSE_SRC + "$$\nq = 1.6 \\times 10^{-20} \\mathrm{C}\n$$\n"
    translation = PROSE_TGT + "\\[\nq = 1.6 \\times 10^{-2} \\mathrm{C}\n\\]\n"
    findings = [f for f in check_pair(source, translation) if f.check == "fidelity-numbers"]

    messages = " ".join(f.message for f in findings)
    assert "^-20 is missing" in messages
    assert "^-2 is invented" in messages
    assert all(f.severity is Severity.REVIEW for f in findings)


def test_a_flipped_sign_fires() -> None:
    source = PROSE_SRC + "其中 $W = -3 \\mathrm{J}$ ，而\n\n$$\nE = - \\nabla \\varphi\n$$\n"
    translation = PROSE_TGT + "where $W = 3~\\mathrm{J}$, and\n\n\\[\nE = \\nabla \\varphi\n\\]\n"
    review = _review(source, translation)

    assert "fidelity-numbers" in review  # -3 became 3
    assert "fidelity-blocks" in review  # the display lost its minus


def test_a_dropped_display_block_fires() -> None:
    source = PROSE_SRC + "$$\nF = k \\frac{q_1 q_2}{r^2}\n$$\n\n于是\n\n$$\nE = \\frac{F}{q_0}\n$$\n"
    translation = PROSE_TGT + "\\[\nF = k \\frac{q_1 q_2}{r^2}\n\\]\n\nHence\n"
    assert "fidelity-display-count" in _review(source, translation)


def test_a_dropped_tag_fires() -> None:
    source = PROSE_SRC + "$$\nF = k \\frac{q_1 q_2}{r^2} \\tag{12.3}\n$$\n"
    translation = PROSE_TGT + "\\[\nF = k \\frac{q_1 q_2}{r^2}\n\\]\n"
    findings = [f for f in check_pair(source, translation) if f.check == "fidelity-tags"]

    assert findings and findings[0].severity is Severity.REVIEW
    assert "missing 12.3" in findings[0].message


def test_a_swapped_subscript_fires() -> None:
    source = PROSE_SRC + "$$\nF_{12} = q_1 E_2\n$$\n"
    translation = PROSE_TGT + "\\[\nF_{12} = q_2 E_1\n\\]\n"
    assert "fidelity-blocks" in _review(source, translation)


# ── Negatives: each must stay quiet ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("source", "translation"),
    [
        pytest.param(
            "$$\nF = k \\frac{q_1 q_2}{r^2} \\tag{12.1}\n$$\n",
            "\\begin{equation}\nF = k \\frac{q_1 q_2}{r^2} \\tag{12.1}\n\\end{equation}\n",
            id="$$-to-equation",
        ),
        pytest.param(
            "$$\nE = \\frac{F}{q_0}\n$$\n",
            "\\[\nE = \\frac{F}{q_0}\n\\]\n",
            id="$$-to-brackets",
        ),
        pytest.param(
            "$$\nU = \\int E dl\n$$\n",
            "\\[\nU = \\int E \\, \\mathrm{d}l\n\\]\n",
            id="mathrm-and-thin-space-added",
        ),
        pytest.param(
            "$$\nn = 1 0 ^ { - 2 0 } \\mathrm { m } ^ { - 3 }\n$$\n",
            "\\[\nn = 10^{-20}\\,\\mathrm{m}^{-3}\n\\]\n",
            id="character-spacing-removed",
        ),
        pytest.param(
            "$$\nv _ {\\text {平均}} = \\frac {\\Delta x}{\\Delta t}\n$$\n",
            "\\[\nv_{\\text{avg}} = \\frac{\\Delta x}{\\Delta t}\n\\]\n",
            id="text-translated",
        ),
        pytest.param(
            "$$\nE = m c ^ {2} \\qquad (12.3)\n$$\n",
            "\\begin{equation}\nE = mc^2 \\tag{12.3}\n\\end{equation}\n",
            id="tag-in-place-of-inline-number",
        ),
        pytest.param(
            "$$\nE = m c ^ {2}\n$$ (12.3)\n",
            "\\begin{equation}\nE = mc^2 \\tag{12.3}\n\\end{equation}\n",
            id="tag-in-place-of-number-after-display",
        ),
        pytest.param(
            "$$\nx \\leqslant 3 \\times 10 ^ {5}\n$$\n",
            "\\[\nx \\le 3 \\times 10^5.\n\\]\n",
            id="aliases-and-closing-period",
        ),
        pytest.param(
            "设 $F = q E$ ，则\n",
            "Let\n\\[\nF = qE,\n\\]\nthen\n",
            id="inline-promoted-to-display",
        ),
        pytest.param(
            "电流 I=1.5 A ，则 $P = 1.5 \\times 2 = 3$ W\n",
            "The current is $I = 1.5$~A, so $P = 1.5 \\times 2 = 3$~W\n",
            id="number-moved-from-prose-into-math",
        ),
        pytest.param(
            "$$\n\\varepsilon_0 = 8.85 \\times 10^{-12} ^{①}\n$$\n",
            "\\[\n\\varepsilon_0 = 8.85 \\times 10^{-12}\n\\]\n",
            id="footnote-marker-deleted",
        ),
    ],
)
def test_formatting_only_differences_stay_quiet(source: str, translation: str) -> None:
    assert _review(PROSE_SRC + source, PROSE_TGT + translation) == []


# ── The pieces ──────────────────────────────────────────────────────────────


def test_parser_spacing_and_exponents_normalise_to_one_stream() -> None:
    assert tokens("1 0 ^ { - 2 0 }") == tokens("10^{-20}") == ["10", "^", "{", "-", "20", "}"]
    assert numbers(tokens("10^{-20}")) == ["10", "^-20"]
    assert numbers(tokens("u_{02}")) == ["_02"]


def test_text_contents_are_prose_and_dropped() -> None:
    assert tokens("F_{\\text{合}}") == tokens("F_{\\text{net}}") == tokens("F_{\\mathrm{net}}")


def test_inline_parentheses_are_references_not_tags() -> None:
    assert split_tags("(12.3)", display=False) == ("(12.3)", [])
    assert split_tags("x = 1 \\quad (12.3)")[1] == ["12.3"]


def test_findings_are_located_in_the_translation() -> None:
    source = PROSE_SRC + "$$\nF_{12} = q_1 E_2\n$$\n"
    translation = PROSE_TGT + "\\[\nF_{12} = q_2 E_1\n\\]\n"
    (finding,) = [f for f in check_pair(source, translation, file="translated/007.tex")
                  if f.check == "fidelity-blocks"]
    assert finding.file == "translated/007.tex"
    assert finding.line == 3
    assert "q_1 E_2" in finding.context and "q_2 E_1" in finding.context


def test_missing_prose_numbers_are_information_only() -> None:
    findings = check_pair("共有 3 种电荷，见第 12 页。\n", "There are three kinds of charge.\n")
    (finding,) = [f for f in findings if f.check == "fidelity-prose-numbers"]
    assert finding.severity is Severity.INFO
    assert "3" in finding.context and "12" in finding.context
