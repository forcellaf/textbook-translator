"""Tests for the cover page (src.cover) and wide-table fitting (src.latex).

The real failures: a translated table whose last column ran off the page
("Main applications" cut in half), and books built with no title page. If
pdflatex is installed, the fixes are also checked in a real compile.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pymupdf
import pytest

from src.cover import cover_latex, find_book_meta, load_book_meta
from src.kit import assemble
from src.latex import assemble_document, fit_wide_tables

META = {
    "title": "电路原理学习指导与习题集",
    "title_en": "Learning Guide and Exercises for Principle of Electric Circuits",
    "subtitle_en": None,
    "authors": ["朱桂萍", "刘秀成", "徐福媛"],
    "authors_en": ["Zhu Guiping", "Liu Xiucheng", "Xu Fuyuan"],
    "publisher": "清华大学出版社",
    "publisher_en": "Tsinghua University Press",
    "evidence": "p1 cover, p3 title page",
    "by": "agent",
}

WIDE_TABLE = (
    "\\begin{table}[htbp]\n\\centering\n\\caption{Resistivities of common conductor materials}\n"
    "\\begin{tabular}{llll}\n\\toprule\n"
    "Material & Resistivity $\\rho$ (20\\,$^\\circ$C) ($\\Omega\\cdot$m) & "
    "Temperature coefficient $\\alpha$ (20\\,$^\\circ$C) ($^\\circ$C$^{-1}$) & Main applications \\\\\n"
    "\\midrule\n"
    "Copper & $1.69 \\times 10^{-8}$ & $3.93 \\times 10^{-3}$ & "
    "Various wires and cables used in power distribution \\\\\n"
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)


def _meta_file(tmp_path: Path, **changes) -> Path:
    path = tmp_path / "book_meta.json"
    path.write_text(json.dumps({**META, **changes}, ensure_ascii=False), encoding="utf-8")
    return path


# ── Wide tables ─────────────────────────────────────────────────────────────


def test_each_outermost_tabular_is_wrapped_on_the_same_lines() -> None:
    nested = "\\begin{tabular}{ll}a & \\begin{tabular}{c}b\\end{tabular} \\\\\n\\end{tabular}"
    text = f"Before.\n\n{WIDE_TABLE}\n{nested}\n"
    fitted, count = fit_wide_tables(text)

    assert count == 2
    assert fitted.count("\\bookfittable{\\begin{tabular}") == 2
    assert fitted.count("\\begin{tabular}") == text.count("\\begin{tabular}")
    assert fitted.count("\n") == text.count("\n")  # error locations stay correct


@pytest.mark.parametrize(
    "text",
    [
        "\\begin{tabular*}{\\textwidth}{ll}a & b\\end{tabular*}",
        "\\begin{longtable}{ll}a & b\\end{longtable}",
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll}a & b\\end{tabular}}",
        "% \\begin{tabular}{ll} commented out",
        "No tables here.",
    ],
)
def test_tables_that_set_or_already_fit_their_width_are_left_alone(text: str) -> None:
    assert fit_wide_tables(text) == (text, 0)


def test_the_fit_macro_is_defined_only_when_a_table_uses_it(tmp_path: Path) -> None:
    with_table = assemble_document([WIDE_TABLE], tmp_path / "a.tex").read_text(encoding="utf-8")
    without = assemble_document(["Just prose."], tmp_path / "b.tex").read_text(encoding="utf-8")

    assert "\\newcommand{\\bookfittable}" in with_table
    assert with_table.index("\\begin{document}") < with_table.index("\\newcommand{\\bookfittable}")
    assert "bookfittable" not in without


# ── Cover ───────────────────────────────────────────────────────────────────


def test_the_cover_carries_title_authors_and_publisher(tmp_path: Path) -> None:
    meta, problems = load_book_meta(_meta_file(tmp_path))
    assert problems == [] and meta is not None
    cover = cover_latex(meta)

    assert "English Translation" in cover
    assert "Learning Guide and Exercises for Principle of Electric Circuits" in cover
    assert "Zhu Guiping, Liu Xiucheng, Xu Fuyuan" in cover
    assert "Tsinghua University Press" in cover
    assert "pdftitle=" in cover


def test_cover_text_is_escaped_for_latex(tmp_path: Path) -> None:
    meta, _ = load_book_meta(_meta_file(tmp_path, title_en="Circuits & Signals: 100% Solved_#1"))
    assert "Circuits \\& Signals: 100\\% Solved\\_\\#1" in cover_latex(meta)


def test_untranslated_fields_are_left_out_not_typeset(tmp_path: Path) -> None:
    meta, problems = load_book_meta(
        _meta_file(tmp_path, authors_en=["Zhu Guiping", "刘秀成"], publisher_en="清华大学出版社")
    )
    assert meta is not None
    assert meta.authors_en == ("Zhu Guiping",) and meta.publisher_en == ""
    assert len(problems) == 2


def test_no_usable_title_means_no_cover(tmp_path: Path) -> None:
    assert load_book_meta(None)[0] is None
    meta, problems = load_book_meta(_meta_file(tmp_path, title_en="电路原理"))
    assert meta is None and any("no cover page" in p for p in problems)


def test_assembly_puts_the_cover_before_the_table_of_contents(tmp_path: Path) -> None:
    work = tmp_path / "physics"
    kit = work / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir()
    (kit / "chunks" / "001.md").write_text("x", encoding="utf-8")
    (kit / "translated" / "001.tex").write_text("\\chapter{Charge}\nText.\n", encoding="utf-8")
    _meta_file(work)
    assert find_book_meta(kit) == work / "book_meta.json"

    tex = assemble(kit)[0].read_text(encoding="utf-8")
    assert tex.index("\\begin{document}") < tex.index("\\begin{titlepage}")
    assert tex.index("\\end{titlepage}") < tex.index("\\tableofcontents")


# ── Real compile ────────────────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_pdflatex_fits_the_wide_table_and_prints_the_cover(tmp_path: Path) -> None:
    meta, _ = load_book_meta(_meta_file(tmp_path))
    body = "\\chapter{Circuit Elements}\n\n" + WIDE_TABLE
    tex = assemble_document([body], tmp_path / "book.tex", cover=cover_latex(meta))

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:]
    log = (tmp_path / "book.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull \\hbox" not in log  # the table no longer runs off the page

    with pymupdf.open(tmp_path / "book.pdf") as pdf:
        first = pdf[0].get_text()
        text = " ".join(page.get_text() for page in pdf)
    assert "English Translation" in first and "Tsinghua University Press" in first
    assert "Main applications" in text
