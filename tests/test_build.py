"""Tests for src.build (assembled Markdown -> PDF via pandoc).

The flag set is asserted directly rather than by building, so the suite does
not need pandoc or a TeX distribution installed. Each flag here fixes
something observed going wrong in a real build.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.build import HEADER_TEX, BuildError, build_pdf, pandoc_command


def _argv(tmp_path: Path, **kwargs: object) -> list[str]:
    markdown = tmp_path / "book.md"
    markdown.write_text("# Title\n", encoding="utf-8")
    return pandoc_command(markdown, tmp_path / "book.pdf", **kwargs)  # type: ignore[arg-type]


def test_command_uses_xelatex_with_a_toc(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    assert "--pdf-engine=xelatex" in argv
    assert "--toc" in argv


def test_command_sets_the_book_class(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    assert "documentclass=book" in argv
    assert "geometry:margin=2.5cm" in argv


def test_openany_removes_the_blank_versos(tmp_path: Path) -> None:
    """Without it the book class starts every chapter on a recto page."""
    assert "classoption=openany" in _argv(tmp_path)


def test_the_verified_header_is_included(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    assert "-H" in argv
    assert str(HEADER_TEX) in argv


def test_the_header_loads_graphicx_before_setting_gin_keys() -> None:
    """Pandoc only emits \\usepackage{graphicx} when the document contains an
    image, so a figure-free chapter dies on "width undefined" without this."""
    header = HEADER_TEX.read_text(encoding="utf-8")
    assert header.index("\\usepackage{graphicx}") < header.index("\\setkeys{Gin}")


def test_resource_path_covers_the_markdown_dir_and_its_images(tmp_path: Path) -> None:
    argv = _argv(tmp_path)
    resource_path = argv[argv.index("--resource-path") + 1]
    parts = resource_path.split(os.pathsep)

    assert str(tmp_path) in parts
    assert str(tmp_path / "images") in parts


def test_cjk_font_is_omitted_unless_asked_for(tmp_path: Path) -> None:
    """A fully translated book needs no CJK font."""
    assert not any("CJKmainfont" in arg for arg in _argv(tmp_path))
    assert any("CJKmainfont=Noto Sans CJK SC" == arg for arg in _argv(tmp_path, cjk_font="Noto Sans CJK SC"))


def test_missing_markdown_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_pdf(tmp_path / "nope.md")


def test_missing_pandoc_is_an_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    markdown = tmp_path / "book.md"
    markdown.write_text("# Title\n", encoding="utf-8")
    monkeypatch.setattr("src.build.shutil.which", lambda _: None)

    with pytest.raises(BuildError, match="pandoc.org/installing"):
        build_pdf(markdown)


def test_a_pandoc_failure_surfaces_the_tail_of_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LaTeX reports its error near the end of a very long log."""
    markdown = tmp_path / "book.md"
    markdown.write_text("# Title\n", encoding="utf-8")

    class Result:
        returncode = 43
        stdout = ""
        stderr = "noise\n" * 5000 + "! Undefined control sequence \\frobnicate"

    monkeypatch.setattr("src.build.shutil.which", lambda _: "/usr/bin/pandoc")
    monkeypatch.setattr("src.build.subprocess.run", lambda *a, **k: Result())

    with pytest.raises(BuildError, match=r"Undefined control sequence"):
        build_pdf(markdown)
