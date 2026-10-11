"""
The cover page: the book's English title, authors and publisher.

Who decides
-----------
The ``page-range-finder`` agent already looks at the cover and title pages
to decide where the body starts, so it also reads the book's metadata off
them and writes ``data/work/<book>/book_meta.json`` (hand-editable, like the
other per-book JSON files):

.. code-block:: json

    {"title": "电路原理学习指导与习题集",
     "title_en": "Learning Guide and Exercises for Principle of Electric Circuits",
     "subtitle_en": null,
     "authors": ["朱桂萍", "刘秀成", "徐福媛"],
     "authors_en": ["Zhu Guiping", "Liu Xiucheng", "Xu Fuyuan"],
     "publisher": "清华大学出版社", "publisher_en": "Tsinghua University Press",
     "evidence": "p1 cover, p3 title page", "by": "agent"}

Only the ``*_en`` fields are typeset. Names are in pinyin, family name
first, as Chinese books print them; a publisher with an established English
name uses it.

Assembly (`src.latex.assemble_document`) puts the cover right after
``\\begin{document}``, before the table of contents, and sets the PDF's
title and author metadata. It is generated LaTeX rather than a preamble
macro so that a book with its own ``kit/preamble.tex`` (a copy made before
this existed) gets one too.

A book with no usable ``book_meta.json`` is built without a cover. The
build reports it; nothing fails.

pdflatex typesets Latin-1 only, so an English field holding anything else
(an untranslated title, say) is refused rather than breaking the build.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

BOOK_META_NAME = "book_meta.json"
COVER_LABEL = "English Translation"

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class BookMeta:
    title_en: str
    authors_en: tuple[str, ...] = ()
    publisher_en: str = ""
    subtitle_en: str = ""


def _typesettable(text: str) -> bool:
    return all(ord(char) <= 0xFF for char in text)


def escape_latex(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(char, char) for char in text)


def find_book_meta(kit_dir: Path) -> Path | None:
    """``kit/book_meta.json`` if the kit has its own, else the work dir's."""
    for path in (kit_dir / BOOK_META_NAME, kit_dir.parent / BOOK_META_NAME):
        if path.exists():
            return path
    return None


def load_book_meta(path: Path | None) -> tuple[BookMeta | None, list[str]]:
    """``(meta, problems)``. ``meta`` is None when there is no usable title."""
    if path is None:
        return None, [f"no {BOOK_META_NAME}: the book is built without a cover page"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path} is unreadable ({exc}): no cover page"]
    if not isinstance(data, dict):
        return None, [f"{path} must hold a JSON object: no cover page"]

    problems: list[str] = []

    def field(name: str) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            return ""
        value = " ".join(value.split())
        if not _typesettable(value):
            problems.append(f"{name} {value!r} is not English/Latin-1 text and was left out")
            return ""
        return value

    title = field("title_en")
    authors = []
    for name in data.get("authors_en") or []:
        if isinstance(name, str) and name.strip():
            if _typesettable(name):
                authors.append(" ".join(name.split()))
            else:
                problems.append(f"author {name!r} is not Latin-1 text and was left out")
    meta = BookMeta(
        title_en=title,
        authors_en=tuple(authors),
        publisher_en=field("publisher_en"),
        subtitle_en=field("subtitle_en"),
    )
    if not meta.title_en:
        problems.append(f"{path} has no usable title_en: no cover page")
        return None, problems
    return meta, problems


def cover_latex(meta: BookMeta) -> str:
    """A title page in the style of the reference covers: a small label,
    the title large and bold, the authors, the publisher at the foot."""
    title = escape_latex(meta.title_en)
    authors = escape_latex(", ".join(meta.authors_en))
    lines = [
        "% Cover page, generated from book_meta.json (src/cover.py).",
        "\\hypersetup{pdftitle={" + title + "}"
        + (", pdfauthor={" + authors + "}" if authors else "") + "}",
        "\\begin{titlepage}",
        "  \\centering",
        "  \\vspace*{2.5cm}",
        "  {\\large " + COVER_LABEL + "\\par}",
        "  \\vspace{2.5cm}",
        "  {\\Huge\\bfseries " + title + "\\par}",
    ]
    if meta.subtitle_en:
        lines += ["  \\vspace{1cm}", "  {\\LARGE " + escape_latex(meta.subtitle_en) + "\\par}"]
    if authors:
        lines += ["  \\vspace{3cm}", "  {\\Large " + authors + "\\par}"]
    lines += ["  \\vfill"]
    if meta.publisher_en:
        lines += ["  {\\Large " + escape_latex(meta.publisher_en) + "\\par}"]
    lines += ["  \\vspace*{1cm}", "\\end{titlepage}"]
    return "\n".join(lines)


__all__ = [
    "BOOK_META_NAME",
    "BookMeta",
    "cover_latex",
    "escape_latex",
    "find_book_meta",
    "load_book_meta",
]
