"""
Parse only the body: the page range, the trimmed PDF, and the cross-checks.

Why
---
Front matter (cover, title and copyright pages, preface, table of contents)
and non-content back matter (index, term lists, ads) are where parsing goes
wrong: a TOC parses into dozens of fake headings, an index into pages of
numbers, and every one of them then has to be recognised and cut by the
structure step. Cutting them *before* parsing is cheaper and spends less of
MinerU's 1000-page daily quota. LaTeX builds its own table of contents, so
nothing of value is lost.

Who decides
-----------
An agent looks at page thumbnails (``scripts/render_pages.py``) and writes
``data/work/<book>/page_range.json``; code does the rest deterministically.
The file is hand-editable like the other per-book JSON files:

.. code-block:: json

    {"start": 15, "end": 402, "first_chapter_number": 12,
     "starts_with_part_title": true,
     "excluded": [{"pages": "1-14", "kind": "toc", "note": "..."}],
     "confidence": "high", "evidence": "...", "by": "agent"}

``start``/``end`` are 1-based *physical* page indexes, inclusive -- never the
printed page numbers. The body is one contiguous block: mid-book pages are
never cut.

Page numbers after trimming
---------------------------
``source_body.pdf`` holds pages ``start..end``, so its page 1 is the
original's page ``start``. The offset (``start - 1``) is recorded in the
parts manifest, and `original_page` turns a body page back into the page a
reader of the original would look up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PAGE_RANGE_NAME = "page_range.json"
SOURCE_BODY_NAME = "source_body.pdf"
MANIFEST_PATH = Path("parts") / "manifest.json"

CONFIDENCES = ("high", "medium", "low")


@dataclass(frozen=True)
class PageRange:
    start: int
    end: int
    first_chapter_number: int | None = None
    starts_with_part_title: bool = False
    excluded: list[dict] = field(default_factory=list)
    confidence: str = "medium"
    evidence: str = ""
    by: str = ""

    @property
    def offset(self) -> int:
        """Pages cut from the front: body page n is original page n + offset."""
        return self.start - 1

    @property
    def page_count(self) -> int:
        return self.end - self.start + 1


def load_page_range(path: Path, *, page_count: int | None = None) -> PageRange:
    """Read and validate ``page_range.json``.

    Raises:
        FileNotFoundError: no such file.
        ValueError: the file does not describe one valid, contiguous range
            (inside the PDF, when ``page_count`` is given).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Decide the body's page range first (the page-range-finder "
            "agent writes it; see scripts/render_pages.py), or parse without --page-range."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")

    start, end = data.get("start"), data.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool):
        raise ValueError(f"{path}: start and end must be integers (1-based physical pages)")
    if not 1 <= start <= end:
        raise ValueError(f"{path}: need 1 <= start <= end, got start={start}, end={end}")
    if page_count is not None and end > page_count:
        raise ValueError(f"{path}: end={end} is past the PDF's last page ({page_count})")

    chapter = data.get("first_chapter_number")
    confidence = str(data.get("confidence") or "medium").lower()
    return PageRange(
        start=start,
        end=end,
        first_chapter_number=chapter if isinstance(chapter, int) else None,
        starts_with_part_title=bool(data.get("starts_with_part_title", False)),
        excluded=[e for e in data.get("excluded") or [] if isinstance(e, dict)],
        confidence=confidence if confidence in CONFIDENCES else "medium",
        evidence=str(data.get("evidence") or ""),
        by=str(data.get("by") or ""),
    )


def trim_pdf(pdf_path: Path, page_range: PageRange, out_path: Path, *, stamp: Path | None = None) -> bool:
    """Write pages ``start..end`` of ``pdf_path`` to ``out_path``.

    Skipped (returns False) when ``out_path`` is newer than both the PDF and
    ``stamp`` (the ``page_range.json`` it was cut by), so its mtime -- which
    decides whether the book needs parsing again -- only moves when the range
    or the book changed.
    """
    from src.splitter import extract_pages

    if out_path.exists():
        newest_input = max(
            pdf_path.stat().st_mtime, stamp.stat().st_mtime if stamp and stamp.exists() else 0
        )
        if out_path.stat().st_mtime > newest_input:
            return False
    extract_pages(pdf_path, page_range.start, page_range.end, out_path)
    return True


def record_offset(work_dir: Path, page_range: PageRange | None, source_pdf: Path) -> None:
    """Note the trim in the parts manifest, beside the parts it describes."""
    manifest = work_dir / MANIFEST_PATH
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["page_offset"] = page_range.offset if page_range else 0
    data["trimmed"] = page_range is not None
    data["original_pdf"] = str(source_pdf)
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def page_offset(work_dir: Path) -> int:
    """The recorded trim offset; 0 for an untrimmed (or unparsed) book."""
    manifest = work_dir / MANIFEST_PATH
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8")).get("page_offset", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def was_trimmed(work_dir: Path) -> bool:
    manifest = work_dir / MANIFEST_PATH
    try:
        return bool(json.loads(manifest.read_text(encoding="utf-8")).get("trimmed", False))
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def original_page(n: int, offset: int) -> int:
    """Body page ``n`` (1-based, in ``source_body.pdf``) as a physical page
    of the original PDF."""
    return n + offset


# ── Cross-check: does merged.md start where a body starts? ──────────────────

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)[ \t]*$", re.MULTILINE)
# Front-matter headings that mean the start page is too early.
_FRONT_MATTER_RE = re.compile(
    r"^(?:目\s*录|目\s*次|前\s*言|序\s*言?|序\s*论|内容(?:简介|提要)|出版说明|编者的话|致\s*读者|"
    r"再版说明|第\S{1,3}版(?:前言|说明)|使用说明|本书(?:导读|使用说明)|"
    r"contents|table of contents|preface|foreword|introduction to the .* edition)\s*$",
    re.IGNORECASE,
)


def first_heading_warning(text: str) -> str | None:
    """A warning when ``merged.md`` does not begin with a chapter, or a part
    followed by a chapter; None when it does.

    "Chapter" is judged without assuming a title format: a 第N章 title, or
    numbered sections (N.1, N.2 ...) under the heading, which is how this
    corpus's chapters look even when the parser dropped 第N章.
    """
    from src.structure import CHAPTER, PART, find_divisions, rule_opinion

    divisions = find_divisions(text)
    if not divisions:
        first = _HEADING_RE.search(text)
        if first is None:
            return "merged.md has no headings at all; check the page range and the parse"
        return f"merged.md has no top-level heading; the first heading is {first.group(2)!r}"

    def kind(index: int) -> str | None:
        division = divisions[index]
        if _FRONT_MATTER_RE.match(division.title.strip()):
            return "front matter"
        rule_opinion(division)
        return division.rule_kind

    first = kind(0)
    if first == CHAPTER:
        return None
    if first == PART and len(divisions) > 1 and kind(1) == CHAPTER:
        return None
    title = divisions[0].title
    if first == "front matter":
        return (
            f"merged.md starts with {title!r}, which is front matter: the page range starts "
            "too early (or the TOC was not cut)"
        )
    return (
        f"merged.md starts with {title!r}, which does not look like a chapter or a part "
        "followed by a chapter; the start page may be wrong"
    )


__all__ = [
    "PAGE_RANGE_NAME",
    "SOURCE_BODY_NAME",
    "PageRange",
    "first_heading_warning",
    "load_page_range",
    "original_page",
    "page_offset",
    "record_offset",
    "trim_pdf",
    "was_trimmed",
]
