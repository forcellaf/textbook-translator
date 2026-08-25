"""
Split a book's merged Markdown (``merged.md``, produced by
``src.merger.merge_chunks`` or, for small books, directly by MinerU) into
one Markdown file per chapter, so a future translation stage can process a
book one chapter at a time instead of one giant document at a time.

Chapter boundaries are detected AFTER MinerU has already produced clean
Markdown, never on the raw PDF: scanned textbooks rarely have any reliable
structure to key off before OCR/layout parsing has happened.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from src.merger import rewrite_image_paths
from src.models import ChapterInfo, ChapterResult

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"

# A single chapter is not allowed to dominate the whole document -- past
# this share of total characters, detection is considered unreliable.
_MAX_SINGLE_CHAPTER_SHARE = 0.60
_MIN_CHAPTERS_FOR_CONFIDENCE = 2

_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# Numbered/known chapter-boundary patterns, checked against a heading's text
# (with leading '#'s already stripped). Kept as a plain list so new
# textbook conventions can be added without touching the detection logic.
# Chinese numerals cover common textbook chapter numbering; Arabic digits
# are matched via `Chapter\s+\d+` and inside the CJK patterns alike.
CHAPTER_HEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"第[0-9零一二三四五六七八九十百千两]+章"),  # 第X章 (chapter)
    re.compile(r"第[0-9零一二三四五六七八九十百千两]+篇"),  # 第X篇 (part)
    re.compile(r"第[0-9零一二三四五六七八九十百千两]+部分"),  # 第X部分 (section/part)
    re.compile(r"Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"Part\s+[IVXLCDM\d]+", re.IGNORECASE),
    re.compile(r"Appendix\b", re.IGNORECASE),
    re.compile(r"附录"),  # appendix
    re.compile(r"习题"),  # exercises
    re.compile(r"练习"),  # practice/exercises
    re.compile(r"索引"),  # index
    re.compile(r"参考文献"),  # references/bibliography
]

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_ORDINAL_RE = re.compile(r"第([0-9零一二三四五六七八九十百千两]+)[章篇部分]")
_ENGLISH_ORDINAL_RE = re.compile(r"Chapter\s+(\d+)", re.IGNORECASE)


def split_into_chapters(merged_md_path: Path, output_dir: Path) -> ChapterResult:
    """
    Split ``merged_md_path`` into per-chapter Markdown files under
    ``output_dir``.

    Detection strategy
    ---------------------
    1. Collect every ATX heading (``^#{1,6}\\s``) in the document.
    2. A heading is a chapter-start *candidate* if its text matches any
       pattern in ``CHAPTER_HEADING_PATTERNS`` (numbered chapters, parts,
       appendices, exercises, index, references -- extend that list for new
       textbook conventions). Candidates can be at any heading level.
    3. If no heading matches any pattern, fall back to top-level headings
       (``#`` and ``##``) as chapter candidates.
    4. Content before the first candidate becomes ``00_front_matter.md``
       (only written if non-empty). Content from each candidate heading up
       to the next candidate (or end of document) becomes that chapter --
       this naturally puts back-matter sections (index/references/appendix)
       into their own chapter files whenever they have their own heading,
       and folds trailing content into the last chapter otherwise.

    Confidence checks (all logged; failing any one disables confidence)
    -----------------------------------------------------------------------
    * At least ``_MIN_CHAPTERS_FOR_CONFIDENCE`` chapters were found.
    * Chapter ordinals (parsed from "第N章"/"Chapter N"-style titles, where
      present) are non-decreasing in document order.
    * No single chapter holds more than ``_MAX_SINGLE_CHAPTER_SHARE`` of the
      document's total characters.

    If any check fails, ``detection_confidence`` is set to False and a
    single chapter spanning the entire document is written instead (the
    translation stage is expected to fall back to token-based chunking via
    ``MAX_CHUNK_TOKENS`` in that case). This function never raises for a
    detection failure -- only for I/O problems reading/writing files.

    Image links: merged.md's image references are relative to
    ``book_work_dir`` (e.g. ``chunks/book_part_001/images/x.jpg``). Since
    chapter files live one directory deeper, in ``chapters/``, every
    reference is rewritten with an extra ``../`` so it still resolves
    (``../chunks/book_part_001/images/x.jpg``).

    Idempotency: if ``output_dir/manifest.json`` exists and is newer than
    ``merged_md_path``, the existing split is reused.

    Args:
        merged_md_path: Path to the book's merged Markdown file.
        output_dir: Directory to write chapter files + manifest.json into
            (typically ``book_work_dir/chapters``).

    Returns:
        A ChapterResult describing every chapter file written.

    Raises:
        FileNotFoundError: If ``merged_md_path`` doesn't exist.
    """
    if not merged_md_path.exists():
        raise FileNotFoundError(f"Merged markdown not found: {merged_md_path}")

    manifest_path = output_dir / _MANIFEST_NAME
    if manifest_path.exists() and manifest_path.stat().st_mtime > merged_md_path.stat().st_mtime:
        existing = _load_manifest(manifest_path, merged_md_path)
        if existing is not None and all(c.file_path.exists() for c in existing.chapters):
            logger.info("Already split into chapters, skipping: %s", manifest_path)
            return existing
        logger.warning(
            "Manifest %s is stale or references missing chapter files; re-splitting",
            manifest_path,
        )

    text = merged_md_path.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)

    headings = [(m.start(), m.group(1), m.group(2).strip()) for m in _ATX_HEADING_RE.finditer(text)]
    candidates = [h for h in headings if _matches_chapter_pattern(h[2])]
    if not candidates:
        candidates = [h for h in headings if len(h[1]) <= 2]  # '#' or '##'

    sections = _slice_sections(text, candidates)
    confidence, reason = _check_confidence(sections, len(text))

    if not confidence:
        logger.warning("Chapter detection not confident (%s); using single whole-document chapter", reason)
        sections = [(0, len(text), "", "Full Document")]

    chapters: list[ChapterInfo] = [
        _build_chapter(i, start, end, source_heading, title, output_dir)
        for i, (start, end, source_heading, title) in enumerate(sections)
    ]

    _write_files(chapters, text)
    _write_manifest(manifest_path, merged_md_path, chapters, confidence)

    logger.info(
        "Split '%s' into %d chapter file(s) (confidence=%s) in %s",
        merged_md_path.name,
        len(chapters),
        confidence,
        output_dir,
    )
    return ChapterResult(
        source_path=merged_md_path,
        output_dir=output_dir,
        chapters=tuple(chapters),
        detection_confidence=confidence,
        manifest_path=manifest_path,
    )


def _matches_chapter_pattern(heading_text: str) -> bool:
    return any(p.search(heading_text) for p in CHAPTER_HEADING_PATTERNS)


def _slice_sections(
    text: str, candidates: list[tuple[int, str, str]]
) -> list[tuple[int, int, str, str]]:
    """Turn a list of (char_offset, hashes, title) candidate headings into a
    list of (start, end, source_heading, title) sections covering the whole
    document: optional front matter (source_heading="") followed by one
    section per candidate."""
    if not candidates:
        return [(0, len(text), "", "Full Document")]

    sections: list[tuple[int, int, str, str]] = []
    first_start = candidates[0][0]
    if text[:first_start].strip():
        sections.append((0, first_start, "", "Front Matter"))

    for i, (start, hashes, title) in enumerate(candidates):
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        source_heading = f"{hashes} {title}"
        sections.append((start, end, source_heading, title))

    return sections


def _check_confidence(sections: list[tuple[int, int, str, str]], total_len: int) -> tuple[bool, str]:
    real_chapters = [s for s in sections if s[2]]  # exclude front matter (no source_heading)
    if len(real_chapters) < _MIN_CHAPTERS_FOR_CONFIDENCE:
        return False, f"only {len(real_chapters)} chapter(s) found (need >= {_MIN_CHAPTERS_FOR_CONFIDENCE})"

    ordinals = [_extract_ordinal(s[3]) for s in real_chapters]
    known = [o for o in ordinals if o is not None]
    if len(known) >= 2:
        for prev, cur in zip(known, known[1:]):
            if cur < prev:
                return False, f"chapter ordinals are not ascending (found {prev} then {cur})"

    if total_len > 0:
        for s in sections:
            share = (s[1] - s[0]) / total_len
            if share > _MAX_SINGLE_CHAPTER_SHARE:
                return False, f"chapter '{s[3]}' holds {share:.0%} of total content (> {_MAX_SINGLE_CHAPTER_SHARE:.0%})"

    return True, ""


def _extract_ordinal(title: str) -> int | None:
    match = _ORDINAL_RE.search(title)
    if match:
        return _cn_to_int(match.group(1))
    match = _ENGLISH_ORDINAL_RE.search(title)
    if match:
        return int(match.group(1))
    return None


def _cn_to_int(s: str) -> int | None:
    """Parse a small Chinese numeral (as used in chapter titles, e.g.
    '十二' -> 12) or a plain digit string. Returns None on unknown chars."""
    if s.isdigit():
        return int(s)
    section = 0
    num = 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            section += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return None
    return section + num


def _sanitize_filename(title: str) -> str:
    """Sanitize a chapter title into a filesystem-safe filename component,
    preserving CJK characters."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_.")
    cleaned = cleaned[:80]
    return cleaned or "untitled"


def _build_chapter(
    index: int,
    start: int,
    end: int,
    source_heading: str,
    title: str,
    output_dir: Path,
) -> ChapterInfo:
    is_front_matter = index == 0 and not source_heading
    display_title = "Front Matter" if is_front_matter else title
    filename = f"{index:02d}_{_sanitize_filename(display_title)}.md"
    file_path = output_dir / filename
    return ChapterInfo(
        index=index,
        title=display_title,
        source_heading=source_heading,
        char_count=end - start,
        char_start=start,
        char_end=end,
        file_path=file_path,
    )


def _write_files(chapters: list[ChapterInfo], text: str) -> None:
    for chapter in chapters:
        body = text[chapter.char_start : chapter.char_end]
        body = rewrite_image_paths(body, Path(".."))
        chapter.file_path.write_text(body.strip("\n") + "\n", encoding="utf-8")


def _write_manifest(
    manifest_path: Path, source_path: Path, chapters: list[ChapterInfo], confidence: bool
) -> None:
    payload = {
        "source_path": str(source_path),
        "detection_confidence": confidence,
        "chapters": [
            {**asdict(c), "file_path": str(c.file_path)} for c in chapters
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_manifest(manifest_path: Path, source_path: Path) -> ChapterResult | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        chapters = tuple(
            ChapterInfo(
                index=c["index"],
                title=c["title"],
                source_heading=c["source_heading"],
                char_count=c["char_count"],
                char_start=c["char_start"],
                char_end=c["char_end"],
                file_path=Path(c["file_path"]),
            )
            for c in payload["chapters"]
        )
        return ChapterResult(
            source_path=source_path,
            output_dir=manifest_path.parent,
            chapters=chapters,
            detection_confidence=payload["detection_confidence"],
            manifest_path=manifest_path,
        )
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not parse existing manifest %s: %s", manifest_path, exc)
        return None
