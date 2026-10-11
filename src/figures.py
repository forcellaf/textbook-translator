"""
Figure metadata, decided outside the translating model.

Every image in a book gets one record in ``figures.json``, keyed by its
**image filename** (MinerU's content hash). The ``IMG_nnnn`` token is not a
usable key: tokens are reassigned whenever the kit is rebuilt, filenames are
not. A record says three things the translation cannot be trusted with:

``number``
    The source's own figure number (``18.3``, ``G.2``), copied from its
    caption. The build prints exactly this, never LaTeX's counter: a counted
    number drifts the moment one non-figure is captioned, and on the
    reference book a decorative chapter glyph became "Figure 18.1" and put
    every later figure in that chapter one off from the prose citing it.
``width``
    The figure's width on the source page as a fraction of the body-text
    width, so a figure that was small in the original is small in the PDF.
``keep``
    Whether it is a figure at all. Decorations, text rendered as an image
    and crop fragments are dropped -- but only on a confident classifier
    verdict, because dropping a real figure is the expensive error and
    keeping a decoration is a cheap one.

The translating model only copies ``IMG_nnnn`` tokens, which it does
reliably; assembly (``src.kit``) applies these records at build time. So a
wrong width, number or drop is fixed by editing one JSON line and
rebuilding, and translations made before this module existed stay valid.

Records with ``"by": "manual"`` are never overwritten -- the same convention
as ``structure.json``.

MinerU's layout data (``content_list.json``)
--------------------------------------------
Inspected on a real result archive for the first 8 pages of the reference
book (``vlm`` backend, October 2026; the archive member is
``<uuid>_content_list.json``, beside a page-nested ``_content_list_v2.json``
that is not used):

* ``bbox`` is ``[x0, y0, x1, y1]`` on a **normalized 0-1000** page grid, not
  in PDF points: the reference PDF's pages are 1566 x 2246 points and the
  largest coordinate in the archive is 957. Widths here are a ratio of two
  bbox widths on the same grid, so the code does not depend on which it is.
* Captions arrive in **both** places: ``image_caption`` (a list of strings)
  on the entry, and as a text line after the image in ``full.md``. The list
  mixes sub-panel labels with the caption -- ``["(b)", "图 12.2 氢分子..."]``
  on the last panel, ``["(a)"]`` on the first -- so every string is tried.
  ``image_caption`` is preferred; the markdown is the fallback, and the only
  source for a book parsed before this file was kept.
* Not every figure is ``"type": "image"``. The VLM backend files plots as
  ``"type": "chart"``, with the caption in ``chart_caption`` (figure 12.1's
  panels, there). Any entry with an ``img_path`` is therefore an image here,
  and its caption is read from ``<type>_caption``.
* Body text is ``"type": "text"``; headings carry a ``text_level``. Running
  heads and page numbers have their own types and do not count as text.
"""

from __future__ import annotations

import html
import json
import logging
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import (
    API_MAX_RETRIES,
    FIGURE_DROP_CONFIDENCE,
    VISION_MIN_INTERVAL_SECONDS,
)
from src.llm.base import VisionLLM, image_mime_type
from src.profiler import extract_json_object

logger = logging.getLogger(__name__)

FIGURES_NAME = "figures.json"
REVIEW_NAME = "figures_review.html"
CONTENT_LIST_NAME = "content_list.json"
# Written by scripts/link_figures.py for a book parsed before content lists
# were kept: the re-parse's layout, re-keyed to this book's image filenames.
LINKED_CONTENT_LIST_NAME = "linked_content_list.json"

DEFAULT_WIDTH = 0.6
MIN_WIDTH = 0.20
MAX_WIDTH = 0.90

# Text blocks shorter than this are left out of the reference width: short
# lines, headings and equation labels are narrower than the text block and
# would pull the median down. A Chinese body line holds ~38 characters, so
# 60 guarantees a block that wraps -- and so spans the full measure.
_REFERENCE_MIN_CHARS = 60

BY_CAPTION = "caption"
BY_CLASSIFIER = "classifier"
BY_DEFAULT = "default"
BY_MANUAL = "manual"

FIGURE = "figure"
DECORATION = "decoration"
LABELS = (FIGURE, DECORATION, "text_as_image", "fragment")

# Review flags. Anything flagged is kept and listed in the review sheet.
FLAG_NO_BBOX = "no-bbox"
FLAG_GAP = "numbering-gap"
FLAG_LOW_CONFIDENCE = "low-confidence"
FLAG_OTHER_LABEL = "not-a-figure"
FLAG_MALFORMED = "malformed-reply"
FLAG_ERROR = "classifier-error"
FLAG_UNCLASSIFIED = "unclassified"
FLAG_WOULD_DROP = "would-drop"

MODE_CLASSIFY = "classify"
MODE_DRY_RUN = "dry-run"
MODE_OFF = "off"

# Backoff for transient vision failures (429, 5xx, timeouts). Module-level so
# tests can neutralize the sleeps.
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=60)

# How much markdown on each side of an image the classifier sees.
_CONTEXT_CHARS = 300


# ── Records ─────────────────────────────────────────────────────────────────


@dataclass
class FigureRecord:
    number: str | None = None
    width: float = DEFAULT_WIDTH
    keep: bool = True
    by: str = BY_DEFAULT
    label: str | None = None
    confidence: float | None = None
    reason: str | None = None
    # The number of the captioned figure this image is a sub-panel of, e.g.
    # panel (a) of a figure whose one caption sits under panel (b).
    panel_of: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "FigureRecord":
        width = data.get("width")
        confidence = data.get("confidence")
        return cls(
            number=_str_or_none(data.get("number")),
            width=float(width) if isinstance(width, (int, float)) else DEFAULT_WIDTH,
            keep=bool(data.get("keep", True)),
            by=str(data.get("by") or BY_DEFAULT),
            label=_str_or_none(data.get("label")),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            reason=_str_or_none(data.get("reason")),
            panel_of=_str_or_none(data.get("panel_of")),
            flags=[str(f) for f in data.get("flags") or []],
        )


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_figures(path: Path) -> dict[str, FigureRecord]:
    """Read ``figures.json``; a missing or unreadable file is an empty map."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        name: FigureRecord.from_json(entry)
        for name, entry in data.items()
        if isinstance(entry, dict)
    }


def save_figures(path: Path, records: dict[str, FigureRecord]) -> None:
    payload = {name: record.to_json() for name, record in records.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def settle_by(record: FigureRecord) -> str:
    """What decided a (non-manual) record, for the ``by`` field.

    A width that fell back to the default outranks everything else, because
    it is the thing a reader of the review sheet has to fix -- except on a
    dropped image, which has no width to fix.
    """
    if not record.keep and record.label in LABELS:
        return BY_CLASSIFIER
    if FLAG_NO_BBOX in record.flags:
        return BY_DEFAULT
    if record.number or record.panel_of:
        return BY_CAPTION
    if record.label in LABELS:
        return BY_CLASSIFIER
    return BY_DEFAULT


# ── Numbers ─────────────────────────────────────────────────────────────────

# Books number figures differently, so this assumes no depth and no prefix:
# an optional letter prefix, then groups of digits joined by `.` or `-`, at
# least two groups, and an optional prime. Real forms it has to read:
#   physics_2:    `图 18.3`, `图18.3`, `图 G.2` (a reading's letter)
#   eee_practice: `图 1.2.1` (chapter.section.figure), `图 T1.4` (exercise
#                 figures), `图A1.8-1` (appendix), `图 23.3.3'` (a variant)
# A two-level pattern read `图 1.2.1` as 1.2 -- 606 figures collapsed onto 49
# numbers. Anchored at the start of a caption: prose that merely cites a
# figure ("如图12.7所示") does not start with 图. `图 12.5(b)` stays 12.5.
_CAPTION_NUMBER_RE = re.compile(
    r"^\s*(?:图|圖|Fig(?:ure)?\.?)\s*"
    r"((?:[A-Z]{1,2}\s*\d*|\d+)(?:\s*[.．·\-–]\s*\d+)+'?)"
)
_NUMBER_SEPARATORS = str.maketrans({"．": ".", "·": ".", "–": "-"})
# A caption is a short label, not a paragraph. A prose paragraph that opens
# with a figure reference ("图12.4画出了两个点电荷 ... 的叠加情况。") is
# longer and contains a full stop.
_CAPTION_MAX_CHARS = 100


def caption_number(text: str) -> str | None:
    """The figure number a caption line starts with, else None."""
    line = text.strip()
    if not line or len(line) > _CAPTION_MAX_CHARS or "。" in line:
        return None
    match = _CAPTION_NUMBER_RE.match(line)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1)).translate(_NUMBER_SEPARATORS)


def caption_from_entry(entry: dict) -> str | None:
    """The number in a content-list entry's caption list.

    ``image_caption`` for an image, ``chart_caption`` for a chart -- a list of
    strings in the archive inspected, panel labels included. A bare string is
    accepted too, in case the schema varies by backend.
    """
    captions = entry.get(f"{entry.get('type')}_caption") or entry.get("image_caption") or []
    if isinstance(captions, str):
        captions = [captions]
    for caption in captions:
        number = caption_number(str(caption))
        if number:
            return number
    return None


_SPLIT_NUMBER_RE = re.compile(r"^(.+)[.\-](\d+)'?$")


def split_number(number: str) -> tuple[str, int] | None:
    """A figure number as ``(sequence it belongs to, position in it)``.

    The position is the last group, whatever the depth: ``"18.3"`` ->
    ``("18", 3)``, ``"1.2.1"`` -> ``("1.2", 1)``, ``"A1.8-1"`` ->
    ``("A1.8", 1)``. None if it is not of that shape.
    """
    match = _SPLIT_NUMBER_RE.match(number)
    return (match.group(1), int(match.group(2))) if match else None


# ── Widths ──────────────────────────────────────────────────────────────────


def _read_bbox(entry: dict) -> tuple[float, float, float, float] | None:
    bbox = entry.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def reference_width(entries: list[dict]) -> float | None:
    """Median bbox width of the long body-text blocks across the book.

    That is the width of the source's text block, in whatever units the
    bbox uses -- so an image's bbox width divided by it is the fraction of
    ``\\textwidth`` the figure should take, whether the coordinates are PDF
    points or a normalized grid.
    """
    widths = []
    for entry in entries:
        if entry.get("type") != "text" or entry.get("text_level"):
            continue
        if len(str(entry.get("text") or "").strip()) < _REFERENCE_MIN_CHARS:
            continue
        bbox = _read_bbox(entry)
        if bbox:
            widths.append(bbox[2] - bbox[0])
    return statistics.median(widths) if widths else None


def width_for(entry: dict | None, reference: float | None) -> float | None:
    """``clamp(image width / reference, 0.20, 0.90)``, or None without a bbox.

    Only the width comes from the bbox. The aspect ratio comes from the image
    file at compile time (``keepaspectratio``), so a crop that differs
    slightly from the bbox cannot distort the figure.
    """
    if entry is None or not reference:
        return None
    bbox = _read_bbox(entry)
    if bbox is None:
        return None
    ratio = (bbox[2] - bbox[0]) / reference
    return round(min(MAX_WIDTH, max(MIN_WIDTH, ratio)), 2)


def load_content_list(work_dir: Path) -> list[dict]:
    """The book's layout entries: the real content list, else a linked one."""
    for name in (CONTENT_LIST_NAME, LINKED_CONTENT_LIST_NAME):
        path = work_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable %s: %s", path, exc)
            continue
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
    return []


def image_entries(entries: list[dict]) -> dict[str, dict]:
    """Content-list entries that are pictures, by image filename (first one
    wins). Any type with an ``img_path`` counts: ``image`` and ``chart`` in
    the archive inspected."""
    found: dict[str, dict] = {}
    for entry in entries:
        path = entry.get("img_path")
        if path:
            found.setdefault(Path(str(path)).name, entry)
    return found


# ── Markdown scan ───────────────────────────────────────────────────────────

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_HTML_IMAGE_RE = re.compile(r'<img\b[^>]*\bsrc="([^"]+)"[^>]*>', re.IGNORECASE)
# A sub-panel label on its own line: `(a)`, `（b）`, `a)`.
_PANEL_LABEL_RE = re.compile(r"^\s*[(（]?\s*[a-zA-Z]\s*[)）]\s*$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LATEX_DIVISION_RE = re.compile(r"^\s*\\(part|chapter)\*?\s*\{(.*)\}\s*$")
_CHAPTER_WORD_RE = re.compile(r"第\s*\S{0,4}\s*([章篇])")
# A short line inside a figure's paragraph that is not its caption: a label
# printed in the figure ("等离子体 感应电流"), a portrait's name.
_FIGURE_LABEL_MAX_CHARS = 20


@dataclass
class ImageSite:
    """One image as it appears in the markdown, in reading order."""

    name: str  # filename, the record key
    path: str  # as written in the markdown
    line: int  # 0-based
    number: str | None = None  # from a caption line after it
    joins_next: bool = False  # only panel labels between it and the next image
    # A trailing panel the parser put after its figure's caption:
    # `![](a)` `(a)` `图 23.1 ...` `![](b)` `(b)`. The caption's number.
    trails_caption: str | None = None
    after_heading: str | None = None  # "chapter: 磁力" when it opens one
    before: str = ""
    after: str = ""


def _image_refs(line: str) -> list[str]:
    return [m.group(1) for m in _MD_IMAGE_RE.finditer(line)] + [
        m.group(1) for m in _HTML_IMAGE_RE.finditer(line)
    ]


def _division_heading(line: str) -> str | None:
    """``"chapter: 磁力"`` / ``"part: ..."`` if ``line`` opens one, else None."""
    match = _LATEX_DIVISION_RE.match(line)
    if match:
        return f"{match.group(1)}: {match.group(2)}"
    match = _MD_HEADING_RE.match(line)
    if not match:
        return None
    title = match.group(2).strip()
    word = _CHAPTER_WORD_RE.search(title)
    if word:
        return f"{'part' if word.group(1) == '篇' else 'chapter'}: {title}"
    if len(match.group(1)) == 1:
        return f"chapter: {title}"
    return None


def _opening_heading(lines: list[str], index: int) -> str | None:
    """The chapter/part heading ``lines[index]`` directly follows, if any.

    Walks back over blank lines and over consecutive headings, because the
    parser often splits one title in two (``## 第18章`` / ``## 磁力``).
    """
    found: str | None = None
    for line in reversed(lines[:index]):
        if not line.strip():
            continue
        if not (_MD_HEADING_RE.match(line) or _LATEX_DIVISION_RE.match(line)):
            break
        found = found or _division_heading(line)
    return found


def _scan_after(lines: list[str], index: int) -> tuple[str | None, bool]:
    """Look past the image on ``lines[index]`` for its caption.

    Returns ``(number, joins_next)``. Skips sub-panel labels, one blank line,
    and short in-figure labels inside the image's own paragraph; stops at
    anything else. Reaching another image after a panel label means this one
    is a panel of the same figure, whose caption comes after the last panel.
    Without the label it is just the next image: an uncaptioned image right
    before a figure is not thereby part of it.
    """
    blanks = 0
    labelled = False
    for line in lines[index + 1 : index + 8]:
        text = line.strip()
        if not text:
            blanks += 1
            if blanks > 1:
                return None, False
            continue
        if _image_refs(text):
            return None, labelled
        if _PANEL_LABEL_RE.match(text):
            labelled = True
            continue
        number = caption_number(text)
        if number:
            return number, False
        if blanks == 0 and len(text) <= _FIGURE_LABEL_MAX_CHARS and "。" not in text:
            continue
        return None, False
    return None, False


def _trailing_panel_of(lines: list[str], index: int) -> str | None:
    """The caption number a panel image directly after a caption belongs to.

    Needs both ends: the image is labelled as a panel (``(b)`` on the next
    line), and the nearest text line before it is a numbered caption.
    """
    following = next((line.strip() for line in lines[index + 1 : index + 3] if line.strip()), "")
    if not _PANEL_LABEL_RE.match(following):
        return None
    for line in reversed(lines[max(0, index - 3) : index]):
        text = line.strip()
        if not text:
            continue
        return caption_number(text)
    return None


def _context(text: str) -> str:
    """Markdown context with image references replaced, for the classifier."""
    text = _MD_IMAGE_RE.sub("[image]", text)
    return _HTML_IMAGE_RE.sub("[image]", text)


def scan_markdown(text: str) -> list[ImageSite]:
    """Every distinct image in ``text``, in reading order."""
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    sites: list[ImageSite] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        refs = _image_refs(line)
        if not refs:
            continue
        number, joins = _scan_after(lines, index)
        heading = _opening_heading(lines, index)
        start, end = offsets[index], offsets[index + 1]
        before = _context(text[max(0, start - _CONTEXT_CHARS * 2) : start])[-_CONTEXT_CHARS:]
        after = _context(text[end : end + _CONTEXT_CHARS * 2])[:_CONTEXT_CHARS]
        for position, ref in enumerate(refs):
            name = Path(ref).name
            if name in seen:
                continue
            seen.add(name)
            last = position == len(refs) - 1
            sites.append(
                ImageSite(
                    name=name,
                    path=ref,
                    line=index,
                    number=number if last else None,
                    joins_next=joins if last else True,
                    trails_caption=_trailing_panel_of(lines, index) if position == 0 else None,
                    after_heading=heading if position == 0 else None,
                    before=before,
                    after=after,
                )
            )
    return sites


# ── Classification ──────────────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = """You check images extracted from a scanned textbook page by a layout parser. \
Most are real figures, but the parser also extracts things that are not figures. Decide which this one is.

Labels:
- "figure": a diagram, graph, photograph, portrait, illustration, table image, or a sub-panel of one. \
Anything a reader would need to see. When in doubt, it is a figure.
- "decoration": an ornament with no content. Textbooks repeat a few of these many times; they are \
decorations even when the crop is small or blurry:
  * a chapter-opening drop cap: one large, often coloured, stylized character (地, 磁, 超 ...), \
usually on a tinted block;
  * a section icon: a small pictogram that marks a recurring section such as a summary (提要) or \
exercises (习题) -- for example a pencil beside a coloured box -- or a banner with that section's \
title;
  * a decorative border or rule, a page-corner ornament, a background tint.
- "text_as_image": ordinary body text, a heading or an equation captured as a picture, such as a \
section number ("G.2") printed on a tinted badge.
- "fragment": a meaningless crop: a sliver of a figure, a stray mark, an almost empty area. Not an \
ornament: a recognisable icon or glyph is a "decoration", however small.

A faint or low-contrast line drawing is still a "figure": printing can be very light.

Dropping a real figure loses content from the book, so only answer "decoration" when the image \
carries no information at all. Give your actual certainty: 0.95 or more only when there is no doubt.

Reply with JSON only, no other text:
{"label": "figure" | "decoration" | "text_as_image" | "fragment", "confidence": 0.0-1.0, "reason": "<one sentence>"}"""


@dataclass(frozen=True)
class Verdict:
    label: str | None
    confidence: float | None
    reason: str | None
    flag: str | None = None  # FLAG_MALFORMED / FLAG_ERROR when it failed


def _user_text(site: ImageSite) -> str:
    opener = (
        f"This image is the first block after a {site.after_heading.split(':')[0]} heading "
        f"({site.after_heading.split(': ', 1)[-1]})."
        if site.after_heading
        else "This image is not the first block after a chapter or part heading."
    )
    return (
        f"{opener}\n\n"
        f"Text before the image:\n<<<\n{site.before.strip() or '(start of book)'}\n>>>\n\n"
        f"Text after the image:\n<<<\n{site.after.strip() or '(end of book)'}\n>>>\n\n"
        "Classify the attached image."
    )


def parse_verdict(raw: str) -> Verdict:
    """Read the classifier's JSON. Anything unusable is a malformed reply."""
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return Verdict(None, None, f"malformed reply: {raw[:120]!r}", FLAG_MALFORMED)
    label = data.get("label")
    confidence = data.get("confidence")
    if label not in LABELS or not isinstance(confidence, (int, float)) or isinstance(
        confidence, bool
    ):
        return Verdict(None, None, f"malformed reply: {raw[:120]!r}", FLAG_MALFORMED)
    reason = data.get("reason")
    return Verdict(
        str(label),
        max(0.0, min(1.0, float(confidence))),
        str(reason).strip() if reason else None,
    )


class FatalClassifierError(RuntimeError):
    """A permanent API failure (bad key, no balance): stop calling."""


def classify_image(vision: VisionLLM, image: bytes, site: ImageSite) -> Verdict:
    """One vision call for one image, with the transient-retry schedule.

    Fails open: a malformed reply or an exhausted retry schedule comes back
    as a verdict with no label, which keeps the image. Only a permanent
    failure raises, so the caller stops spending calls that cannot succeed.
    """
    mime = image_mime_type(image)
    if mime is None:
        return Verdict(None, None, "unsupported image format", FLAG_ERROR)

    retryer = Retrying(
        stop=stop_after_attempt(max(1, API_MAX_RETRIES)),
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    try:
        raw = retryer(
            vision.generate_with_images,
            CLASSIFY_SYSTEM_PROMPT,
            _user_text(site),
            [(image, mime)],
            0.0,
        )
    except ValueError as exc:
        raise FatalClassifierError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - fail open on anything else
        return Verdict(None, None, f"API failure: {exc}"[:300], FLAG_ERROR)
    return parse_verdict(raw)


# ── The drop guard ──────────────────────────────────────────────────────────


def numbering_gaps(sites: list[ImageSite], numbers: dict[str, str | None]) -> set[str]:
    """Images that sit where a caption number is missing.

    Collects the caption numbers in reading order; an uncaptioned image
    between ``18.3`` and ``18.5`` -- or before ``18.2`` with no ``18.1`` --
    is probably the real figure whose caption the parser lost, so it must
    never be dropped automatically.
    """
    ordered = [
        (index, parsed)
        for index, site in enumerate(sites)
        if (number := numbers.get(site.name)) and (parsed := split_number(number))
    ]
    in_gap: set[str] = set()
    for index, site in enumerate(sites):
        if numbers.get(site.name):
            continue
        previous = next((p for i, p in reversed(ordered) if i < index), None)
        following = next((p for i, p in ordered if i > index), None)
        if following is None:
            continue
        if previous is not None and previous[0] == following[0]:
            missing = following[1] > previous[1] + 1
        else:
            missing = following[1] > 1
        if missing:
            in_gap.add(site.name)
    return in_gap


def decide(
    record: FigureRecord, *, in_gap: bool, mode: str, threshold: float
) -> None:
    """Apply the decision rule to an uncaptioned record, in place.

    | classifier says                                  | outcome            |
    | figure                                           | keep               |
    | decoration >= threshold, drop guard passes       | drop               |
    | anything else (low confidence, other label,      | keep, flagged      |
    |   malformed reply, API failure, no call)         |                    |
    """
    record.keep = True
    label, confidence = record.label, record.confidence

    if mode == MODE_OFF or label is None:
        if not any(f in record.flags for f in (FLAG_MALFORMED, FLAG_ERROR)):
            record.flags.append(FLAG_UNCLASSIFIED)
        return
    if label == FIGURE:
        return
    if label == DECORATION and confidence is not None and confidence >= threshold:
        if in_gap:
            record.flags.append(FLAG_GAP)
        elif mode == MODE_DRY_RUN:
            record.flags.append(FLAG_WOULD_DROP)
        else:
            record.keep = False
        return
    record.flags.append(FLAG_LOW_CONFIDENCE if label == DECORATION else FLAG_OTHER_LABEL)


# ── Building figures.json ───────────────────────────────────────────────────


@dataclass
class FiguresReport:
    records: dict[str, FigureRecord]
    order: list[str]  # filenames in reading order
    mode: str
    calls: int = 0
    stopped: str | None = None  # why classification stopped early

    def _ordered(self) -> list[tuple[str, FigureRecord]]:
        return [(name, self.records[name]) for name in self.order if name in self.records]

    @property
    def dropped(self) -> list[str]:
        return [n for n, r in self._ordered() if not r.keep]

    @property
    def flagged(self) -> list[str]:
        return [n for n, r in self._ordered() if r.keep and r.flags]

    @property
    def defaulted(self) -> list[str]:
        return [n for n, r in self._ordered() if r.by == BY_DEFAULT]

    @property
    def would_drop(self) -> list[str]:
        return [n for n, r in self._ordered() if FLAG_WOULD_DROP in r.flags]

    def label_counts(self) -> Counter[str]:
        """Classifier labels over the candidates (uncaptioned, not a panel)."""
        return Counter(
            r.label or "none"
            for _, r in self._ordered()
            if not r.number and not r.panel_of and r.by != BY_MANUAL
        )

    def render(self) -> str:
        records = [r for _, r in self._ordered()]
        by = Counter(r.by for r in records)
        lines = [
            f"{len(records)} image(s): {sum(1 for r in records if r.number)} numbered from "
            f"captions, {sum(1 for r in records if r.panel_of)} sub-panel(s), "
            f"{len(self.dropped)} dropped, {len(self.flagged)} flagged for review; "
            f"by {dict(sorted(by.items()))}; {self.calls} vision call(s)"
        ]
        if self.stopped:
            lines.append(f"  classification stopped early: {self.stopped}")
        if self.mode != MODE_OFF:
            counts = self.label_counts()
            lines.append(
                "  labels: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                if counts
                else "  labels: none"
            )
            drops = self.dropped if self.mode == MODE_CLASSIFY else self.would_drop
            heading = "dropped" if self.mode == MODE_CLASSIFY else "would drop (dry run)"
            lines.append(f"  {heading}: {len(drops)}")
            for name in drops:
                record = self.records[name]
                confidence = "" if record.confidence is None else f"{record.confidence:.2f}"
                lines.append(
                    f"    {name}  {record.label or record.by} {confidence}  {record.reason or ''}"
                )
        flags = Counter(f for r in records for f in r.flags)
        if flags:
            lines.append("  flags: " + ", ".join(f"{k} {v}" for k, v in sorted(flags.items())))
        return "\n".join(lines)


def build_figures(
    text: str,
    *,
    entries: list[dict] | None = None,
    images_dir: Path | None = None,
    existing: dict[str, FigureRecord] | None = None,
    vision: VisionLLM | None = None,
    mode: str = MODE_CLASSIFY,
    threshold: float = FIGURE_DROP_CONFIDENCE,
    min_interval: float = VISION_MIN_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    reclassify: bool = False,
) -> FiguresReport:
    """Decide a record for every image in ``text``.

    Args:
        text: The book's markdown, with real image paths (not tokens).
        entries: MinerU's content list for the book, if there is one.
        images_dir: Where the image files are, for the classifier.
        existing: The current ``figures.json``. Manual records are kept
            verbatim; a cached classifier label is reused instead of a call.
        vision: The classifier. None makes no calls (cached labels still
            count unless ``mode`` is off).
        mode: ``classify`` (drop confident decorations), ``dry-run`` (label,
            flag the would-be drops, drop nothing) or ``off`` (keep and flag
            every uncaptioned image).
        reclassify: Ignore cached labels and ask again -- after the prompt
            or the model changed. Manual records are still never touched.
    """
    existing = existing or {}
    entries = entries or []
    sites = scan_markdown(text)
    by_name = image_entries(entries)
    reference = reference_width(entries)
    if entries and reference is None:
        logger.warning("Content list has no long body-text blocks; widths fall back to default")

    records: dict[str, FigureRecord] = {}
    numbers: dict[str, str | None] = {}

    manual = {name: r for name, r in existing.items() if r.by == BY_MANUAL}

    # Pass 1: numbers and widths, which need no model.
    for site in sites:
        if site.name in manual:
            numbers[site.name] = manual[site.name].number
            continue
        entry = by_name.get(site.name)
        number = (caption_from_entry(entry) if entry else None) or site.number
        numbers[site.name] = number
        width = width_for(entry, reference)
        record = FigureRecord(number=number, width=width or DEFAULT_WIDTH)
        if width is None:
            record.flags.append(FLAG_NO_BBOX)
        records[site.name] = record

    # Sub-panels: a run of images joined only by panel labels shares the
    # caption under its last member -- or, when the parser put the caption
    # in the middle of the run, the caption just before its trailing panel.
    group: list[ImageSite] = []
    for site in sites:
        group.append(site)
        if site.joins_next:
            continue
        carriers = [numbers.get(s.name) or s.trails_caption for s in group]
        number = next((n for n in reversed(carriers) if n), None)
        if number:
            for member in group:
                if not numbers[member.name] and member.name in records:
                    records[member.name].panel_of = number
        group = []

    in_gap = numbering_gaps(sites, numbers)

    # Pass 2: classify what is left, reusing cached labels.
    calls = 0
    stopped: str | None = None
    last_call = 0.0
    for site in sites:
        record = records.get(site.name)
        if record is None or record.number or record.panel_of:
            continue
        cached = existing.get(site.name)
        if cached and cached.label in LABELS and not reclassify:
            record.label, record.confidence, record.reason = (
                cached.label,
                cached.confidence,
                cached.reason,
            )
        elif mode != MODE_OFF and vision is not None and stopped is None:
            image = _read_image(images_dir, site.name)
            if image is None:
                verdict = Verdict(None, None, "image file not found", FLAG_ERROR)
            else:
                wait = min_interval - (time.monotonic() - last_call)
                if calls and wait > 0:
                    sleep(wait)
                try:
                    verdict = classify_image(vision, image, site)
                except FatalClassifierError as exc:
                    stopped = str(exc)
                    logger.error("Vision classification stopped: %s", exc)
                    verdict = Verdict(None, None, f"not classified: {exc}"[:300], FLAG_ERROR)
                else:
                    calls += 1
                    last_call = time.monotonic()
                    if calls % 25 == 0:
                        logger.info("Classified %d image(s)", calls)
            record.label, record.confidence, record.reason = (
                verdict.label,
                verdict.confidence,
                verdict.reason,
            )
            if verdict.flag:
                record.flags.append(verdict.flag)
        decide(record, in_gap=site.name in in_gap, mode=mode, threshold=threshold)

    for record in records.values():
        record.by = settle_by(record)

    # Manual records win, including ones for images no longer in the text.
    records.update(manual)
    ordered = [s.name for s in sites]
    order = ordered + [n for n in records if n not in set(ordered)]
    records = {name: records[name] for name in order}
    return FiguresReport(records, order, mode, calls, stopped)


def figures_step(
    text: str, *, work_dir: Path, kit_dir: Path, mode: str, reclassify: bool = False
) -> tuple[FiguresReport, str | None]:
    """Build and save ``kit/figures.json`` for a book; the shared entry point
    of ``prepare_kit.py`` and ``prepare_figures.py``.

    Returns ``(report, note)``; ``note`` explains why no classifier was
    available, when one was wanted and could not be had.
    """
    vision: VisionLLM | None = None
    note: str | None = None
    if mode != MODE_OFF:
        from src.llm.factory import get_vision_llm

        try:
            vision = get_vision_llm(work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001 - fail open: keep everything
            note = f"no vision model available ({exc}); uncaptioned images are kept"

    images_dir = work_dir / "images"
    if not images_dir.is_dir():
        images_dir = kit_dir / "images"
    figures_path = kit_dir / FIGURES_NAME
    report = build_figures(
        text,
        entries=load_content_list(work_dir),
        images_dir=images_dir,
        existing=load_figures(figures_path),
        vision=vision,
        mode=mode,
        reclassify=reclassify,
    )
    save_figures(figures_path, report.records)
    return report, note


def _read_image(images_dir: Path | None, name: str) -> bytes | None:
    if images_dir is None:
        return None
    path = images_dir / name
    try:
        return path.read_bytes()
    except OSError:
        return None


def strip_dropped_images(text: str, records: dict[str, FigureRecord]) -> tuple[str, int]:
    """Remove dropped images from the markdown, before it is tokenized.

    The translator never sees them. A line left empty by the removal goes
    with it.
    """
    dropped = {name for name, record in records.items() if not record.keep}
    if not dropped:
        return text, 0
    removed = 0

    def drop(match: re.Match[str]) -> str:
        nonlocal removed
        if Path(match.group(1)).name in dropped:
            removed += 1
            return ""
        return match.group(0)

    out: list[str] = []
    for line in text.split("\n"):
        if not _image_refs(line):
            out.append(line)
            continue
        new = _HTML_IMAGE_RE.sub(drop, _MD_IMAGE_RE.sub(drop, line))
        if new == line or new.strip():
            out.append(new)
    return "\n".join(out), removed


# ── Linking a re-parse to an existing book ──────────────────────────────────
#
# A book parsed before content lists were kept has no layout data. It is
# re-parsed into a separate work directory, and each of its images is matched
# to an image of the re-parse, whose bbox then sizes it. A different backend
# may crop differently, so filenames can change: an exact name is tried
# first, then a perceptual hash near the same reading-order position.

METHOD_EXACT = "exact"
METHOD_DHASH = "dhash"
# Out of 64 bits. Two crops of the same figure land within a few bits; two
# different line drawings almost never come within ten.
DHASH_MAX_DISTANCE = 10


def dhash(path: Path) -> int | None:
    """64-bit difference hash: grayscale, 9x8, one bit per horizontal step.

    Built on ``pymupdf.Pixmap``, already a dependency. None if unreadable.
    """
    import pymupdf

    try:
        pixmap = pymupdf.Pixmap(str(path))
        if pixmap.alpha:
            pixmap = pymupdf.Pixmap(pixmap, 0)
        if pixmap.n != 1:
            pixmap = pymupdf.Pixmap(pymupdf.csGRAY, pixmap)
        small = pymupdf.Pixmap(pixmap, 9, 8, None)
    except Exception:  # noqa: BLE001 - any unreadable image is just unmatched
        return None
    samples = small.samples
    stride = small.stride
    value = 0
    for row in range(8):
        for col in range(8):
            left = samples[row * stride + col]
            right = samples[row * stride + col + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


@dataclass
class LinkResult:
    matches: dict[str, tuple[str, str]]  # old name -> (method, new name)
    unmatched: list[str]

    def counts(self) -> Counter[str]:
        counts = Counter(method for method, _ in self.matches.values())
        counts["unmatched"] = len(self.unmatched)
        return counts


def link_images(
    old_names: list[str],
    old_images_dir: Path,
    new_entries: list[dict],
    new_images_dir: Path,
    *,
    window: int | None = None,
    max_distance: int = DHASH_MAX_DISTANCE,
) -> LinkResult:
    """Match each old image (in reading order) to a re-parse image entry.

    Exact filename first. Otherwise the closest difference hash among the
    not-yet-matched new images within ``window`` positions of where the old
    image's reading-order position projects onto the new list -- so a
    generic-looking diagram cannot be matched to its twin three chapters on.
    """
    new_names = list(image_entries(new_entries))
    new_set = set(new_names)
    matches: dict[str, tuple[str, str]] = {}
    taken: set[str] = set()

    for name in old_names:
        if name in new_set:
            matches[name] = (METHOD_EXACT, name)
            taken.add(name)

    if window is None:
        window = max(10, len(new_names) // 20)
    hashes: dict[str, int | None] = {}

    def hash_of(directory: Path, name: str) -> int | None:
        key = f"{directory}/{name}"
        if key not in hashes:
            hashes[key] = dhash(directory / name)
        return hashes[key]

    scale = (len(new_names) - 1) / (len(old_names) - 1) if len(old_names) > 1 else 0.0
    unmatched: list[str] = []
    for index, name in enumerate(old_names):
        if name in matches:
            continue
        old_hash = hash_of(old_images_dir, name)
        if old_hash is None or not new_names:
            unmatched.append(name)
            continue
        centre = round(index * scale)
        best: tuple[int, str] | None = None
        for candidate in new_names[max(0, centre - window) : centre + window + 1]:
            if candidate in taken:
                continue
            new_hash = hash_of(new_images_dir, candidate)
            if new_hash is None:
                continue
            distance = (old_hash ^ new_hash).bit_count()
            if distance <= max_distance and (best is None or distance < best[0]):
                best = (distance, candidate)
        if best is None:
            unmatched.append(name)
        else:
            matches[name] = (METHOD_DHASH, best[1])
            taken.add(best[1])
    return LinkResult(matches, unmatched)


def linked_entries(new_entries: list[dict], result: LinkResult) -> list[dict]:
    """The re-parse's layout, with matched images renamed to the old names.

    Text blocks are kept as they are: they define the reference width.
    Unmatched new images are left out.
    """
    renamed = {new: old for old, (_, new) in result.matches.items()}
    out: list[dict] = []
    for entry in new_entries:
        if not entry.get("img_path"):
            out.append(entry)
            continue
        name = Path(str(entry["img_path"])).name
        if name in renamed:
            out.append({**entry, "img_path": f"images/{renamed[name]}"})
    return out


def apply_linked_widths(records: dict[str, FigureRecord], entries: list[dict]) -> int:
    """Re-size every non-manual record from ``entries``; returns how many got a
    measured width. The rest fall back to the default and are flagged."""
    reference = reference_width(entries)
    by_name = image_entries(entries)
    measured = 0
    for name, record in records.items():
        if record.by == BY_MANUAL:
            continue
        width = width_for(by_name.get(name), reference)
        record.flags = [f for f in record.flags if f != FLAG_NO_BBOX]
        if width is None:
            record.width = DEFAULT_WIDTH
            record.flags.insert(0, FLAG_NO_BBOX)
        else:
            record.width = width
            measured += 1
        record.by = settle_by(record)
    return measured


# ── Review sheet ────────────────────────────────────────────────────────────

_REVIEW_STYLE = """
body{font:14px/1.4 system-ui,sans-serif;margin:1.5em;color:#222}
h1{font-size:1.3em}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f4f4f4;position:sticky;top:0}
img{max-width:220px;max-height:160px;border:1px solid #ccc;background:#fff}
.dropped{background:#fde8e8}.flagged{background:#fff6dc}.default{background:#eef3fb}
code{font-size:12px;word-break:break-all}.d{font-weight:bold}
"""


def write_review(
    kit_dir: Path,
    records: dict[str, FigureRecord],
    image_map: dict[str, str] | None = None,
    order: list[str] | None = None,
) -> Path:
    """Write ``kit/figures_review.html``: every dropped, flagged or defaulted
    image, dropped first, as thumbnails with the reason. Plain HTML, no
    external resources; thumbnails are relative paths into ``images/``."""
    tokens = {Path(path).name: token for token, path in (image_map or {}).items()}
    names = list(order or records)
    names += [n for n in records if n not in names]

    def group(record: FigureRecord) -> str | None:
        if not record.keep:
            return "dropped"
        if record.flags:
            return "flagged"
        if record.by == BY_DEFAULT:
            return "default"
        return None

    rows: dict[str, list[str]] = {"dropped": [], "flagged": [], "default": []}
    for name in names:
        record = records.get(name)
        if record is None:
            continue
        kind = group(record)
        if kind is None:
            continue
        confidence = "" if record.confidence is None else f"{record.confidence:.2f}"
        decision = "DROP" if not record.keep else "keep"
        rows[kind].append(
            f'<tr class="{kind}"><td><a href="images/{html.escape(name)}">'
            f'<img src="images/{html.escape(name)}" loading="lazy" alt=""></a></td>'
            f"<td><code>{html.escape(name)}</code><br>{html.escape(tokens.get(name, '-'))}</td>"
            f'<td class="d">{decision}</td>'
            f"<td>{html.escape(record.number or record.panel_of and f'panel of {record.panel_of}' or '')}</td>"
            f"<td>{html.escape(record.label or '')} {confidence}</td>"
            f"<td>{html.escape(record.reason or '')}</td>"
            f"<td>{html.escape(', '.join(record.flags))}</td>"
            f"<td>{html.escape(record.by)}, width {record.width:.2f}</td></tr>"
        )

    counts = ", ".join(f"{len(v)} {k}" for k, v in rows.items())
    body = "\n".join(rows["dropped"] + rows["flagged"] + rows["default"])
    document = (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>Figure review</title><style>{_REVIEW_STYLE}</style></head><body>\n"
        f"<h1>Figure review &mdash; {counts}</h1>\n"
        "<p>Red: dropped. Yellow: kept, needs a look. Blue: kept with a default "
        f"width. To overrule a decision, edit the image's entry in {FIGURES_NAME}, "
        'set <code>"by": "manual"</code>, and rebuild.</p>\n'
        "<table><tr><th>image</th><th>file / token</th><th>decision</th><th>number</th>"
        "<th>label</th><th>reason</th><th>flags</th><th>by</th></tr>\n"
        f"{body}\n</table></body></html>\n"
    )
    path = kit_dir / REVIEW_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "BY_CAPTION",
    "BY_CLASSIFIER",
    "BY_DEFAULT",
    "BY_MANUAL",
    "CONTENT_LIST_NAME",
    "FIGURES_NAME",
    "LABELS",
    "LINKED_CONTENT_LIST_NAME",
    "MODE_CLASSIFY",
    "MODE_DRY_RUN",
    "MODE_OFF",
    "REVIEW_NAME",
    "FigureRecord",
    "FiguresReport",
    "ImageSite",
    "LinkResult",
    "apply_linked_widths",
    "build_figures",
    "caption_number",
    "classify_image",
    "decide",
    "dhash",
    "figures_step",
    "image_entries",
    "link_images",
    "linked_entries",
    "load_content_list",
    "load_figures",
    "numbering_gaps",
    "reference_width",
    "save_figures",
    "scan_markdown",
    "settle_by",
    "split_number",
    "strip_dropped_images",
    "width_for",
    "write_review",
]
