"""
Put back the footnotes MinerU leaves out of ``full.md``.

The VLM backend tags footnotes ``page_footnote`` in the content list, and
``full.md`` -- which ``merged.md`` is concatenated from -- drops that type
along with running heads and page numbers. On physics_2 that lost 109 blocks
of three kinds:

* real footnotes, whose ``①`` markers survived in the body text;
* body text the VLM mis-tagged as a footnote (the statements of Examples
  13.3 and 30.2), which simply vanished;
* long footnote derivations, whose display equations were typed
  ``equation`` and so stayed behind, stranded in the body.

``restore_footnotes`` rebuilds them from ``content_list.json``, page by page,
by the same logic the physics_2 run used by hand:

1. **The footnote region** of a page is every entry at or below the top of
   its first ``page_footnote`` (running heads, page numbers and margin notes
   excluded).
2. **It is grouped at circled markers.** A ``page_footnote`` block (or a
   line inside one) starting with ``①`` opens a footnote; later
   ``page_footnote`` text and display equations join it, until the next
   marker or a block of another type (text, figure ...), which stays body.
3. **A marked footnote** goes in as its own paragraph, ``① text``, right
   after the paragraph holding the marker. The translation prompt
   (``assets/system_prompt_latex.txt``) already turns such a line into
   ``\\footnote{}`` at the marker. Its equations are taken out of the body.
4. **Unmarked footnote text** (the region before the page's first marker) is
   treated as mis-tagged body text and goes in after the body block above
   it. A marked footnote whose marker cannot be found is placed the same
   way, still as a ``①`` line.

Entries are matched to ``merged.md`` paragraphs by their text, in reading
order; nothing depends on the markdown carrying page breaks. Text that is
already present is skipped, so running this twice -- or on a book restored
by hand -- changes nothing.
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

FOOTNOTE_TYPE = "page_footnote"
EQUATION_TYPE = "equation"

# Never part of the footnote region: page furniture, which full.md drops too.
# (A page number usually sits below the footnotes.)
_FURNITURE_TYPES = frozenset({"header", "footer", "page_number", "aside_text"})

# Matched to merged.md paragraphs, and anchors for insertion. Everything not
# dropped by full.md: furniture and footnotes are not looked for.
_UNMATCHED_TYPES = _FURNITURE_TYPES | {FOOTNOTE_TYPE}

# ①..⑳, or \textcircled{1} as the VLM sometimes writes it.
_CIRCLED_FIRST = 0x2460
_MARK = r"(?:([\u2460-\u2473])|\\textcircled\s*\{\s*(\d{1,2})\s*\})"
# A marker at the start of a footnote, bare (`①`) or in math (`$^{①}$`,
# `${ } ^ { ① }$`).
_LEAD_RE = re.compile(rf"\s*(?:\$[\s{{}}^]*{_MARK}[\s{{}}]*\$|\^?\{{?{_MARK}\}}?)\s*")
# Where one footnote block holds several: a marker at the start of the block,
# of a line, or of a sentence. A circled number mid-sentence ("由①式得")
# refers to an equation and must not split.
_SPLIT_RE = re.compile(rf"(?:^|(?<=[\n。]))[ \t]*(?=\$[\s{{}}^]*{_MARK}|\^?\{{?{_MARK})")

_PARAGRAPH_SEP_RE = re.compile(r"(\n[ \t]*\n\s*)")
_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n\s*")
_WS_RE = re.compile(r"\s+")
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}")

# A containment match (entry text inside a longer paragraph) needs this many
# characters, and an exact match this many to jump arbitrarily far ahead (a
# part whose content list was missing leaves paragraphs nothing matches).
_MIN_CONTAINED_CHARS = 10
_MIN_FAR_MATCH_CHARS = 20
_LOOKAHEAD_PARAGRAPHS = 40
# Shorter text counts as "already present" only as a whole paragraph. The
# missing-block report, which changes nothing, accepts shorter text.
_MIN_PRESENT_CHARS = 12
_MIN_MISSING_CHECK_CHARS = 6


@dataclass
class FootnoteReport:
    """What ``restore_footnotes`` did. Counts are pieces (one footnote, or
    one run of unmarked text), not content-list blocks, except
    ``footnote_blocks``."""

    footnote_blocks: int = 0
    at_marker: int = 0
    marker_not_found: int = 0
    unmarked: int = 0
    moved_into_footnotes: int = 0
    already_present: int = 0
    # Pages whose marker could not be found, for the log.
    marker_not_found_pages: list[int] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.at_marker or self.marker_not_found or self.unmarked)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.footnote_blocks} page_footnote block(s): {self.at_marker} footnote(s) placed "
            f"at their marker, {self.marker_not_found} without one, {self.unmarked} unmarked "
            f"text run(s) restored as body, {self.moved_into_footnotes} stranded block(s) moved "
            f"into footnotes, {self.already_present} already present"
        )


@dataclass
class _Insert:
    text: str
    footnote: bool


@dataclass
class _Group:
    number: int
    page: int
    pieces: list[str]
    stranded: list[int]  # merged.md paragraph indices moved into this footnote


def normalize(text: str) -> str:
    """Whitespace-free, with heading hashes gone: how entries and paragraphs
    are compared."""
    return _WS_RE.sub("", _HEADING_PREFIX_RE.sub("", text.strip()))


def split_markers(text: str) -> list[tuple[int | None, str]]:
    """Split footnote text at circled markers into ``(number, body)`` pieces;
    text before the first marker comes back with number ``None``."""
    starts = [m.start() for m in _SPLIT_RE.finditer(text)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    pieces: list[tuple[int | None, str]] = []
    for begin, end in zip(starts, starts[1:] + [len(text)]):
        chunk = text[begin:end]
        lead = _LEAD_RE.match(chunk)
        number = _marker_number(lead) if lead else None
        body = chunk[lead.end():] if lead and number else chunk
        body = body.strip()
        if body or number:
            pieces.append((number, body))
    return pieces


def _marker_number(match: re.Match[str]) -> int | None:
    for group in match.groups():
        if not group:
            continue
        if len(group) == 1 and _CIRCLED_FIRST <= ord(group) < _CIRCLED_FIRST + 20:
            return ord(group) - _CIRCLED_FIRST + 1
        # \textcircled{n}. (Not str.isdigit, which is true for ① itself.)
        return int(group) if 1 <= int(group) <= 20 else None
    return None


def circled(number: int) -> str:
    return chr(_CIRCLED_FIRST + number - 1)


def _has_marker(text: str, number: int) -> bool:
    return circled(number) in text or bool(
        re.search(rf"\\textcircled\s*\{{\s*{number}\s*\}}", text)
    )


def _entry_key(entry: dict) -> str:
    """The text an entry leaves in full.md, normalized; "" when unknown."""
    kind = entry.get("type")
    if kind == "list" and isinstance(entry.get("list_items"), list):
        return normalize("".join(str(i) for i in entry["list_items"]))
    if entry.get("img_path"):
        # The image link is the one stable part of a figure's paragraph.
        return str(entry["img_path"]).rsplit("/", 1)[-1]
    for name in ("text", "table_body", "code_body"):
        value = entry.get(name)
        if isinstance(value, str) and value.strip():
            return normalize(value)
    return ""


def _top(entry: dict) -> float | None:
    bbox = entry.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return float(bbox[1])
        except (TypeError, ValueError):
            return None
    return None


class _Document:
    """merged.md as paragraphs, with each content entry mapped to the
    paragraph holding its text."""

    def __init__(self, markdown: str, entries: list[dict]) -> None:
        body = markdown.rstrip()
        self.tail = markdown[len(body):]
        split = _PARAGRAPH_SEP_RE.split(body)
        self.paras = split[0::2]
        self.seps = split[1::2] + [""]
        self.norm = [normalize(p) for p in self.paras]
        self.norm_set = set(self.norm)
        self.norm_all = "".join(self.norm)
        self.para_of: dict[int, int] = {}
        self._match(entries)

    def _match(self, entries: list[dict]) -> None:
        by_text: dict[str, list[int]] = {}
        for index, text in enumerate(self.norm):
            by_text.setdefault(text, []).append(index)

        cursor = 0  # the paragraph of the last matched entry
        started = False
        for k, entry in enumerate(entries):
            if entry.get("type") in _UNMATCHED_TYPES:
                continue
            key = _entry_key(entry)
            if len(key) < 2:
                continue
            found = None
            # An exact paragraph, after the last one (two identical entries
            # are two paragraphs).
            hits = by_text.get(key, [])
            pos = bisect.bisect_left(hits, cursor + 1 if started else 0)
            if pos < len(hits) and (
                len(key) >= _MIN_FAR_MATCH_CHARS or hits[pos] - cursor <= _LOOKAHEAD_PARAGRAPHS
            ):
                found = hits[pos]
            elif len(key) >= _MIN_CONTAINED_CHARS or entry.get("img_path"):
                # Part of a paragraph (a heading, a paragraph merged across a
                # page break), which may be the last one again.
                stop = min(len(self.norm), cursor + _LOOKAHEAD_PARAGRAPHS)
                found = next((j for j in range(cursor, stop) if key in self.norm[j]), None)
            if found is not None:
                self.para_of[k] = found
                cursor, started = found, True

    def present(self, text: str, min_chars: int = _MIN_PRESENT_CHARS) -> bool:
        key = normalize(text)
        return key in self.norm_set or (len(key) >= min_chars and key in self.norm_all)


def restore_footnotes(markdown: str, entries: list[dict]) -> tuple[str, FootnoteReport]:
    """Return ``markdown`` with the ``page_footnote`` content of ``entries``
    (a book's content list, global ``page_idx``, reading order) put back,
    and a report. The markdown is returned unchanged when there is nothing
    to restore."""
    report = FootnoteReport(
        footnote_blocks=sum(1 for e in entries if e.get("type") == FOOTNOTE_TYPE)
    )
    if not report.footnote_blocks:
        return markdown, report

    doc = _Document(markdown, entries)
    inserts: dict[int, list[_Insert]] = {}  # after paragraph j (-1: before the first)
    removed: set[int] = set()

    # Entry index -> the last paragraph matched at or before it.
    last_para: list[int] = []
    current = -1
    for k in range(len(entries)):
        current = doc.para_of.get(k, current)
        last_para.append(current)

    pages: dict[int, list[int]] = {}
    for k, entry in enumerate(entries):
        if isinstance(entry.get("page_idx"), int):
            pages.setdefault(entry["page_idx"], []).append(k)

    for page, indices in pages.items():
        footnote_tops = [
            _top(entries[k]) for k in indices if entries[k].get("type") == FOOTNOTE_TYPE
        ]
        if not footnote_tops:
            continue
        known = [t for t in footnote_tops if t is not None]
        region_top = min(known) if known else None
        region = [
            k
            for k in indices
            if entries[k].get("type") not in _FURNITURE_TYPES
            and (
                entries[k].get("type") == FOOTNOTE_TYPE
                or (region_top is not None and (_top(entries[k]) or -1) >= region_top)
            )
        ]

        groups: list[_Group] = []
        group: _Group | None = None
        for k in region:
            entry = entries[k]
            kind = entry.get("type")
            if kind == FOOTNOTE_TYPE:
                for number, body in split_markers(str(entry.get("text") or "")):
                    if number is not None:
                        group = _Group(number, page, [body] if body else [], [])
                        groups.append(group)
                    elif group is not None:
                        group.pieces.append(body)
                    elif doc.present(body):
                        report.already_present += 1
                    else:
                        _append(inserts, last_para[k], _Insert(body, footnote=False))
                        report.unmarked += 1
            elif kind == EQUATION_TYPE and group is not None:
                group.pieces.append(str(entry.get("text") or "").strip())
                if k in doc.para_of and doc.norm[doc.para_of[k]] == _entry_key(entry):
                    group.stranded.append(doc.para_of[k])
            else:
                group = None  # body text below the footnotes: it stays

        page_paras = sorted(
            {doc.para_of[k] for k in indices if k in doc.para_of}
            - {j for g in groups for j in g.stranded}
        )
        before = last_para[indices[0] - 1] if indices[0] > 0 else -1
        for g in groups:
            text = _BLANK_LINES_RE.sub("\n", "\n".join(p for p in g.pieces if p)).strip()
            line = f"{circled(g.number)} {text}".rstrip()
            if doc.present(line):
                report.already_present += 1
                continue
            removed.update(g.stranded)
            report.moved_into_footnotes += len(g.stranded)
            # The marker is on this page, or at the end of a paragraph that
            # began on the previous one.
            candidates = ([before] if before >= 0 else []) + page_paras
            if _place_at_marker(doc, inserts, candidates, g.number, line):
                report.at_marker += 1
            else:
                _append(inserts, last_para[region[0]], _Insert(line, footnote=True))
                report.marker_not_found += 1
                report.marker_not_found_pages.append(page)

    if not report.changed:
        return markdown, report
    if report.marker_not_found:
        logger.warning(
            "%d footnote marker(s) not found; their text went after the body above, on "
            "content-list page(s) %s",
            report.marker_not_found,
            sorted(set(report.marker_not_found_pages)),
        )
    return _render(doc, inserts, removed), report


def _append(inserts: dict[int, list[_Insert]], after: int, item: _Insert) -> None:
    inserts.setdefault(after, []).append(item)


def _place_at_marker(
    doc: _Document,
    inserts: dict[int, list[_Insert]],
    candidates: list[int],
    number: int,
    line: str,
) -> bool:
    """Insert ``line`` right after the first paragraph (or restored body text)
    holding marker ``number``, after any footnotes already placed there."""
    for j in candidates:
        if j in inserts or _has_marker(doc.paras[j], number):
            items = inserts.setdefault(j, [])
            if _has_marker(doc.paras[j], number):
                at = 0
            else:
                at = next(
                    (i + 1 for i, item in enumerate(items)
                     if not item.footnote and _has_marker(item.text, number)),
                    None,
                )
                if at is None:
                    continue
            while at < len(items) and items[at].footnote:
                at += 1
            items.insert(at, _Insert(line, footnote=True))
            return True
    return False


def _render(doc: _Document, inserts: dict[int, list[_Insert]], removed: set[int]) -> str:
    out: list[tuple[str, str]] = [(item.text, "\n\n") for item in inserts.get(-1, [])]
    for j, para in enumerate(doc.paras):
        if j not in removed:
            out.append((para, doc.seps[j] or "\n\n"))
        out.extend((item.text, "\n\n") for item in inserts.get(j, []))
    if not out:
        return doc.tail
    # Each paragraph keeps the separator that followed it; the last one has none.
    text = "".join(p + s for p, s in out[:-1]) + out[-1][0]
    return text + (doc.tail or "\n")


def missing_footnote_blocks(markdown: str, entries: list[dict]) -> list[dict]:
    """The ``page_footnote`` entries none of whose text is in ``markdown``:
    what a parse lost, for a report."""
    doc = _Document(markdown, [])
    lost = []
    for entry in entries:
        if entry.get("type") != FOOTNOTE_TYPE:
            continue
        pieces = split_markers(str(entry.get("text") or ""))
        # A marker makes even short text distinctive; restored footnote
        # lines carry theirs.
        if not any(
            doc.present(circled(number) + body, min_chars=2)
            if number
            else body and doc.present(body, min_chars=_MIN_MISSING_CHECK_CHARS)
            for number, body in pieces
        ):
            lost.append(entry)
    return lost


__all__ = [
    "FootnoteReport",
    "missing_footnote_blocks",
    "normalize",
    "restore_footnotes",
    "split_markers",
]
