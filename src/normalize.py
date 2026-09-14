"""
Deterministic repairs applied to ``merged.md`` before chunking.

Scope
-----
Almost nothing needs repairing any more. Measured across the whole book, the
cloud VLM parse has **zero** whitespace-inside-``$`` spans (of 6,641 inline
spans), zero space-mangled LaTeX, zero escaped ``\\$`` closing delimiters,
zero welded ``$$$$``, and even ``$$`` parity. Writing normalizer passes for
those would be writing fixes for defects that no longer occur.

What is left is one genuine repair -- **split headings** -- plus a handful of
cheap diagnostics kept as insurance against a future parser change
reintroducing something. Every function here is idempotent and reports a
count.

Split headings: the two verified patterns
------------------------------------------
Measured on the test book's 442 headings, only two shapes are safe to merge:

1. ``## 第13章`` immediately followed by ``## 电势`` -- a chapter number torn
   off its title. **5 occurrences.**
2. A run of bare fragments forming a part title: ``## 第`` / ``## 篇`` /
   ``## 电磁学``. **5 bare 第/篇 fragments** across 3 runs.

Everything else is left alone, deliberately. ``提要`` (Summary) and ``习题``
(Exercises) are two characters long and appear **19 times each** as
legitimate headings; ``目录`` sits immediately before ``CONTENTS`` and
``提要`` immediately before ``1. 相干光``. A "merge short adjacent headings"
rule would corrupt all of those. Short headings that are not merged are
*reported*, never guessed at.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)[ \t]*$")

# Pandoc's inline-math rule: an opening `$` must be followed by a non-space
# and a closing `$` preceded by one. `(?<!\$)`/`(?!\$)` keep this from
# matching the halves of a `$$` display delimiter.
INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")

_DISPLAY_ONE_LINE_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# CJK and ASCII digits, plus the CJK numerals a Chinese textbook numbers
# chapters and parts with.
_NUM = r"[0-9０-９一二三四五六七八九十百]"

# `第13章`, `第 13 章`, `第十三章` -- a chapter number with no title attached.
_CHAPTER_NUMBER_RE = re.compile(rf"^第\s*{_NUM}+\s*章$")

# A bare part-title fragment: `第`, `第3`, `篇`, `3篇`. Never matches `提要`,
# `习题`, `目录` or a real chapter title -- those have no 第/篇 at all.
_PART_FRAGMENT_RE = re.compile(rf"^(?:第\s*{_NUM}*|{_NUM}*\s*篇)$")

# A numbered section (`12.1 电荷`) or subsection (`1. 电荷的种类`) is a
# heading in its own right and must never be swallowed into a chapter title.
_NUMBERED_RE = re.compile(rf"^{_NUM}+\s*[.．]")

# A title long enough to be a paragraph is not a torn-off heading.
_MAX_TITLE_CHARS = 40

# Short headings are worth eyeballing, but the vast majority are legitimate.
_SHORT_HEADING_CHARS = 2


@dataclass(frozen=True)
class Change:
    """One applied repair, for the run report."""

    kind: str
    line: int  # 1-based line number in the INPUT text
    before: str
    after: str


@dataclass(frozen=True)
class Note:
    """Something worth a human's eyes that was deliberately left untouched."""

    kind: str
    line: int
    text: str
    detail: str = ""


@dataclass
class NormalizeResult:
    text: str
    changes: list[Change] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def counts(self) -> Counter[str]:
        return Counter(change.kind for change in self.changes)

    def summary(self) -> str:
        counts = self.counts()
        applied = ", ".join(f"{kind}: {n}" for kind, n in sorted(counts.items())) or "none"
        return f"repairs applied ({applied}); {len(self.notes)} note(s) for review"


# ── Math span helpers (shared with src.lint) ────────────────────────────────


@dataclass(frozen=True)
class MathSpan:
    """One math span with the 1-based line it starts on."""

    line: int
    kind: str  # "inline" or "display"
    body: str


def iter_math_spans(text: str) -> Iterator[MathSpan]:
    """Yield every inline and display math span in ``text``.

    Uses a line-by-line display-math state machine rather than one big
    regex. An earlier regex-only version of this logic corrupted the file it
    was run on: with a stray unpaired ``$$`` anywhere in an 840,000-character
    book, a `re.DOTALL` display pattern happily swallows several pages of
    prose as "math". Tracking the state per line cannot do that.
    """
    in_display = False
    start_line = 0
    buffer: list[str] = []

    for lineno, line in enumerate(text.split("\n"), start=1):
        if line.strip() == "$$":
            if in_display:
                yield MathSpan(start_line, "display", "\n".join(buffer))
                buffer = []
                in_display = False
            else:
                in_display = True
                start_line = lineno
            continue

        if in_display:
            buffer.append(line)
            continue

        # Single-line `$$...$$` first, so its contents aren't also read as
        # two inline spans.
        for match in _DISPLAY_ONE_LINE_RE.finditer(line):
            yield MathSpan(lineno, "display", match.group(1))
        remainder = _DISPLAY_ONE_LINE_RE.sub(" ", line)

        for match in INLINE_MATH_RE.finditer(remainder):
            yield MathSpan(lineno, "inline", match.group(1))

    if in_display:
        # An unterminated block is itself a defect; surface what we have so
        # the caller's parity check reports it rather than silently dropping.
        yield MathSpan(start_line, "display", "\n".join(buffer))


def dollar_run_histogram(text: str) -> dict[int, int]:
    """Count runs of consecutive ``$`` by length.

    A healthy document has only length-1 (inline) and length-2 (display)
    runs. A length-4 run is the welded ``$$$$`` defect the old parser
    produced; anything longer is worse.
    """
    histogram = Counter(len(match.group(0)) for match in re.finditer(r"\$+", text))
    return dict(sorted(histogram.items()))


def display_parity(text: str) -> tuple[int, bool]:
    """Return ``(count of "$$" delimiters, whether that count is even)``.

    An odd count means a display block is never closed, which makes pandoc
    read the entire rest of the document as mathematics.
    """
    count = text.count("$$")
    return count, count % 2 == 0


def find_delimiter_whitespace(text: str) -> list[tuple[int, str]]:
    """Find inline spans padded with whitespace inside their ``$`` delimiters.

    Pandoc does not read ``$n S { \\mathrm { d } } l $`` as math at all: it
    escapes the ``$`` and the LaTeX leaks into text mode, giving
    ``! LaTeX Error: \\mathrm allowed only in math mode``.

    Zero of the cloud VLM parse's 6,641 inline spans hit this, so it is a
    diagnostic now, not a repair. See ``fix_inline_math_spacing``.
    """
    return [
        (span.line, span.body)
        for span in iter_math_spans(text)
        if span.kind == "inline" and span.body != span.body.strip()
    ]


def fix_inline_math_spacing(text: str) -> tuple[str, int]:
    """Trim whitespace immediately inside inline ``$`` delimiters.

    **Not run by default** -- the current parser produces none of these. Kept
    available behind ``normalize(..., fix_math_spacing=True)`` in case a
    future parser change reintroduces the defect.

    Display math is left completely alone, and the mathematical content is
    unchanged: LaTeX ignores this whitespace when typesetting.
    """
    changed = 0

    def tidy(match: re.Match[str]) -> str:
        nonlocal changed
        inner = match.group(1)
        stripped = inner.strip()
        if stripped != inner:
            changed += 1
        return f"${stripped}$"

    in_display = False
    out: list[str] = []
    for line in text.split("\n"):
        if line.strip() == "$$":
            in_display = not in_display
            out.append(line)
        elif in_display:
            out.append(line)
        else:
            out.append(INLINE_MATH_RE.sub(tidy, line))

    return "\n".join(out), changed


# ── Split-heading repair ────────────────────────────────────────────────────


def _is_mergeable_title(title: str) -> bool:
    """Is ``title`` a plausible torn-off title rather than a heading of its own?"""
    return (
        len(title) <= _MAX_TITLE_CHARS
        and not _CHAPTER_NUMBER_RE.match(title)
        and not _PART_FRAGMENT_RE.match(title)
        and not _NUMBERED_RE.match(title)
    )


def _next_heading(lines: list[str], start: int) -> tuple[int, re.Match[str]] | None:
    """The next heading at or after ``start``, if only blank lines precede it."""
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return None
    match = HEADING_RE.match(lines[index])
    return (index, match) if match else None


def merge_split_headings(text: str) -> tuple[str, list[Change], list[Note]]:
    """Repair the two verified split-heading patterns. Idempotent.

    A merged heading takes the *shallowest* level of its parts: the pieces
    are one heading, so the most prominent level either half was given is the
    one that survives. (Heading levels are re-assigned downstream anyway by
    ``src.profiler``'s classification; this only has to not lose information.)
    """
    lines = text.split("\n")
    out: list[str] = []
    changes: list[Change] = []
    notes: list[Note] = []
    merged_lines: set[int] = set()

    index = 0
    while index < len(lines):
        match = HEADING_RE.match(lines[index])
        if not match:
            out.append(lines[index])
            index += 1
            continue

        hashes, title = match.group(1), match.group(2)

        # ── Pattern 2: a run of bare 第/篇 fragments, then the real title ──
        if _PART_FRAGMENT_RE.match(title):
            fragments = [title]
            levels = [len(hashes)]
            cursor = index + 1
            following = _next_heading(lines, cursor)
            while following is not None and _PART_FRAGMENT_RE.match(following[1].group(2)):
                fragments.append(following[1].group(2))
                levels.append(len(following[1].group(1)))
                cursor = following[0] + 1
                following = _next_heading(lines, cursor)

            if following is not None and _is_mergeable_title(following[1].group(2)):
                part_title = following[1].group(2)
                levels.append(len(following[1].group(1)))
                merged = "".join(f.replace(" ", "") for f in fragments) + " " + part_title
                line = "#" * min(levels) + " " + merged
                out.append(line)
                changes.append(
                    Change("part-title-fragments", index + 1, " / ".join(fragments), merged)
                )
                if not re.search(_NUM, merged):
                    notes.append(
                        Note(
                            "part-ordinal-missing",
                            index + 1,
                            merged,
                            "the part number was not in the parsed text; insert it by hand",
                        )
                    )
                merged_lines.update(range(index, following[0] + 1))
                index = following[0] + 1
                continue

            notes.append(
                Note(
                    "unmerged-fragment",
                    index + 1,
                    title,
                    "looks like a part-title fragment but is not followed by a title heading",
                )
            )
            out.append(lines[index])
            index += 1
            continue

        # ── Pattern 1: `第N章` immediately followed by its title heading ──
        if _CHAPTER_NUMBER_RE.match(title):
            following = _next_heading(lines, index + 1)
            if following is not None and _is_mergeable_title(following[1].group(2)):
                chapter_title = following[1].group(2)
                merged = f"{title} {chapter_title}"
                level = min(len(hashes), len(following[1].group(1)))
                out.append("#" * level + " " + merged)
                changes.append(Change("chapter-title-split", index + 1, title, merged))
                merged_lines.update(range(index, following[0] + 1))
                index = following[0] + 1
                continue

        out.append(lines[index])
        index += 1

    # Report every remaining short heading so a genuinely new fragment shape
    # is visible without this function ever guessing at one.
    for lineno, line in enumerate(lines, start=1):
        if lineno - 1 in merged_lines:
            continue
        match = HEADING_RE.match(line)
        if match and len(match.group(2)) <= _SHORT_HEADING_CHARS:
            notes.append(
                Note("short-heading", lineno, match.group(2), "left unchanged; verify by eye")
            )

    return "\n".join(out), changes, notes


# ── Entry point ─────────────────────────────────────────────────────────────


def normalize(text: str, *, fix_math_spacing: bool = False) -> NormalizeResult:
    """Run every deterministic repair over ``text``.

    Args:
        text: The book's ``merged.md`` content.
        fix_math_spacing: Also trim whitespace inside inline ``$``
            delimiters. Off by default: the current parser produces none of
            those, and a repair pass for a defect that does not occur is a
            liability, not insurance.

    Returns:
        A ``NormalizeResult`` with the repaired text, the applied changes,
        the notes needing human eyes, and the cheap diagnostics.
    """
    result_text, changes, notes = merge_split_headings(text)

    if fix_math_spacing:
        result_text, trimmed = fix_inline_math_spacing(result_text)
        if trimmed:
            changes.append(Change("inline-math-spacing", 0, f"{trimmed} span(s)", "trimmed"))

    dollar_count, even = display_parity(result_text)
    padded = find_delimiter_whitespace(result_text)
    heading_levels = Counter(
        len(match.group(1))
        for line in result_text.split("\n")
        if (match := HEADING_RE.match(line))
    )

    diagnostics: dict[str, object] = {
        "dollar_run_histogram": dollar_run_histogram(result_text),
        "display_delimiter_count": dollar_count,
        "display_delimiters_even": even,
        "inline_spans_with_padding": len(padded),
        "heading_level_counts": dict(sorted(heading_levels.items())),
    }

    if not even:
        logger.warning(
            "Odd number of `$$` delimiters (%d): a display block is never closed, "
            "which makes pandoc read the rest of the document as mathematics.",
            dollar_count,
        )
    welded = [n for n in dollar_run_histogram(result_text) if n > 2]
    if welded:
        logger.warning("Welded `$` runs of length %s present", welded)
    if padded and not fix_math_spacing:
        logger.warning(
            "%d inline math span(s) have whitespace inside their delimiters. The "
            "current parser produces none of these; re-run with fix_math_spacing=True "
            "if the parser has regressed.",
            len(padded),
        )

    return NormalizeResult(
        text=result_text, changes=changes, notes=notes, diagnostics=diagnostics
    )


__all__ = [
    "HEADING_RE",
    "INLINE_MATH_RE",
    "Change",
    "MathSpan",
    "Note",
    "NormalizeResult",
    "display_parity",
    "dollar_run_histogram",
    "find_delimiter_whitespace",
    "fix_inline_math_spacing",
    "iter_math_spans",
    "merge_split_headings",
    "normalize",
]
