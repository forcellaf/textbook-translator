"""
Convert MinerU's raw HTML ``<table>`` blocks into pandoc pipe tables.

Why this is not optional
------------------------
MinerU emits tables as raw HTML. Pandoc's markdown reader keeps raw HTML as
an opaque block, and **the LaTeX writer discards it** -- so every table
vanishes from the PDF, leaving the cell text scattered as loose lines. This
conversion is the single change that gets tables into the output at all.

Degeneracy policy
-----------------
The cloud VLM parser produces well-formed tables (19 of the test book's 24
are perfectly rectangular), but a confidently wrong table is worse than an
obviously missing one, so anything that cannot be represented faithfully is
**warned about and left exactly as it was**. Three signatures trigger that:

* ``colspan``/``rowspan`` greater than 1. A pipe table has no way to express
  a merged cell, and expanding one by repeating its content invents data.
  All 5 flagged tables in the test book are this case -- and inspecting them
  confirms they are *legitimate* merged header/grouping cells (``抗磁质``
  spanning four rows of materials), not the old parser's data loss. They stay
  as HTML, which at least keeps the text visible in the markdown.
* Rows with inconsistent cell counts.
* A row collapsed to one cell holding as many space-separated groups as its
  sibling rows have cells -- the old parser's merged-cell signature, where
  five material names ended up in one cell against three values.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)

_BARE_PIPE_RE = re.compile(r"(?<!\\)\|")

# Collapse heuristic tuning. A cell must hold at least `_COLLAPSE_MIN_GROUPS`
# space-separated groups AND that many times more than any sibling cell in
# the same column before it is called collapsed. Deliberately blunt: a false
# positive costs one table left as readable HTML, a false negative ships a
# confidently wrong table.
_MIN_COLS_FOR_GROUP_CHECK = 2
_COLLAPSE_MIN_GROUPS = 4
_COLLAPSE_RATIO = 3


@dataclass(frozen=True)
class TableReport:
    """What happened to one ``<table>`` block."""

    index: int  # 1-based, in document order
    line: int  # 1-based line the block starts on
    rows: int
    converted: bool
    reason: str = ""  # why it was left alone, when it was


class _TableParser(HTMLParser):
    """Collect ``<tr>``/``<td>``/``<th>`` into rows of (text, colspan, rowspan)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._span: tuple[int, int] = (1, 1)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._close_cell()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:  # a <td> outside any <tr>
                self._row = []
            values = dict(attrs)
            self._span = (_as_span(values.get("colspan")), _as_span(values.get("rowspan")))
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_cell()
            self._close_row()
        elif tag == "table":
            self._close_cell()
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _close_cell(self) -> None:
        if self._cell is None:
            return
        text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
        assert self._row is not None
        self._row.append((text, *self._span))
        self._cell = None
        self._span = (1, 1)

    def _close_row(self) -> None:
        if self._row:
            self.rows.append(self._row)
        self._row = None


def _as_span(value: str | None) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return 1


def _degeneracy_reason(rows: list[list[tuple[str, int, int]]]) -> str:
    """Return why ``rows`` cannot be converted faithfully, or ``""`` if it can."""
    if not rows:
        return "no <tr> rows found"

    spans = sum(1 for row in rows for _, colspan, rowspan in row if colspan > 1 or rowspan > 1)
    if spans:
        return (
            f"{spans} merged cell(s) (colspan/rowspan); a pipe table cannot "
            "express these without inventing content"
        )

    widths = {len(row) for row in rows}
    if len(widths) > 1:
        return f"inconsistent cell counts across rows: {sorted(widths)}"

    columns = widths.pop()
    if columns == 0:
        return "rows contain no cells"

    groups = [[len(text.split()) for text, _, _ in row] for row in rows]

    if columns < _MIN_COLS_FOR_GROUP_CHECK:
        # A one-column table whose single cell holds several space-separated
        # groups is a table whose columns were collapsed into one.
        for index, row in enumerate(groups, start=1):
            if row[0] >= _COLLAPSE_MIN_GROUPS:
                return (
                    f"single-column table whose row {index} holds {row[0]} "
                    "space-separated groups; its columns were probably collapsed"
                )
        return ""

    # A cell holding far more space-separated groups than every other cell in
    # its own column is the old parser's merged-cell signature: five material
    # names crammed into one cell against three values.
    for column in range(columns):
        counts = [row[column] for row in groups]
        for index, count in enumerate(counts, start=1):
            others = [c for j, c in enumerate(counts, start=1) if j != index]
            ceiling = max(others) if others else 0
            if count >= _COLLAPSE_MIN_GROUPS and count >= max(1, ceiling) * _COLLAPSE_RATIO:
                return (
                    f"row {index}, column {column + 1} holds {count} space-separated "
                    f"groups where every sibling row holds at most {ceiling}; its "
                    "cells were probably collapsed"
                )

    return ""


def _cell_to_markdown(text: str) -> str:
    """Escape a cell so it survives as one pipe-table cell.

    Only a bare ``|`` is escaped. Backslashes are left exactly as they are:
    these cells are full of LaTeX (``$2 \\times {10}^{21}$``) and escaping
    the backslashes would turn every formula into literal text. A ``\\|``
    that is already escaped -- or is LaTeX's norm delimiter -- is left alone.
    """
    # convert_charrefs already decoded entities; unescaping again is harmless
    # and catches double-encoded input.
    plain = html.unescape(text)
    return _BARE_PIPE_RE.sub(r"\\|", plain).strip() or " "


def _to_pipe_table(rows: list[list[tuple[str, int, int]]]) -> str:
    """Render rectangular ``rows`` as a pandoc pipe table.

    The first row becomes the header. MinerU rarely emits ``<th>``, and
    pandoc requires a header row, so the first row is promoted -- it stays
    visible either way, just set in bold.
    """
    rendered = [[_cell_to_markdown(text) for text, _, _ in row] for row in rows]
    columns = len(rendered[0])
    lines = ["| " + " | ".join(rendered[0]) + " |", "| " + " | ".join(["---"] * columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered[1:])
    return "\n".join(lines)


def convert_html_tables(text: str) -> tuple[str, list[TableReport]]:
    """Replace well-formed ``<table>`` blocks with pipe tables.

    Args:
        text: Markdown containing MinerU's raw HTML tables.

    Returns:
        ``(new_text, reports)``. Every table gets a ``TableReport``, whether
        or not it was converted; ``report.reason`` says why one was skipped.
        Degenerate tables are returned byte-identical to their input.
    """
    reports: list[TableReport] = []
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        block = match.group(0)
        line = text.count("\n", 0, match.start()) + 1

        parser = _TableParser()
        try:
            parser.feed(block)
            parser.close()
        except Exception as exc:  # noqa: BLE001 - malformed HTML is a warn, not a crash
            reports.append(TableReport(counter, line, 0, False, f"unparseable HTML: {exc}"))
            return block

        rows = parser.rows
        reason = _degeneracy_reason(rows)
        if reason:
            reports.append(TableReport(counter, line, len(rows), False, reason))
            logger.warning("Table %d at line %d left as HTML: %s", counter, line, reason)
            return block

        reports.append(TableReport(counter, line, len(rows), True))
        # Blank lines on both sides, or pandoc glues the table to adjacent
        # prose and reads the whole thing as a paragraph.
        return "\n\n" + _to_pipe_table(rows) + "\n\n"

    converted = TABLE_RE.sub(replace, text)
    # Collapse the runs of blank lines the padding above can create.
    converted = re.sub(r"\n{3,}", "\n\n", converted)

    done = sum(1 for r in reports if r.converted)
    logger.info(
        "Converted %d of %d HTML table(s) to pipe tables; %d left as HTML for review",
        done,
        len(reports),
        len(reports) - done,
    )
    return converted, reports


__all__ = ["TABLE_RE", "TableReport", "convert_html_tables"]
