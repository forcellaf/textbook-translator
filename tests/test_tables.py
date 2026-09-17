"""Tests for src.tables (HTML tables -> markdown pipe tables).

A pipe table makes the grid explicit, so the translating model reproduces it
as a LaTeX tabular instead of inferring the structure from <td> tags. The
other half of the contract matters just as much: a table that cannot be
represented faithfully must come back byte-identical, never guessed at.
"""

from __future__ import annotations

from src.tables import convert_html_tables

WELL_FORMED = (
    "<table>"
    "<tr><td>铀核表面</td><td> $2 \\times {10}^{21}$ </td></tr>"
    "<tr><td>中子星表面</td><td>约 ${10}^{14}$ </td></tr>"
    "<tr><td>空气的电击穿强度</td><td> $3 \\times {10}^{6}$ </td></tr>"
    "</table>"
)

# Legitimate merged header/grouping cells, as seen in the real book: `抗磁质`
# spans four rows of materials. Not data loss -- but not expressible as a
# pipe table either.
MERGED_CELLS = (
    "<table>"
    '<tr><td colspan="2">磁介质种类</td><td>相对磁导率</td></tr>'
    '<tr><td rowspan="2">抗磁质</td><td>铋</td><td>0.99983</td></tr>'
    "<tr><td>汞</td><td>0.99997</td></tr>"
    "</table>"
)

# The old parser's data loss: five material names collapsed into one cell
# against three values.
COLLAPSED_CELLS = (
    "<table>"
    "<tr><td>material</td><td>value</td></tr>"
    "<tr><td>iron nickel cobalt copper silver</td><td>3</td></tr>"
    "<tr><td>lead</td><td>7</td></tr>"
    "</table>"
)

RAGGED = (
    "<table>"
    "<tr><td>a</td><td>b</td><td>c</td></tr>"
    "<tr><td>d</td><td>e</td></tr>"
    "</table>"
)


# ── Conversion ──────────────────────────────────────────────────────────────


def test_well_formed_table_becomes_a_pipe_table() -> None:
    result, reports = convert_html_tables(f"前言\n\n{WELL_FORMED}\n\n后记")

    assert "<table" not in result
    assert "| 铀核表面 | $2 \\times {10}^{21}$ |" in result
    assert "| --- | --- |" in result
    assert len(reports) == 1 and reports[0].converted


def test_latex_inside_cells_survives_untouched() -> None:
    """Escaping backslashes here would turn every formula into literal text."""
    result, _ = convert_html_tables(WELL_FORMED)

    assert "$2 \\times {10}^{21}$" in result
    assert "\\\\times" not in result


def test_the_table_is_surrounded_by_blank_lines() -> None:
    """Glued to adjacent prose, the table reads as part of the paragraph."""
    result, _ = convert_html_tables(f"前言\n{WELL_FORMED}\n后记")

    lines = result.split("\n")
    first = next(i for i, line in enumerate(lines) if line.startswith("|"))
    last = max(i for i, line in enumerate(lines) if line.startswith("|"))
    assert lines[first - 1].strip() == ""
    assert lines[last + 1].strip() == ""


def test_html_entities_are_decoded() -> None:
    table = "<table><tr><td>s&#x27;</td><td>s&gt;2f</td></tr><tr><td>a</td><td>b</td></tr></table>"
    result, reports = convert_html_tables(table)

    assert reports[0].converted
    assert "s'" in result and "s>2f" in result


def test_a_bare_pipe_in_a_cell_is_escaped() -> None:
    table = "<table><tr><td>a|b</td><td>c</td></tr><tr><td>d</td><td>e</td></tr></table>"
    result, _ = convert_html_tables(table)

    assert "a\\|b" in result
    # Escaping must not split the row into three cells.
    assert result.count("|", result.index("a\\|b") - 2) >= 3


def test_th_cells_are_handled_like_td() -> None:
    table = "<table><tr><th>name</th><th>value</th></tr><tr><td>a</td><td>1</td></tr></table>"
    result, reports = convert_html_tables(table)

    assert reports[0].converted
    assert "| name | value |" in result


def test_conversion_is_idempotent() -> None:
    once, _ = convert_html_tables(WELL_FORMED)
    twice, reports = convert_html_tables(once)

    assert twice == once
    assert reports == []


# ── Degeneracy: warn and leave ──────────────────────────────────────────────


def test_merged_cells_warn_and_leave_the_table_untouched() -> None:
    result, reports = convert_html_tables(MERGED_CELLS)

    assert result == MERGED_CELLS
    assert not reports[0].converted
    assert "merged cell" in reports[0].reason


def test_ragged_rows_warn_and_leave_the_table_untouched() -> None:
    result, reports = convert_html_tables(RAGGED)

    assert result == RAGGED
    assert not reports[0].converted
    assert "inconsistent cell counts" in reports[0].reason


def test_collapsed_cells_warn_and_leave_the_table_untouched() -> None:
    result, reports = convert_html_tables(COLLAPSED_CELLS)

    assert result == COLLAPSED_CELLS
    assert not reports[0].converted
    assert "collapsed" in reports[0].reason


def test_single_column_table_with_crowded_cells_is_degenerate() -> None:
    table = "<table><tr><td>iron nickel cobalt copper</td></tr><tr><td>lead</td></tr></table>"
    result, reports = convert_html_tables(table)

    assert result == table
    assert not reports[0].converted


def test_an_empty_table_is_reported_not_converted() -> None:
    result, reports = convert_html_tables("<table></table>")

    assert result == "<table></table>"
    assert not reports[0].converted


def test_a_good_and_a_bad_table_are_handled_independently() -> None:
    result, reports = convert_html_tables(f"{WELL_FORMED}\n\n{MERGED_CELLS}")

    assert [r.converted for r in reports] == [True, False]
    assert MERGED_CELLS in result  # the degenerate one survives verbatim
    assert "| 铀核表面" in result  # the good one converted


def test_no_tables_is_a_no_op() -> None:
    text = "Just prose with $x$ and no tables.\n"
    result, reports = convert_html_tables(text)

    assert result == text
    assert reports == []


def test_reports_carry_line_numbers() -> None:
    _, reports = convert_html_tables(f"line one\nline two\n\n{MERGED_CELLS}")
    assert reports[0].line == 4
