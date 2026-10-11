"""Tests for src.page_range and the --page-range path of parse_book.py.

No MinerU call is made: trimming is local, and the parse_book tests stop
before anything is uploaded.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pymupdf
import pytest

from src.page_range import (
    PageRange,
    first_heading_warning,
    load_page_range,
    original_page,
    page_offset,
    record_offset,
    trim_pdf,
)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _make_pdf(path: Path, pages: int) -> None:
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), f"physical page {number}")
    doc.save(path)
    doc.close()


def _write_range(path: Path, **fields) -> Path:
    data = {"start": 15, "end": 30, "first_chapter_number": 12, "by": "agent", **fields}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_trimming_keeps_exactly_the_body_and_the_offset_round_trips(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 40)
    page_range = load_page_range(_write_range(tmp_path / "page_range.json"), page_count=40)
    body = tmp_path / "source_body.pdf"

    assert trim_pdf(pdf, page_range, body)
    with pymupdf.open(body) as doc:
        assert doc.page_count == 16 == page_range.page_count
        first = doc[0].get_text().strip()
        last = doc[doc.page_count - 1].get_text().strip()
    assert first == "physical page 15" and last == "physical page 30"

    assert page_range.offset == 14
    assert original_page(1, page_range.offset) == 15
    assert original_page(16, page_range.offset) == 30


def test_the_offset_is_recorded_in_the_parts_manifest(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "manifest.json").write_text(json.dumps({"chunks": []}), encoding="utf-8")
    record_offset(tmp_path, PageRange(start=15, end=30), tmp_path / "book.pdf")

    manifest = json.loads((tmp_path / "parts" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["page_offset"] == 14 and manifest["trimmed"] is True
    assert page_offset(tmp_path) == 14


def test_a_cut_is_reused_until_the_range_changes(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 40)
    stamp = _write_range(tmp_path / "page_range.json")
    body = tmp_path / "source_body.pdf"
    page_range = load_page_range(stamp)
    assert trim_pdf(pdf, page_range, body, stamp=stamp)
    assert not trim_pdf(pdf, page_range, body, stamp=stamp)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"start": 0}, "1 <= start"),
        ({"start": 31}, "1 <= start"),
        ({"end": 99}, "past the PDF's last page"),
        ({"start": "15"}, "integers"),
    ],
)
def test_an_invalid_range_is_refused(tmp_path: Path, fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_page_range(_write_range(tmp_path / "page_range.json", **fields), page_count=40)


def test_page_range_without_the_file_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    parse_book = importlib.import_module("parse_book")
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 5)

    code = parse_book.main(
        [str(pdf), str(tmp_path / "work"), "--page-range", str(tmp_path / "nope.json")]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "nope.json not found" in err and "page-range-finder" in err


# ── The first-heading cross-check ───────────────────────────────────────────


def test_a_book_that_starts_with_its_table_of_contents_warns() -> None:
    merged = "# 目录\n\n第12章 静电场 ...... 1\n\n# 静电场\n\n## 12.1 电荷\n\n正文。\n"
    warning = first_heading_warning(merged)
    assert warning is not None and "目录" in warning and "front matter" in warning


def test_a_book_that_starts_with_a_chapter_does_not_warn() -> None:
    merged = "# 静电场\n\n## 12.1 电荷\n\n正文。\n\n## 12.2 电场\n\n正文。\n"
    assert first_heading_warning(merged) is None
    assert first_heading_warning("# 第12章 静电场\n\n正文。\n") is None


def test_a_part_followed_by_a_chapter_does_not_warn() -> None:
    merged = "# 第三篇 电磁学\n\n本篇。\n\n# 第12章 静电场\n\n## 12.1 电荷\n"
    assert first_heading_warning(merged) is None


def test_an_unrecognisable_first_heading_warns() -> None:
    warning = first_heading_warning("# 致谢\n\n感谢。\n\n# 静电场\n\n## 12.1 电荷\n")
    assert warning is not None and "start page may be wrong" in warning


# ── render_pages.py ─────────────────────────────────────────────────────────


def test_render_pages_names_files_by_physical_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    render_pages = importlib.import_module("render_pages")
    pdf = tmp_path / "book.pdf"
    _make_pdf(pdf, 12)
    out = tmp_path / "page_images"

    assert render_pages.main([str(pdf), str(out), "--pages", "1-2,-2", "--dpi", "20"]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == [
        "p0001.png", "p0002.png", "p0011.png", "p0012.png"
    ]
    manifest = json.loads((out / "pages.json").read_text(encoding="utf-8"))
    assert manifest["page_count"] == 12
    assert manifest["text_layer"]["1"] is True
