"""Tests for src.splitter (PDF -> fixed-page-range chunk PDFs)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from src.splitter import split_pdf


def _make_pdf(path: Path, num_pages: int, toc: list[list] | None = None) -> None:
    """Write a `num_pages`-page PDF to `path`, each page with a bit of text
    so pages aren't byte-identical (keeps sizes/hash realistic)."""
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} of {num_pages}")
    if toc:
        doc.set_toc(toc)
    doc.save(path)
    doc.close()


def _make_encrypted_pdf(path: Path, num_pages: int = 3) -> None:
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page()
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    doc.close()


# ── No-op: small file within limits produces exactly one chunk ─────────────


def test_small_pdf_produces_single_chunk(tmp_path: Path) -> None:
    pdf_path = tmp_path / "small.pdf"
    _make_pdf(pdf_path, num_pages=5)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)

    assert result.total_pages == 5
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert (chunk.start_page, chunk.end_page) == (1, 5)
    assert chunk.output_path.exists()
    assert result.manifest_path.exists()


# ── Exact tiling math with overlap (the off-by-one risk area) ──────────────


def test_tiling_math_450_pages_default_settings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "big.pdf"
    _make_pdf(pdf_path, num_pages=450)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)

    ranges = [(c.start_page, c.end_page) for c in result.chunks]
    assert ranges == [(1, 190), (189, 378), (377, 450)]
    # Each pair of consecutive chunks shares exactly `overlap_pages` pages.
    for prev, cur in zip(ranges, ranges[1:]):
        assert prev[1] - cur[0] + 1 == 2
    # Chunk indices are sequential starting at 1, and filenames sort
    # lexicographically in document order.
    assert [c.index for c in result.chunks] == [1, 2, 3]
    names = sorted(c.output_path.name for c in result.chunks)
    assert names == [c.output_path.name for c in result.chunks]


def test_tiling_math_no_split_needed_when_exactly_at_limit(tmp_path: Path) -> None:
    pdf_path = tmp_path / "exact.pdf"
    _make_pdf(pdf_path, num_pages=190)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)

    assert len(result.chunks) == 1
    assert (result.chunks[0].start_page, result.chunks[0].end_page) == (1, 190)


# ── Bookmark snapping ────────────────────────────────────────────────────────


def test_bookmark_snap_adjusts_chunk_boundaries(tmp_path: Path) -> None:
    pdf_path = tmp_path / "toc.pdf"
    # Fixed boundary for chunk 1 would be page 100; put a level-1 TOC entry
    # at page 96 (within the +/-10 snap window) so it should snap there.
    toc = [[1, "Chapter A", 1], [1, "Chapter B", 96], [1, "Chapter C", 150]]
    _make_pdf(pdf_path, num_pages=200, toc=toc)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=100, max_size_mb=180, overlap_pages=2)

    ranges = [(c.start_page, c.end_page) for c in result.chunks]
    assert ranges == [(1, 96), (95, 194), (193, 200)]


def test_no_toc_falls_back_to_fixed_ranges_silently(tmp_path: Path, caplog) -> None:
    pdf_path = tmp_path / "no_toc.pdf"
    _make_pdf(pdf_path, num_pages=200)  # no TOC
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=100, max_size_mb=180, overlap_pages=2)

    ranges = [(c.start_page, c.end_page) for c in result.chunks]
    assert ranges == [(1, 100), (99, 198), (197, 200)]
    assert not any(r.levelno >= 30 for r in caplog.records)  # no warnings/errors


def test_bookmark_snap_ignored_if_it_would_shrink_below_minimum(tmp_path: Path) -> None:
    pdf_path = tmp_path / "toc_too_close.pdf"
    # A TOC entry very close to the chunk start would shrink the chunk well
    # below the minimum snapped size; the fixed boundary must be kept.
    toc = [[1, "Chapter A", 1], [1, "Chapter B", 5]]
    _make_pdf(pdf_path, num_pages=200, toc=toc)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=100, max_size_mb=180, overlap_pages=2)

    ranges = [(c.start_page, c.end_page) for c in result.chunks]
    assert ranges[0] == (1, 100)  # fixed boundary kept, snap rejected


# ── Encrypted PDF ────────────────────────────────────────────────────────────


def test_encrypted_pdf_raises_runtime_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "encrypted.pdf"
    _make_encrypted_pdf(pdf_path)
    output_dir = tmp_path / "chunks"

    with pytest.raises(RuntimeError, match=r"(?i)encrypt"):
        split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)


def test_missing_pdf_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        split_pdf(missing, tmp_path / "chunks", max_pages=190, max_size_mb=180, overlap_pages=2)


# ── Size guard backstop ──────────────────────────────────────────────────────


def test_size_guard_splits_oversized_chunk_further(tmp_path: Path) -> None:
    pdf_path = tmp_path / "size_guard.pdf"
    _make_pdf(pdf_path, num_pages=8)
    output_dir = tmp_path / "chunks"

    # An unreasonably small size limit forces every page range to be
    # recursively halved down to single pages.
    result = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=0, overlap_pages=2)

    # No page lost or duplicated by the size-guard subdivision.
    covered = sorted({p for c in result.chunks for p in range(c.start_page, c.end_page + 1)})
    assert covered == list(range(1, 9))
    assert len(result.chunks) >= 8  # halved down to (at most) single pages
    for chunk in result.chunks:
        assert chunk.output_path.exists()


# ── Idempotency ──────────────────────────────────────────────────────────────


def test_idempotent_second_call_reuses_existing_split(tmp_path: Path) -> None:
    pdf_path = tmp_path / "idempotent.pdf"
    _make_pdf(pdf_path, num_pages=450)
    output_dir = tmp_path / "chunks"

    first = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)
    first_hashes = [c.sha256 for c in first.chunks]

    second = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)
    second_hashes = [c.sha256 for c in second.chunks]

    assert first_hashes == second_hashes
    assert len(second.chunks) == len(first.chunks)


# ── Manifest ─────────────────────────────────────────────────────────────────


def test_manifest_json_written_with_chunk_fields(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manifest.pdf"
    _make_pdf(pdf_path, num_pages=10)
    output_dir = tmp_path / "chunks"

    result = split_pdf(pdf_path, output_dir, max_pages=190, max_size_mb=180, overlap_pages=2)

    assert result.manifest_path.exists()
    import json

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["total_pages"] == 10
    assert len(payload["chunks"]) == 1
    chunk_payload = payload["chunks"][0]
    for key in ("index", "start_page", "end_page", "output_path", "size_bytes", "sha256"):
        assert key in chunk_payload
