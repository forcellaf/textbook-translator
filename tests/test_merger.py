"""Tests for src.merger (chunk Markdown -> merged.md)."""

from __future__ import annotations

from pathlib import Path

from src.merger import merge_chunks
from src.models import SplitChunk


def _make_chunk(
    book_work_dir: Path,
    stem: str,
    index: int,
    start_page: int,
    end_page: int,
    md_content: str,
    md_name: str = "full.md",
    image_name: str | None = "img1.jpg",
) -> SplitChunk:
    chunk_dir = book_work_dir / "chunks" / stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / md_name).write_text(md_content, encoding="utf-8")
    if image_name:
        images_dir = chunk_dir / "images"
        images_dir.mkdir(exist_ok=True)
        (images_dir / image_name).write_bytes(b"fake-image-bytes")
    fake_pdf_path = book_work_dir / "chunks" / f"{stem}.pdf"
    return SplitChunk(
        index=index,
        start_page=start_page,
        end_page=end_page,
        output_path=fake_pdf_path,
        size_bytes=1234,
        sha256="deadbeef",
    )


def test_image_path_rewriting_markdown_and_html(tmp_path: Path) -> None:
    book_work_dir = tmp_path / "book"
    chunk = _make_chunk(
        book_work_dir,
        "book_part_001",
        1,
        1,
        10,
        "# Title\n\n![alt text](images/img1.jpg)\n\n"
        '<img src="images/img1.jpg" alt="dup">\n',
    )

    merged_path = merge_chunks([chunk], book_work_dir)
    merged_text = merged_path.read_text(encoding="utf-8")

    assert "chunks/book_part_001/images/img1.jpg" in merged_text
    # The rewritten path must actually resolve on disk from merged.md's dir.
    for line in merged_text.splitlines():
        if "images/img1.jpg" in line and "(" in line:
            rel = line.split("(")[1].split(")")[0]
            assert (book_work_dir / rel).exists()


def test_single_chunk_no_dedupe_needed(tmp_path: Path) -> None:
    book_work_dir = tmp_path / "book"
    chunk = _make_chunk(book_work_dir, "book_part_001", 1, 1, 10, "# Only chunk\n\nHello world.\n")

    merged_path = merge_chunks([chunk], book_work_dir)
    assert "Hello world." in merged_path.read_text(encoding="utf-8")


def test_overlap_dedupe_confident_match_drops_duplicate(tmp_path: Path) -> None:
    book_work_dir = tmp_path / "book"
    shared_paragraph = (
        "This paragraph appears on the shared overlap page between chunk one "
        "and chunk two, and is long enough to be a confident match."
    )
    chunk1 = _make_chunk(
        book_work_dir,
        "book_part_001",
        1,
        1,
        10,
        f"# Chunk One\n\nUnique chunk-one content.\n\n{shared_paragraph}\n",
        image_name=None,
    )
    chunk2 = _make_chunk(
        book_work_dir,
        "book_part_002",
        2,
        9,
        20,
        f"{shared_paragraph}\n\nUnique chunk-two content.\n",
        image_name=None,
    )

    merged_path = merge_chunks([chunk1, chunk2], book_work_dir)
    merged_text = merged_path.read_text(encoding="utf-8")

    assert merged_text.count(shared_paragraph) == 1
    assert "Unique chunk-one content." in merged_text
    assert "Unique chunk-two content." in merged_text
    assert "OVERLAP BOUNDARY" not in merged_text


def test_overlap_dedupe_mismatch_keeps_both_and_inserts_marker(tmp_path: Path) -> None:
    book_work_dir = tmp_path / "book"
    chunk1 = _make_chunk(
        book_work_dir,
        "book_part_001",
        1,
        1,
        10,
        "# Chunk One\n\nEnd of chunk one, OCR'd this way this time.\n",
        image_name=None,
    )
    chunk2 = _make_chunk(
        book_work_dir,
        "book_part_002",
        2,
        9,
        20,
        "Start of chunk two, OCR'd completely differently on this run.\n",
        image_name=None,
    )

    merged_path = merge_chunks([chunk1, chunk2], book_work_dir)
    merged_text = merged_path.read_text(encoding="utf-8")

    # Both copies survive since no confident match was found.
    assert "End of chunk one" in merged_text
    assert "Start of chunk two" in merged_text
    assert "OVERLAP BOUNDARY" in merged_text


def test_non_overlapping_chunks_concatenated_without_marker(tmp_path: Path) -> None:
    """Chunks produced by the size-guard backstop (non-overlapping page
    ranges) must not trigger dedupe warnings/markers."""
    book_work_dir = tmp_path / "book"
    chunk1 = _make_chunk(
        book_work_dir, "book_part_001", 1, 1, 5, "# Part 1\n\nContent A.\n", image_name=None
    )
    chunk2 = _make_chunk(
        book_work_dir, "book_part_002", 2, 6, 10, "# Part 2\n\nContent B.\n", image_name=None
    )

    merged_path = merge_chunks([chunk1, chunk2], book_work_dir)
    merged_text = merged_path.read_text(encoding="utf-8")

    assert "OVERLAP BOUNDARY" not in merged_text
    assert "Content A." in merged_text
    assert "Content B." in merged_text


def test_manifest_written(tmp_path: Path) -> None:
    book_work_dir = tmp_path / "book"
    chunk = _make_chunk(book_work_dir, "book_part_001", 1, 1, 10, "# Title\n\nBody.\n")

    merge_chunks([chunk], book_work_dir)

    manifest_path = book_work_dir / "manifest.json"
    assert manifest_path.exists()
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "merged_path" in payload
    assert len(payload["chunks"]) == 1
