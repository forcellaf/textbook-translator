"""Tests for src.chapter_splitter (merged.md -> chapters/)."""

from __future__ import annotations

import json
from pathlib import Path

from src.chapter_splitter import split_into_chapters


def _padded(label: str, repeats: int = 40) -> str:
    """Generate a filler paragraph so a chapter's share of total content
    stays comfortably under the 60% confidence threshold."""
    return f"{label} content. " * repeats


def test_no_heading_falls_back_to_single_whole_document_chapter(tmp_path: Path, caplog) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text("Just some plain text with no headings at all.\n" * 5, encoding="utf-8")
    output_dir = tmp_path / "chapters"

    with caplog.at_level("WARNING"):
        result = split_into_chapters(merged_path, output_dir)

    assert result.detection_confidence is False
    assert len(result.chapters) == 1
    assert result.chapters[0].file_path.exists()
    assert "Just some plain text" in result.chapters[0].file_path.read_text(encoding="utf-8")
    assert any("not confident" in r.message for r in caplog.records)


def test_numbered_chinese_chapters_produce_ordered_files(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        "本书前言，介绍写作背景。\n\n"
        f"# 第一章 绪论\n\n{_padded('intro')}\n\n"
        f"# 第二章 分析方法\n\n{_padded('methods')}\n\n"
        f"# 第三章 总结\n\n{_padded('summary')}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    assert result.detection_confidence is True
    titles = [c.title for c in result.chapters]
    assert titles[0] == "Front Matter"
    assert "第一章 绪论" in titles[1]
    assert "第二章 分析方法" in titles[2]
    assert "第三章 总结" in titles[3]

    # Files are numbered/ordered and actually written.
    filenames = sorted(p.name for p in output_dir.glob("*.md"))
    assert filenames == sorted(c.file_path.name for c in result.chapters)
    assert filenames[0].startswith("00_")
    assert filenames[1].startswith("01_")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["detection_confidence"] is True
    assert len(manifest["chapters"]) == 4


def test_front_matter_extracted_into_its_own_file(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        "Preface text goes here before any chapter starts.\n\n"
        f"# Chapter 1\n\n{_padded('one')}\n\n"
        f"# Chapter 2\n\n{_padded('two')}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    front_matter = result.chapters[0]
    assert front_matter.title == "Front Matter"
    assert "Preface text" in front_matter.file_path.read_text(encoding="utf-8")
    assert front_matter.file_path.name.startswith("00_")


def test_no_front_matter_file_when_document_starts_with_heading(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        f"# Chapter 1\n\n{_padded('one')}\n\n# Chapter 2\n\n{_padded('two')}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    assert all(c.title != "Front Matter" for c in result.chapters)


def test_image_paths_shifted_up_one_directory_level(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        f"# Chapter 1\n\n![figure](chunks/book_part_001/images/fig1.jpg)\n\n{_padded('one')}\n\n"
        f"# Chapter 2\n\n{_padded('two')}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    chapter1_text = result.chapters[0].file_path.read_text(encoding="utf-8")
    assert "../chunks/book_part_001/images/fig1.jpg" in chapter1_text


def test_low_confidence_when_only_one_chapter_found(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(f"# 第一章 唯一的章节\n\n{_padded('solo', repeats=200)}\n", encoding="utf-8")
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    assert result.detection_confidence is False
    assert len(result.chapters) == 1


def test_low_confidence_when_chapter_dominates_content(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        f"# 第一章 小\n\n{_padded('tiny', repeats=2)}\n\n"
        f"# 第二章 巨大\n\n{_padded('huge', repeats=500)}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"

    result = split_into_chapters(merged_path, output_dir)

    assert result.detection_confidence is False
    assert len(result.chapters) == 1


def test_idempotent_second_call_reuses_existing_split(tmp_path: Path) -> None:
    merged_path = tmp_path / "merged.md"
    merged_path.write_text(
        f"# 第一章 甲\n\n{_padded('a')}\n\n# 第二章 乙\n\n{_padded('b')}\n", encoding="utf-8"
    )
    output_dir = tmp_path / "chapters"

    first = split_into_chapters(merged_path, output_dir)
    second = split_into_chapters(merged_path, output_dir)

    assert [c.file_path for c in first.chapters] == [c.file_path for c in second.chapters]
    assert first.detection_confidence == second.detection_confidence
