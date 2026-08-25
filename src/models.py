"""
Shared dataclasses for the PDF splitting / merging / chapter-splitting
pipeline (src.splitter, src.merger, src.chapter_splitter).

Kept in their own module (rather than living inside the modules that
produce them) so all three pipeline stages -- and their tests -- can import
a single, stable set of value types without risking circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitChunk:
    """One page-range slice of a source PDF, written out as its own PDF file.

    ``start_page``/``end_page`` are 1-based and inclusive, and always refer
    to page positions in the ORIGINAL (unsplit) document -- not to page
    numbers within the chunk file itself.
    """

    index: int  # 1-based chunk number (chunk 1, 2, 3, ...), in document order
    start_page: int  # 1-based inclusive start page in the original PDF
    end_page: int  # 1-based inclusive end page in the original PDF
    output_path: Path  # path to the written chunk PDF
    size_bytes: int
    sha256: str

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass(frozen=True)
class SplitResult:
    """Result of splitting one source PDF into chunk PDFs."""

    source_path: Path
    total_pages: int
    total_size_bytes: int
    output_dir: Path
    chunks: tuple[SplitChunk, ...]
    manifest_path: Path


@dataclass(frozen=True)
class ChapterInfo:
    """One detected chapter (or front/back-matter section) of a merged doc.

    ``char_start``/``char_end`` are 0-based character offsets (``end``
    exclusive) into the source ``merged.md`` text this chapter was sliced
    from. They stand in for a "page span": once MinerU has flattened a PDF
    to Markdown there is no reliable original-PDF page number left to
    attach to a chapter, so character offsets are the next best positional
    metadata.
    """

    index: int  # 0 = front matter, 1..N = chapters, in document order
    title: str
    source_heading: str  # raw Markdown heading line that triggered detection ("" for front matter)
    char_count: int
    char_start: int
    char_end: int
    file_path: Path


@dataclass(frozen=True)
class ChapterResult:
    """Result of splitting a merged Markdown document into per-chapter files."""

    source_path: Path
    output_dir: Path
    chapters: tuple[ChapterInfo, ...]
    detection_confidence: bool
    manifest_path: Path
