"""
Split an oversized PDF into fixed page-range chunks so each chunk fits
MinerU's Precision Extract API limits (<= 200 pages, <= 200 MB per file).

Uses PyMuPDF (imported as ``pymupdf``, bundled with ``magic-pdf[full]`` --
no new dependency needed) for all page manipulation.

See ``split_pdf`` for the full tiling/overlap/bookmark-snapping/size-guard
contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pymupdf

from src.models import SplitChunk, SplitResult

logger = logging.getLogger(__name__)

# Bookmark snapping never lets a chunk end up smaller than this, and never
# searches further than this many pages from the fixed boundary.
_MIN_SNAPPED_CHUNK_PAGES = 20
_SNAP_WINDOW_PAGES = 10

_MANIFEST_NAME = "manifest.json"


def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    max_pages: int,
    max_size_mb: int,
    overlap_pages: int,
) -> SplitResult:
    """
    Split ``pdf_path`` into overlapping, fixed-size page-range chunk PDFs.

    Chunk tiling convention (read this before touching the math below)
    --------------------------------------------------------------------
    Page numbers are 1-based and inclusive, and refer to positions in the
    ORIGINAL document. The first chunk always starts at page 1. Every
    subsequent chunk starts ``overlap_pages`` pages before the previous
    chunk's end page, so each pair of consecutive chunks shares exactly
    ``overlap_pages`` pages:

        chunk 1: [1, max_pages]
        chunk 2: [end(1) - overlap_pages + 1, end(1) - overlap_pages + max_pages]
        chunk k (k>1): [end(k-1) - overlap_pages + 1, end(k-1) - overlap_pages + max_pages]

    Concretely, with ``max_pages=190``, ``overlap_pages=2`` and a 450-page
    document:

        chunk 1: pages 1-190
        chunk 2: pages 189-378   (190 - 2 + 1 = 189)
        chunk 3: pages 377-450   (378 - 2 + 1 = 377, clipped to doc length)

    The overlap exists so MinerU sees each shared page in full in both
    neighboring chunks (preserving paragraph/table continuity across the
    cut); ``src.merger.merge_chunks`` is responsible for deduplicating the
    repeated pages afterward. This function asserts that its own fixed
    (pre-size-guard) tiling honors this invariant exactly -- an off-by-one
    here would silently corrupt every downstream stage.

    Bookmark snapping (best-effort only)
    -------------------------------------
    If ``pdf_path`` has a table of contents (``doc.get_toc()``), each
    chunk's end page (except the last chunk, which always runs to the end
    of the document) is "snapped" to the nearest top-level (TOC level 1)
    bookmark page within +/- ``_SNAP_WINDOW_PAGES`` pages, so chunks tend to
    end on chapter boundaries instead of mid-chapter. The next chunk's
    start is then re-derived from the (possibly snapped) end, so the
    overlap invariant above still holds exactly even after snapping.

    A snap is only applied if it keeps the chunk within
    [``_MIN_SNAPPED_CHUNK_PAGES``, ``max_pages``] pages; otherwise the fixed
    boundary is kept. If the document has no TOC (or no level-1 entries),
    snapping is silently skipped -- fixed ranges are used with zero
    warnings.

    Size guard (backstop, not the primary constraint)
    ----------------------------------------------------
    Page count is normally the binding constraint, but image-heavy scans
    can still exceed ``max_size_mb`` within ``max_pages`` pages. After the
    primary tiling (above) is computed, any chunk whose rendered size
    exceeds ``max_size_mb`` is recursively split in half (by page count,
    no added overlap -- this is a pure size backstop, not a content-aware
    split) until every resulting piece is under the limit. This can
    increase the total chunk count beyond what the tiling formula above
    predicts; the final, possibly-larger chunk list is what gets numbered,
    written, and recorded in the manifest.

    Idempotency
    -----------
    If ``output_dir/manifest.json`` already exists and is newer than
    ``pdf_path``, the existing split is reused (no PDF is re-opened or
    re-written) and a ``SplitResult`` reconstructed from the manifest is
    returned directly.

    Args:
        pdf_path: Source PDF to split.
        output_dir: Directory to write chunk PDFs + manifest.json into.
        max_pages: Maximum pages per chunk (soft target -- snapping and the
            size guard can only ever shrink a chunk further, never grow it
            past this).
        max_size_mb: Maximum size per chunk file, in megabytes.
        overlap_pages: Number of pages shared between consecutive
            tiling-derived chunks.

    Returns:
        A SplitResult describing every chunk written (or reused).

    Raises:
        FileNotFoundError: If ``pdf_path`` doesn't exist.
        RuntimeError: If the PDF is encrypted/password-protected, or
            PyMuPDF fails to open/read it.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    manifest_path = output_dir / _MANIFEST_NAME

    # ── Idempotency: skip re-splitting if already done and up to date ──────
    if manifest_path.exists() and manifest_path.stat().st_mtime > pdf_path.stat().st_mtime:
        existing = _load_manifest(manifest_path)
        if existing is not None and all(c.output_path.exists() for c in existing.chunks):
            logger.info("Already split, skipping: %s", manifest_path)
            return existing
        logger.warning(
            "Manifest %s is stale or references missing chunk files; re-splitting", manifest_path
        )

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    try:
        if doc.is_encrypted or doc.needs_pass:
            raise RuntimeError(
                f"'{pdf_path}' is password-protected/encrypted and cannot be split. "
                f"Decrypt it first, e.g.: qpdf --decrypt '{pdf_path}' '{pdf_path.stem}_decrypted.pdf'"
            )

        total_pages = doc.page_count
        total_size_bytes = pdf_path.stat().st_size

        toc_pages = _top_level_toc_pages(doc)
        ranges = _build_chunk_ranges(total_pages, max_pages, overlap_pages, toc_pages)
        _assert_tiling(ranges, overlap_pages)

        max_size_bytes = max_size_mb * 1024 * 1024
        final_ranges: list[tuple[int, int]] = []
        for start, end in ranges:
            final_ranges.extend(_split_for_size(doc, start, end, max_size_bytes))

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem.replace(" ", "_")
        chunks: list[SplitChunk] = []
        for i, (start, end) in enumerate(final_ranges, start=1):
            chunk_path = output_dir / f"{stem}_part_{i:03d}.pdf"
            data = _render_pages(doc, start, end)
            chunk_path.write_bytes(data)
            chunks.append(
                SplitChunk(
                    index=i,
                    start_page=start,
                    end_page=end,
                    output_path=chunk_path,
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to split PDF '{pdf_path}': {exc}") from exc
    finally:
        doc.close()

    result = SplitResult(
        source_path=pdf_path,
        total_pages=total_pages,
        total_size_bytes=total_size_bytes,
        output_dir=output_dir,
        chunks=tuple(chunks),
        manifest_path=manifest_path,
    )
    _write_manifest(result)
    logger.info(
        "Split '%s' (%d pages) into %d chunk(s) in %s",
        pdf_path.name,
        total_pages,
        len(chunks),
        output_dir,
    )
    return result


def _top_level_toc_pages(doc: "pymupdf.Document") -> list[int]:
    """Return sorted, de-duplicated 1-based page numbers of level-1 TOC
    entries, or an empty list if the document has no TOC."""
    try:
        toc = doc.get_toc(simple=True)
    except Exception:  # noqa: BLE001 - TOC is best-effort only
        return []
    pages = sorted({entry[2] for entry in toc if entry[0] == 1 and entry[2] > 0})
    return pages


def _snap_end(fixed_end: int, chunk_start: int, max_pages: int, toc_pages: list[int]) -> int:
    """Best-effort snap of a chunk's fixed end page to the nearest level-1
    TOC page within the snap window, honoring min/max chunk size. Returns
    ``fixed_end`` unchanged if there's no TOC or no safe candidate."""
    if not toc_pages:
        return fixed_end

    candidates = [p for p in toc_pages if abs(p - fixed_end) <= _SNAP_WINDOW_PAGES]
    if not candidates:
        return fixed_end

    best = min(candidates, key=lambda p: abs(p - fixed_end))
    new_length = best - chunk_start + 1
    if _MIN_SNAPPED_CHUNK_PAGES <= new_length <= max_pages:
        return best
    return fixed_end


def _build_chunk_ranges(
    total_pages: int, max_pages: int, overlap_pages: int, toc_pages: list[int]
) -> list[tuple[int, int]]:
    """Compute the primary (pre-size-guard) chunk page ranges, applying
    best-effort bookmark snapping. See ``split_pdf`` docstring for the
    tiling contract this must satisfy."""
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        fixed_end = min(start + max_pages - 1, total_pages)
        if fixed_end >= total_pages:
            end = total_pages  # last chunk always runs to the end; nothing to snap
        else:
            end = _snap_end(fixed_end, start, max_pages, toc_pages)
        ranges.append((start, end))
        if end >= total_pages:
            break
        start = end - overlap_pages + 1
    return ranges


def _assert_tiling(ranges: list[tuple[int, int]], overlap_pages: int) -> None:
    """Assert the primary tiling covers the document with exactly
    ``overlap_pages`` shared pages between consecutive chunks."""
    assert ranges, "split produced zero chunks"
    assert ranges[0][0] == 1, f"first chunk must start at page 1, got {ranges[0]}"
    for i in range(1, len(ranges)):
        prev_end = ranges[i - 1][1]
        cur_start = ranges[i][0]
        shared = prev_end - cur_start + 1
        assert shared == overlap_pages, (
            f"chunk tiling broke the overlap invariant between chunks {i} and {i + 1}: "
            f"expected {overlap_pages} shared pages, got {shared} "
            f"(prev_end={prev_end}, cur_start={cur_start})"
        )


def _render_pages(doc: "pymupdf.Document", start: int, end: int) -> bytes:
    """Render 1-based inclusive page range [start, end] of ``doc`` to a
    standalone PDF byte buffer."""
    sub = pymupdf.open()
    try:
        sub.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
        return sub.tobytes(garbage=4, deflate=True)
    finally:
        sub.close()


def _split_for_size(
    doc: "pymupdf.Document", start: int, end: int, max_size_bytes: int
) -> list[tuple[int, int]]:
    """Recursively halve [start, end] (1-based inclusive) until every piece
    renders under ``max_size_bytes``. Pure size backstop: no overlap is
    added between the halves it produces."""
    if start == end:
        return [(start, end)]  # can't split a single page any further

    size = len(_render_pages(doc, start, end))
    if size <= max_size_bytes:
        return [(start, end)]

    mid = (start + end) // 2
    logger.warning(
        "Chunk pages %d-%d is %.1f MB, exceeding the size limit; splitting into "
        "%d-%d and %d-%d",
        start,
        end,
        size / (1024 * 1024),
        start,
        mid,
        mid + 1,
        end,
    )
    return _split_for_size(doc, start, mid, max_size_bytes) + _split_for_size(
        doc, mid + 1, end, max_size_bytes
    )


def _write_manifest(result: SplitResult) -> None:
    payload = {
        "source_path": str(result.source_path),
        "total_pages": result.total_pages,
        "total_size_bytes": result.total_size_bytes,
        "output_dir": str(result.output_dir),
        "chunks": [
            {**asdict(chunk), "output_path": str(chunk.output_path)} for chunk in result.chunks
        ],
    }
    result.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_manifest(manifest_path: Path) -> SplitResult | None:
    """Best-effort reconstruction of a SplitResult from a previously written
    manifest.json. Returns None if the manifest can't be parsed."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = tuple(
            SplitChunk(
                index=c["index"],
                start_page=c["start_page"],
                end_page=c["end_page"],
                output_path=Path(c["output_path"]),
                size_bytes=c["size_bytes"],
                sha256=c["sha256"],
            )
            for c in payload["chunks"]
        )
        return SplitResult(
            source_path=Path(payload["source_path"]),
            total_pages=payload["total_pages"],
            total_size_bytes=payload["total_size_bytes"],
            output_dir=Path(payload["output_dir"]),
            chunks=chunks,
            manifest_path=manifest_path,
        )
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Could not parse existing manifest %s: %s", manifest_path, exc)
        return None
