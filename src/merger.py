"""
Merge per-chunk MinerU Markdown output (produced by parsing the PDF chunks
that ``src.splitter.split_pdf`` writes) back into a single ``merged.md`` for
a book, with image links rewritten to resolve from the merged file's
location and best-effort deduplication of the pages chunks overlap on.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from src.models import SplitChunk

logger = logging.getLogger(__name__)

_MERGED_NAME = "merged.md"
_MANIFEST_NAME = "manifest.json"

# MinerU emits image references as Markdown `![alt](images/x.jpg)` and,
# occasionally, raw HTML `<img src="images/x.jpg">`. Both forms are
# rewritten identically; only the path inside the reference is touched.
_MD_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\)|\s[^)]*\))")
_HTML_IMAGE_RE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)

# A normalized text block is considered a confident overlap match if it
# has at least this many characters -- shorter matches are too likely to
# be coincidental (e.g. a lone "```" fence or a page number).
_MIN_OVERLAP_MATCH_CHARS = 40


def merge_chunks(chunks: list[SplitChunk], book_work_dir: Path) -> Path:
    """
    Concatenate each chunk's parsed Markdown into ``book_work_dir/merged.md``.

    For each chunk, this looks for ``full.md`` (local MinerU output) or
    ``cloud_full.md`` (MinerU Cloud output) inside
    ``book_work_dir/chunks/<chunk pdf stem>/`` -- i.e. the same per-chunk
    work directory ``src.parser`` parses that chunk's PDF into.

    Image path rewriting
    ----------------------
    MinerU writes image references relative to the chunk's own Markdown
    file (e.g. ``images/xxx.jpg``, relative to
    ``chunks/<chunk_stem>/full.md``). Since ``merged.md`` lives one level up
    (in ``book_work_dir``), every image reference is rewritten to be
    relative to ``book_work_dir`` instead, e.g.
    ``chunks/<chunk_stem>/images/xxx.jpg``. Both Markdown ``![]()`` and
    HTML ``<img src="...">`` forms are handled. References that are already
    absolute paths or full URLs (``http://``, ``https://``, ``data:``) are
    left untouched.

    Overlap dedup (best-effort, never fatal)
    -------------------------------------------
    Each chunk after the first nominally repeats the previous chunk's last
    ``overlap_pages`` pages (see ``src.splitter.split_pdf``). This function
    only attempts dedup between chunks whose page ranges actually overlap
    (``chunks[i-1].end_page >= chunks[i].start_page``) -- chunks that don't
    overlap (e.g. after a size-guard split) are concatenated as-is with no
    marker or warning. For chunks that do overlap, it looks for the tail of
    the previous chunk's content matching the head of the next chunk's
    content (on normalized whitespace). If a confident match is found, the
    duplicated tail is dropped from the EARLIER chunk. If no confident
    match is found (MinerU can OCR the same page slightly differently
    across separate runs), both copies are kept, an
    ``<!-- OVERLAP BOUNDARY: chunk N/M, pages X-Y --> `` comment is inserted
    at the seam, and a warning is logged. This function never raises for a
    dedup mismatch.

    Args:
        chunks: The chunk list from ``SplitResult.chunks``, in document
            order.
        book_work_dir: The book's ``data/work/<book_name>`` directory
            (i.e. the parent of ``chunks/``). ``merged.md`` and
            ``manifest.json`` are written here.

    Returns:
        Path to the written ``merged.md``.

    Raises:
        FileNotFoundError: If ``chunks`` is empty, or a chunk's parsed
            Markdown file cannot be found at all.
    """
    if not chunks:
        raise FileNotFoundError("merge_chunks called with an empty chunk list")

    merged_path = book_work_dir / _MERGED_NAME
    manifest_path = book_work_dir / _MANIFEST_NAME

    chunk_texts: list[str] = []
    for chunk in chunks:
        chunk_dir = book_work_dir / "chunks" / chunk.output_path.stem
        md_path = _find_chunk_markdown(chunk_dir)
        raw_text = md_path.read_text(encoding="utf-8")
        rel_prefix = Path("chunks") / chunk.output_path.stem
        chunk_texts.append(rewrite_image_paths(raw_text, rel_prefix))

    merged_parts: list[str] = [chunk_texts[0]]
    total = len(chunks)
    for i in range(1, total):
        prev_chunk, cur_chunk = chunks[i - 1], chunks[i]
        prev_text, cur_text = merged_parts[-1], chunk_texts[i]

        pages_overlap = prev_chunk.end_page >= cur_chunk.start_page
        if not pages_overlap:
            merged_parts.append(cur_text)
            continue

        deduped_prev, deduped_cur, matched = _dedupe_overlap(prev_text, cur_text)
        if matched:
            merged_parts[-1] = deduped_prev
            merged_parts.append(deduped_cur)
        else:
            logger.warning(
                "Could not confidently locate overlap between chunk %d and %d "
                "(pages %d-%d vs %d-%d); keeping both copies at the seam",
                prev_chunk.index,
                cur_chunk.index,
                prev_chunk.start_page,
                prev_chunk.end_page,
                cur_chunk.start_page,
                cur_chunk.end_page,
            )
            boundary = (
                f"\n<!-- OVERLAP BOUNDARY: chunk {prev_chunk.index}/{total}, "
                f"pages {cur_chunk.start_page}-{prev_chunk.end_page} -->\n"
            )
            merged_parts.append(boundary + cur_text)

    merged_text = "\n\n".join(part.strip("\n") for part in merged_parts) + "\n"
    book_work_dir.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(merged_text, encoding="utf-8")

    _write_manifest(manifest_path, chunks, merged_path)
    logger.info("Merged %d chunk(s) into %s", total, merged_path)
    return merged_path


def _find_chunk_markdown(chunk_dir: Path) -> Path:
    """Locate a chunk's parsed Markdown file, whichever backend produced it."""
    for name in ("full.md", "cloud_full.md"):
        candidate = chunk_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No parsed Markdown (full.md or cloud_full.md) found in {chunk_dir}; "
        "has this chunk been parsed yet?"
    )


def _is_rewritable_ref(ref: str) -> bool:
    return not (
        ref.startswith(("http://", "https://", "data:", "/"))
        or (len(ref) > 1 and ref[1] == ":")  # Windows absolute path, e.g. C:\...
    )


def rewrite_image_paths(markdown_text: str, rel_prefix: Path) -> str:
    """Prefix every relative image reference in ``markdown_text`` with
    ``rel_prefix`` (a POSIX-style relative path from the new file's
    directory to the referenced images' actual directory), leaving
    absolute/URL refs alone. Shared with ``src.chapter_splitter``, which
    uses it to shift merged.md's image refs up one directory level."""

    def _md_sub(match: re.Match[str]) -> str:
        ref = match.group(2)
        if not _is_rewritable_ref(ref):
            return match.group(0)
        new_ref = (rel_prefix / ref).as_posix()
        return f"{match.group(1)}{new_ref}{match.group(3)}"

    def _html_sub(match: re.Match[str]) -> str:
        ref = match.group(2)
        if not _is_rewritable_ref(ref):
            return match.group(0)
        new_ref = (rel_prefix / ref).as_posix()
        return f"{match.group(1)}{new_ref}{match.group(3)}"

    text = _MD_IMAGE_RE.sub(_md_sub, markdown_text)
    text = _HTML_IMAGE_RE.sub(_html_sub, text)
    return text


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace so OCR/MinerU run-to-run formatting jitter
    (extra blank lines, trailing spaces) doesn't block overlap matching."""
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_overlap(prev_text: str, cur_text: str) -> tuple[str, str, bool]:
    """Best-effort: find the previous chunk's tail duplicated at the head of
    the next chunk, and drop it from the previous chunk. Returns
    ``(new_prev_text, cur_text, matched)``; on no confident match, returns
    ``(prev_text, cur_text, False)`` unchanged.

    Strategy: try candidate tail lengths of the previous chunk (from a
    generous chunk of trailing text down to a minimum confident length),
    normalize whitespace, and check whether that normalized tail appears
    at (or very near) the start of the normalized next chunk. The first
    -- longest -- match wins.
    """
    # Search within a generous trailing/leading window; overlap pages are
    # only ever a handful, so a few thousand characters is ample and keeps
    # this cheap even for huge chunks.
    window = 6000
    prev_tail_raw = prev_text[-window:]
    cur_head_raw = cur_text[:window]

    prev_norm = _normalize_for_match(prev_tail_raw)
    cur_norm = _normalize_for_match(cur_head_raw)

    best_len = 0
    # Try progressively shorter candidate lengths so we prefer the longest
    # confident match rather than stopping at the first (shortest) hit.
    for candidate_len in range(min(len(prev_norm), len(cur_norm)), _MIN_OVERLAP_MATCH_CHARS - 1, -1):
        candidate = prev_norm[-candidate_len:]
        if cur_norm.startswith(candidate):
            best_len = candidate_len
            break

    if best_len == 0:
        return prev_text, cur_text, False

    # We matched on normalized text; map back to a raw cut point in
    # prev_text by trimming characters off the end until the normalized
    # remainder no longer contains the matched candidate's normalized form.
    matched_norm = prev_norm[-best_len:]
    new_prev = _trim_matched_tail(prev_text, matched_norm)
    return new_prev, cur_text, True


def _trim_matched_tail(text: str, matched_norm: str) -> str:
    """Remove a trailing slice of raw ``text`` whose normalized form equals
    ``matched_norm``, by trimming characters one at a time from the end
    until the normalized tail no longer starts with the match. This is a
    simple, robust (if not maximally efficient) way to map a match found on
    normalized text back onto the original raw text."""
    target_len = len(matched_norm)
    # Grow the raw trim window until its normalized form is at least as
    # long as the match, then trim precisely to that length.
    trim = 0
    for trim in range(1, len(text) + 1):
        if len(_normalize_for_match(text[-trim:])) >= target_len:
            break
    return text[: len(text) - trim].rstrip()


def _write_manifest(manifest_path: Path, chunks: list[SplitChunk], merged_path: Path) -> None:
    payload = {
        "merged_path": str(merged_path),
        "chunks": [
            {**asdict(chunk), "output_path": str(chunk.output_path)} for chunk in chunks
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
