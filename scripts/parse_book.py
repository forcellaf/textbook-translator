#!/usr/bin/env python3
"""
Parse a book's PDF with the MinerU cloud API.

    python scripts/parse_book.py <pdf> <work-dir>
    python scripts/parse_book.py <pdf> <work-dir> --page-range <work-dir>/page_range.json
    python scripts/parse_book.py <pdf> <work-dir> --wait-minutes 9 --json

Splits the PDF to fit the API's 200-page / 200 MB per-file limits, uploads
every part in one batch, polls with per-file page progress, and merges the
results into ``<work-dir>/merged.md`` plus a flat ``<work-dir>/images/``.

**Only the body.** With a page range (``--page-range``, or
``<work-dir>/page_range.json`` when it exists) the PDF is first cut to pages
``start..end`` into ``<work-dir>/source_body.pdf``, and only that is sent:
front matter and non-content back matter are where parsing fails, and they
cost quota. The page offset is recorded in ``parts/manifest.json`` so page
numbers can be traced back to the original.

**Resumable.** The batch id is saved in ``<work-dir>/parse_batch.json`` as
soon as the upload finishes. Re-running the same command while the batch is
still in flight resumes polling it -- it never uploads again and never
spends quota twice. ``--wait-minutes`` bounds one run's polling, so the
command can be re-invoked from a shell with a time limit. Uploading again
takes ``--force``.

A book whose ``merged.md`` is newer than the PDF is not parsed again, even
when a page range appears later: re-parsing costs quota. The summary says
whether the existing parse was trimmed.

Exit codes: 0 parsed (or already parsed), 1 failure (including a failed
part: a partial book is never mistaken for a complete one), 4 the batch is
still running -- re-run to keep polling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src import config  # noqa: E402
from src.mineru_api import MinerUError, batch_in_flight, parse_pdf  # noqa: E402
from src.page_range import (  # noqa: E402
    PAGE_RANGE_NAME,
    SOURCE_BODY_NAME,
    first_heading_warning,
    load_page_range,
    page_offset,
    record_offset,
    trim_pdf,
    was_trimmed,
)

EXIT_STILL_RUNNING = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="the book's PDF")
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument(
        "--model",
        default=None,
        choices=["vlm", "pipeline"],
        help="MinerU backend (default: MINERU_MODEL_VERSION, itself defaulting to vlm)",
    )
    parser.add_argument("--language", default=None, help="OCR language hint (default: ch)")
    parser.add_argument("--token", default=None, help="overrides MINERU_TOKEN")
    parser.add_argument(
        "--page-range",
        type=Path,
        default=None,
        help=f"parse only the body pages in this file (default: <work-dir>/{PAGE_RANGE_NAME} "
        "if it exists)",
    )
    parser.add_argument(
        "--wait-minutes",
        type=float,
        default=None,
        help="poll for at most this long, then exit 4 if still running "
        f"(default: MINERU_TIMEOUT_MINUTES = {config.MINERU_TIMEOUT_MINUTES})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-parse even if merged.md is current, and upload again even if a batch is in "
        "flight (spends quota)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    work_dir: Path = args.work_dir
    summary: dict = {"work_dir": str(work_dir), "pdf": str(args.pdf)}

    def finish(code: int, state: str, message: str = "") -> int:
        summary.update(state=state, exit_code=code)
        if message:
            summary["message"] = message
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        elif message:
            print(message, file=sys.stderr if code else sys.stdout)
        return code

    if not args.pdf.exists():
        return finish(1, "failed", f"error: {args.pdf} not found")

    merged = work_dir / "merged.md"
    if (
        not args.force
        and merged.exists()
        and merged.stat().st_mtime > args.pdf.stat().st_mtime
    ):
        summary.update(_parsed_summary(work_dir))
        return finish(0, "skipped", f"already parsed: {merged} (pass --force to re-parse)")

    # ── The page range ───────────────────────────────────────────────────
    range_path = args.page_range or (work_dir / PAGE_RANGE_NAME)
    page_range = None
    if args.page_range is not None or range_path.exists():
        try:
            page_range = load_page_range(range_path, page_count=_page_count(args.pdf))
        except (FileNotFoundError, ValueError) as exc:
            return finish(1, "failed", f"error: {exc}")
    to_parse = args.pdf
    if page_range is not None:
        to_parse = work_dir / SOURCE_BODY_NAME
        trimmed = trim_pdf(args.pdf, page_range, to_parse, stamp=range_path)
        summary["page_range"] = {"start": page_range.start, "end": page_range.end}
        print(
            f"body: pages {page_range.start}-{page_range.end} of {args.pdf.name} "
            f"({page_range.page_count} pages) -> {to_parse}"
            + ("" if trimmed else " (already cut)"),
            file=sys.stderr if args.json else sys.stdout,
        )

    # ── Parse ────────────────────────────────────────────────────────────
    try:
        result = parse_pdf(
            to_parse,
            work_dir,
            token=args.token,
            model_version=args.model,
            language=args.language,
            force=args.force,
            wait_minutes=args.wait_minutes,
        )
    except TimeoutError as exc:
        batch = batch_in_flight(work_dir)
        summary["batch_id"] = batch
        if batch:
            return finish(
                EXIT_STILL_RUNNING,
                "running",
                f"still running: {exc}\nre-run the same command to keep polling batch {batch}",
            )
        return finish(1, "failed", f"error: {exc}")
    except (MinerUError, FileNotFoundError, RuntimeError, ValueError) as exc:
        return finish(1, "failed", f"error: {exc}")

    record_offset(work_dir, page_range, args.pdf)
    summary.update(_parsed_summary(work_dir))
    summary["pages_parsed"] = result.total_pages
    summary["quota_pages_per_day"] = config.MINERU_DAILY_PAGE_QUOTA
    if result.footnotes is not None:
        summary["footnotes"] = result.footnotes.as_dict()

    if not args.json:
        print(f"\nwrote {result.merged_path}")
        print(f"      {result.images_dir}")
        print(f"      {result.total_pages} page(s) parsed")
        if result.footnotes is not None and result.footnotes.footnote_blocks:
            print(f"      footnotes: {result.footnotes.summary()}")
        if summary.get("first_heading_warning"):
            print(f"\nWARNING: {summary['first_heading_warning']}")

    failed = result.failed_parts
    if failed:
        summary["failed_parts"] = [
            {"file": p.file_name, "state": p.state, "error": p.err_msg} for p in failed
        ]
        lines = [f"{len(failed)} part(s) FAILED and are missing from merged.md:"]
        lines += [f"  {p.file_name}: {p.state} - {p.err_msg}" for p in failed]
        return finish(1, "partial", "\n".join(lines))

    return finish(0, "parsed", "" if args.json else f"\nnext: python scripts/prepare_kit.py {work_dir}")


def _parsed_summary(work_dir: Path) -> dict:
    """What a resuming orchestrator needs about an existing parse."""
    merged = work_dir / "merged.md"
    text = merged.read_text(encoding="utf-8") if merged.exists() else ""
    return {
        "merged": str(merged),
        "merged_chars": len(text),
        "trimmed": was_trimmed(work_dir),
        "page_offset": page_offset(work_dir),
        "first_heading_warning": first_heading_warning(text) if text else None,
    }


def _page_count(pdf: Path) -> int:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return doc.page_count


if __name__ == "__main__":
    sys.exit(main())
