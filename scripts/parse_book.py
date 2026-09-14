#!/usr/bin/env python3
"""
Parse a book's PDF with the MinerU cloud API.

    python scripts/parse_book.py <pdf> <work-dir>

Splits the PDF to fit the API's 200-page / 200 MB per-file limits, uploads
every part in one batch, polls with per-file page progress, and merges the
results into ``<work-dir>/merged.md`` plus a flat ``<work-dir>/images/``.

The repo is the tool and books are the data: every book gets its own
``data/work/<book_name>/``, and several coexist in one checkout. Never copy
the repo per book.

Exits non-zero if any part failed to parse, so a partial book is never
mistaken for a complete one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.mineru_api import MinerUError, parse_pdf  # noqa: E402


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
    parser.add_argument("--force", action="store_true", help="re-parse even if merged.md is current")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    try:
        result = parse_pdf(
            args.pdf,
            args.work_dir,
            token=args.token,
            model_version=args.model,
            language=args.language,
            force=args.force,
        )
    except (MinerUError, FileNotFoundError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nwrote {result.merged_path}")
    print(f"      {result.images_dir}")
    print(f"      {result.total_pages} page(s) parsed")

    failed = result.failed_parts
    if failed:
        print(f"\n{len(failed)} part(s) FAILED and are missing from merged.md:", file=sys.stderr)
        for part in failed:
            print(f"  {part.file_name}: {part.state} - {part.err_msg}", file=sys.stderr)
        return 1

    print("\nnext: python scripts/prepare_kit.py " + str(args.work_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
