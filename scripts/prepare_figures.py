#!/usr/bin/env python3
"""
Build figures.json for a book whose kit already exists.

    python scripts/prepare_figures.py <work-dir>
    python scripts/prepare_figures.py <work-dir> --classify-dry-run
    python scripts/prepare_figures.py <work-dir> --no-classify

For a book prepared before the figures step existed. Reads the kit's source
chunks and ``image_map.json`` -- that is where the figure numbers come from
-- and writes ``kit/figures.json`` and ``kit/figures_review.html``. It does
not touch ``merged.md``, the chunks or the translations: a dropped figure is
removed at build time, so existing translations stay valid.

Widths come from ``<work-dir>/content_list.json`` when the book was parsed
with it, or from ``linked_content_list.json`` written by
``scripts/link_figures.py``. Without either, every figure gets the default
width and is listed in the review sheet.

Re-running is cheap: classifier labels are cached in figures.json, and
entries marked ``"by": "manual"`` are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.figures import (  # noqa: E402
    MODE_CLASSIFY,
    MODE_DRY_RUN,
    MODE_OFF,
    figures_step,
    write_review,
)
from src.kit import CHUNKS_DIRNAME, IMAGE_MAP_NAME, restore_images  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument("--kit-dir", type=Path, default=None, help="default: <work-dir>/kit")
    classify = parser.add_mutually_exclusive_group()
    classify.add_argument(
        "--no-classify",
        action="store_true",
        help="no vision calls: keep every uncaptioned image and list it for review",
    )
    classify.add_argument(
        "--classify-dry-run",
        action="store_true",
        help="classify uncaptioned images and fill in the review sheet, but drop nothing",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="ignore the labels cached in figures.json and ask the vision model again "
        "(after a prompt or model change); manual entries are never touched",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.reclassify and args.no_classify:
        parser.error("--reclassify needs the vision model; drop --no-classify")

    configure_logging(args.verbose)

    work_dir: Path = args.work_dir
    kit_dir: Path = args.kit_dir or (work_dir / "kit")
    chunks = sorted((kit_dir / CHUNKS_DIRNAME).glob("*.md"))
    map_path = kit_dir / IMAGE_MAP_NAME
    if not chunks or not map_path.exists():
        print(
            f"error: {kit_dir} has no chunks or no {IMAGE_MAP_NAME} -- run "
            "scripts/prepare_kit.py first",
            file=sys.stderr,
        )
        return 1

    mapping: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    tokenized = "\n\n".join(path.read_text(encoding="utf-8") for path in chunks)
    text, unknown = restore_images(tokenized, mapping)
    if unknown:
        print(f"warning: {len(unknown)} unknown image token(s) in the chunks: {unknown[:5]}")

    mode = (
        MODE_OFF
        if args.no_classify
        else MODE_DRY_RUN
        if args.classify_dry_run
        else MODE_CLASSIFY
    )
    report, note = figures_step(
        text, work_dir=work_dir, kit_dir=kit_dir, mode=mode, reclassify=args.reclassify
    )
    print(f"figures ({mode}): {report.render()}")
    if note:
        print(f"  {note}")
    review = write_review(kit_dir, report.records, mapping, report.order)
    print(f"\nwrote {kit_dir / 'figures.json'}")
    print(f"review sheet: {review}")
    print(f"next: python scripts/build_pdf.py {kit_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
