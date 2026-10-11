#!/usr/bin/env python3
"""
Render page images of a PDF, for an agent deciding the body's page range.

    python scripts/render_pages.py <pdf> <out-dir> --pages 1-40,350-380 --dpi 60
    python scripts/render_pages.py <pdf> <out-dir> --pages 14-17 --dpi 150

Files are ``p0001.png`` ... numbered by **1-based physical page index**,
never the printed page number. ``<out-dir>/pages.json`` records the PDF's
physical page count and, for every page, whether it has a text layer (scans
usually do not, so text extraction cannot replace looking).

60 dpi is a thumbnail for the first pass over the front and back of the
book; 150 dpi is a readable view of the pages either side of a candidate
boundary. Renders are cheap and local: no quota is spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

import pymupdf  # noqa: E402


def parse_ranges(spec: str, page_count: int) -> list[int]:
    """``"1-40,350-380"`` -> sorted 1-based pages, clipped to the PDF.

    A negative start counts from the end: ``"-40"`` is the last 40 pages.
    """
    pages: set[int] = set()
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if part.startswith("-") and part[1:].isdigit():
            count = int(part[1:])
            pages.update(range(max(1, page_count - count + 1), page_count + 1))
            continue
        first, _, last = part.partition("-")
        try:
            start = int(first)
            end = int(last) if last else start
        except ValueError as exc:
            raise ValueError(f"bad page range {part!r}; use e.g. 1-40,350-380") from exc
        if start > end:
            start, end = end, start
        pages.update(range(max(1, start), min(end, page_count) + 1))
    return sorted(pages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--pages", default="1-40,-40", help="ranges to render (default: the first and last 40)"
    )
    parser.add_argument("--dpi", type=int, default=60, help="60 for thumbnails, 150 to read")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as doc:
        count = doc.page_count
        try:
            pages = parse_ranges(args.pages, count)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for number in pages:
            pixmap = doc[number - 1].get_pixmap(dpi=args.dpi)
            pixmap.save(args.out_dir / f"p{number:04d}.png")

        manifest_path = args.out_dir / "pages.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("page_count") != count or "text_layer" not in manifest:
            manifest = {
                "pdf": str(args.pdf),
                "page_count": count,
                "text_layer": {
                    str(i + 1): bool(doc[i].get_text("text").strip()) for i in range(count)
                },
                "rendered": {},
            }
    rendered = manifest.setdefault("rendered", {})
    rendered[str(args.dpi)] = sorted(set(rendered.get(str(args.dpi), [])) | set(pages))
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    layer = sum(1 for has in manifest["text_layer"].values() if has)
    print(
        f"rendered {len(pages)} page(s) of {count} at {args.dpi} dpi into {args.out_dir} "
        f"({layer} of {count} pages have a text layer)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
