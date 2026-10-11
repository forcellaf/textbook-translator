#!/usr/bin/env python3
"""
Size an existing book's figures from a re-parse of the same PDF.

    python scripts/link_figures.py <old-work-dir> <reparse-work-dir>

For a book parsed before MinerU's layout data (``content_list.json``) was
kept. Parse the same PDF again into a **separate** work directory, then run
this: each image of the old book is matched to an image of the re-parse,
and the re-parse's bbox gives it its width.

A different backend may crop images differently, so filenames can change.
Matching tries the exact filename first, then a difference hash among the
re-parse's images near the same reading-order position. Match counts are
printed by method; an unmatched image keeps the default width and is listed
in the review sheet.

Writes ``<old-work-dir>/linked_content_list.json`` (so a later
``prepare_figures.py`` keeps these widths) and updates
``<old-work-dir>/kit/figures.json`` in place. Touches nothing else in the old
book: not ``merged.md``, not the chunks, not the translations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.figures import (  # noqa: E402
    CONTENT_LIST_NAME,
    FIGURES_NAME,
    LINKED_CONTENT_LIST_NAME,
    apply_linked_widths,
    link_images,
    linked_entries,
    load_figures,
    save_figures,
    write_review,
)
from src.kit import IMAGE_MAP_NAME  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("old_work_dir", type=Path, help="the book that is already translated")
    parser.add_argument("new_work_dir", type=Path, help="a fresh parse of the same PDF")
    parser.add_argument("--kit-dir", type=Path, default=None, help="default: <old-work-dir>/kit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    old_dir: Path = args.old_work_dir
    new_dir: Path = args.new_work_dir
    kit_dir: Path = args.kit_dir or (old_dir / "kit")
    map_path = kit_dir / IMAGE_MAP_NAME
    content_path = new_dir / CONTENT_LIST_NAME

    if old_dir.resolve() == new_dir.resolve():
        print("error: the re-parse must be in a separate work directory", file=sys.stderr)
        return 1
    if not map_path.exists():
        print(f"error: {map_path} not found", file=sys.stderr)
        return 1
    if not content_path.exists():
        print(
            f"error: {content_path} not found -- parse the PDF into {new_dir} with "
            "scripts/parse_book.py first",
            file=sys.stderr,
        )
        return 1

    mapping: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    old_names = list(dict.fromkeys(Path(path).name for path in mapping.values()))
    new_entries = json.loads(content_path.read_text(encoding="utf-8"))
    old_images = old_dir / "images" if (old_dir / "images").is_dir() else kit_dir / "images"

    result = link_images(old_names, old_images, new_entries, new_dir / "images")
    counts = result.counts()
    print(
        f"linked {len(old_names)} image(s): {counts['exact']} by exact filename, "
        f"{counts['dhash']} by difference hash, {counts['unmatched']} unmatched"
    )
    for name in result.unmatched[:20]:
        print(f"  unmatched: {name}")
    if len(result.unmatched) > 20:
        print(f"  ... and {len(result.unmatched) - 20} more (see the review sheet)")

    entries = linked_entries(new_entries, result)
    linked_path = old_dir / LINKED_CONTENT_LIST_NAME
    linked_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {linked_path}")

    figures_path = kit_dir / FIGURES_NAME
    if not figures_path.exists():
        print(
            f"\n{figures_path} does not exist yet. Run scripts/prepare_figures.py "
            f"{old_dir}; it reads {LINKED_CONTENT_LIST_NAME} for the widths."
        )
        return 0

    records = load_figures(figures_path)
    measured = apply_linked_widths(records, entries)
    save_figures(figures_path, records)
    review = write_review(kit_dir, records, mapping, old_names)
    print(f"updated {figures_path}: {measured} of {len(records)} record(s) sized from the re-parse")
    print(f"review sheet: {review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
