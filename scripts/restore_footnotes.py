#!/usr/bin/env python3
"""
Report, and optionally restore, the footnotes an existing parse lost.

    python scripts/restore_footnotes.py <work-dir>            # report only
    python scripts/restore_footnotes.py <work-dir> --write    # restore
    python scripts/restore_footnotes.py <work-dir> --json

For a book parsed before ``parse_book.py`` restored ``page_footnote`` blocks
itself (see ``src.footnotes``). Reads ``<work-dir>/merged.md`` and
``<work-dir>/content_list.json``; calls no API and spends no quota.

The report counts the content list's ``page_footnote`` blocks and how many of
them have no text in ``merged.md``, then what a restore would do. ``--write``
copies ``merged.md`` to ``merged.before_footnotes.md`` (once, never
overwritten) and writes the restored text. A book restored earlier, by this
or by hand, is left as it is: text already present is skipped.

Restore before ``prepare_kit.py``: chunks cut from the old ``merged.md`` are
not re-cut by this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.footnotes import missing_footnote_blocks, restore_footnotes  # noqa: E402
from src.mineru_api import CONTENT_LIST_NAME, MERGED_NAME  # noqa: E402

BACKUP_NAME = "merged.before_footnotes.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument("--write", action="store_true", help="write the restored merged.md")
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    merged_path = args.work_dir / MERGED_NAME
    list_path = args.work_dir / CONTENT_LIST_NAME
    for path in (merged_path, list_path):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    markdown = merged_path.read_text(encoding="utf-8")
    entries = [e for e in json.loads(list_path.read_text(encoding="utf-8")) if isinstance(e, dict)]
    lost = missing_footnote_blocks(markdown, entries)
    restored, report = restore_footnotes(markdown, entries)

    written = False
    if args.write and restored != markdown:
        backup = args.work_dir / BACKUP_NAME
        if not backup.exists():
            shutil.copy2(merged_path, backup)
        merged_path.write_text(restored, encoding="utf-8")
        written = True

    summary = {
        "work_dir": str(args.work_dir),
        "page_footnote_blocks": report.footnote_blocks,
        "missing_from_merged": len(lost),
        "missing_pages": sorted({e.get("page_idx") for e in lost if isinstance(e.get("page_idx"), int)}),
        "restore": report.as_dict(),
        "written": written,
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    print(
        f"{report.footnote_blocks} page_footnote block(s) in {list_path.name}, "
        f"{len(lost)} with no text in {merged_path.name}"
    )
    print(f"restore: {report.summary()}")
    if written:
        print(f"wrote {merged_path} (previous version: {args.work_dir / BACKUP_NAME})")
    elif report.changed:
        print("dry run: pass --write to update merged.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
