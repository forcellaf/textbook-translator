#!/usr/bin/env python3
"""
Copy a book's deliverables to ``data/output/<book>/``.

    python scripts/publish.py data/work/<book>
    python scripts/publish.py data/work/<book> --json

Publishes ``kit/translated_book.pdf`` as ``<book>_translated.pdf``,
``kit/translated_book.tex`` as ``<book>_translated.tex`` and the run log
(``<work-dir>/run_log.md``). Copies, never moves.

If the last build failed, only the run log is published, and a PDF left over
from an earlier run is renamed ``*.stale.pdf``. Also available as
``scripts/build_pdf.py --publish``. Exits 1 when the PDF was not published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.publish import publish  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if not args.work_dir.is_dir():
        print(f"error: {args.work_dir} is not a directory", file=sys.stderr)
        return 1
    result = publish(args.work_dir)
    if args.json:
        print(json.dumps(result.to_json(), indent=2, ensure_ascii=False))
    else:
        for path in result.copied:
            print(f"published {path}")
        for path in result.stale:
            print(f"renamed a previous run's file to {path}")
        for problem in result.problems:
            print(f"WARNING: {problem}")
    return 0 if result.built else 1


if __name__ == "__main__":
    sys.exit(main())
