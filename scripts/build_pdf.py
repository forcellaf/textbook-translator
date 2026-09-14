#!/usr/bin/env python3
"""
Assemble a translated kit and build the PDF.

    python scripts/build_pdf.py <kit-dir>
    python scripts/build_pdf.py <kit-dir> --no-lint

Concatenates the saved translations, restores the real image filenames from
``image_map.json``, lints the result, and runs pandoc.

The lint gate is on by default and refuses to build when anything needs
review. A broken figure or an untranslated chunk is far cheaper to fix now
than to find in a 700-page PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.build import BuildError, build_pdf  # noqa: E402
from src.kit import assemble, verify_images  # noqa: E402
from src.lint import lint_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, help="the translation kit folder")
    parser.add_argument("-o", "--output", type=Path, default=None, help="default: <kit>/translated_book.pdf")
    parser.add_argument(
        "--no-lint", action="store_true", help="build even when findings need review"
    )
    parser.add_argument(
        "--cjk-font",
        default=None,
        help="e.g. 'Noto Sans CJK SC'. Only needed if source-script text survives "
        "in the output; a fully translated book does not need one.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not kit_dir.is_dir():
        print(f"error: {kit_dir} is not a directory", file=sys.stderr)
        return 1

    try:
        assembled, problems = assemble(kit_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = assembled.read_text(encoding="utf-8")
    print(f"assembled {assembled} ({len(text):,} chars)")
    for problem in problems:
        print(f"  WARNING: {problem}")

    missing = verify_images(text, kit_dir)
    if missing:
        print(f"\nerror: {len(missing)} image reference(s) do not resolve on disk:", file=sys.stderr)
        for ref in missing[:10]:
            print(f"  {ref}", file=sys.stderr)
        return 1

    report = lint_markdown(text, base_dir=kit_dir)
    print(report.render())

    if report.needs_review and not args.no_lint:
        print(
            "\nNot building: fix the findings above, or pass --no-lint to build anyway.",
            file=sys.stderr,
        )
        return 1
    if problems and not args.no_lint:
        print("\nNot building: resolve the assembly warnings, or pass --no-lint.", file=sys.stderr)
        return 1

    output = args.output or (kit_dir / "translated_book.pdf")
    try:
        pdf_path = build_pdf(assembled, output, cjk_font=args.cjk_font)
    except (BuildError, FileNotFoundError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print(f"\nwrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
