#!/usr/bin/env python3
"""
Assemble a translated kit and build the PDF.

    python scripts/build_pdf.py <kit-dir>
    python scripts/build_pdf.py <kit-dir> --no-lint
    python scripts/build_pdf.py <kit-dir> --engine xelatex

Concatenates the saved LaTeX replies, restores the real image filenames from
``image_map.json``, wraps the result in ``assets/preamble.tex``, lints it, and
runs the LaTeX engine twice.

The lint gate is on by default and refuses to build when anything needs
review. A broken figure, an undefined environment or an untranslated chunk is
far cheaper to fix now than to find one compile error at a time in a 700-page
book.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.build import DEFAULT_ENGINE, build_pdf  # noqa: E402
from src.kit import assemble  # noqa: E402
from src.lint import apply_auto_fixes, lint_kit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, help="the translation kit folder")
    parser.add_argument(
        "--no-lint", action="store_true", help="build even when findings need review"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply the provably safe repairs to the assembled .tex first "
        "(doubled commands and duplicated section/caption numbers)",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help=f"LaTeX engine (default: {DEFAULT_ENGINE}). Use xelatex only with the "
        "CJK swap described in assets/preamble.tex.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not kit_dir.is_dir():
        print(f"error: {kit_dir} is not a directory", file=sys.stderr)
        return 1

    # Lint the fragments, not the assembled file: the preamble legitimately
    # contains the \documentclass and \begin{document} that are defects in a
    # translated chunk.
    report = lint_kit(kit_dir)
    print(f"linting {kit_dir}")
    print(report.render())

    if report.needs_review and not args.no_lint:
        print(
            "\nNot building: fix the findings above, or pass --no-lint to build anyway.",
            file=sys.stderr,
        )
        return 1

    try:
        assembled, problems = assemble(kit_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nassembled {assembled} ({len(assembled.read_text(encoding='utf-8')):,} chars)")
    for problem in problems:
        print(f"  WARNING: {problem}")

    if problems and not args.no_lint:
        print("\nNot building: resolve the assembly warnings, or pass --no-lint.", file=sys.stderr)
        return 1

    if args.fix:
        fixed, count = apply_auto_fixes(assembled.read_text(encoding="utf-8"))
        if count:
            assembled.write_text(fixed, encoding="utf-8")
            print(f"auto-fixed {count} finding(s) in {assembled}")

    succeeded, message = build_pdf(assembled, engine=args.engine)
    if not succeeded:
        print(f"\nerror: {message}", file=sys.stderr)
        return 1

    print(f"\n{message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
