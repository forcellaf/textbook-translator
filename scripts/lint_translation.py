#!/usr/bin/env python3
"""
Find every defect in a translated kit, in one pass.

    python scripts/lint_translation.py <kit-dir>
    python scripts/lint_translation.py <kit-dir> --fix
    python scripts/lint_translation.py --markdown some.md

This is what ends the one-error-per-compile debugging loop: LaTeX reports a
single error per build, against the generated .tex rather than the markdown,
so four defects used to cost four full build cycles. Every check runs at once
here, against the markdown, with line numbers and context.

Exits non-zero when anything needs review, so it can gate the build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.kit import ASSEMBLED_NAME  # noqa: E402
from src.lint import apply_auto_fixes, lint_kit, lint_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, nargs="?", help="the translation kit folder")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="lint one markdown file instead of a kit"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply the provably safe fixes in place (doubled commands only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    if args.markdown is None and args.kit_dir is None:
        parser.error("give a kit directory, or --markdown <file>")

    if args.markdown is not None:
        path: Path = args.markdown
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        if args.fix:
            text = _autofix(path, text)
        report = lint_markdown(text, base_dir=path.parent)
        print(f"linting {path}")
    else:
        kit_dir: Path = args.kit_dir
        if not kit_dir.is_dir():
            print(f"error: {kit_dir} is not a directory", file=sys.stderr)
            return 1
        if args.fix:
            assembled = kit_dir / ASSEMBLED_NAME
            if assembled.exists():
                _autofix(assembled, assembled.read_text(encoding="utf-8"))
            else:
                print(f"note: --fix needs {assembled}; run scripts/build_pdf.py first")
        report = lint_kit(kit_dir)
        print(f"linting {kit_dir}")

    print(report.render())

    if report.needs_review:
        print("\nFix the findings above, then re-run. The build is gated on this.")
    return report.exit_code


def _autofix(path: Path, text: str) -> str:
    fixed, count = apply_auto_fixes(text)
    if count:
        path.write_text(fixed, encoding="utf-8")
        print(f"auto-fixed {count} doubled command(s) in {path}")
    return fixed


if __name__ == "__main__":
    sys.exit(main())
