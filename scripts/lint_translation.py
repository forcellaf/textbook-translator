#!/usr/bin/env python3
"""
Find every defect in a translated kit, in one pass.

    python scripts/lint_translation.py <kit-dir>
    python scripts/lint_translation.py <kit-dir> --fix
    python scripts/lint_translation.py --tex one_chunk.tex
    python scripts/lint_translation.py --markdown merged.md

This is what ends the one-error-per-compile debugging loop: LaTeX reports a
single error per build, so four defects used to cost four full build cycles.
Every check runs at once here, with line numbers and context.

``--tex`` checks a translated fragment (what the model sent back);
``--markdown`` checks the parsed source (what it is about to read).

Exits non-zero when anything needs review, so it can gate the build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.kit import TRANSLATED_DIRNAME  # noqa: E402
from src.lint import apply_auto_fixes, lint_kit, lint_latex, lint_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, nargs="?", help="the translation kit folder")
    parser.add_argument(
        "--tex", type=Path, default=None, help="lint one translated .tex fragment instead"
    )
    parser.add_argument(
        "--markdown", type=Path, default=None, help="lint one source markdown file instead"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply the provably safe repairs in place (doubled commands, "
        "duplicated section/caption numbers, ...). For a kit, the saved replies "
        "in translated/ are repaired",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    if args.markdown is None and args.tex is None and args.kit_dir is None:
        parser.error("give a kit directory, or --tex <file>, or --markdown <file>")

    if args.tex is not None or args.markdown is not None:
        path: Path = args.tex or args.markdown
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        if args.fix:
            text = _autofix(path, text)
        lint = lint_latex if args.tex is not None else lint_markdown
        report = lint(text, base_dir=path.parent)
        print(f"linting {path}")
    else:
        kit_dir: Path = args.kit_dir
        if not kit_dir.is_dir():
            print(f"error: {kit_dir} is not a directory", file=sys.stderr)
            return 1
        if args.fix:
            # Repair the saved replies themselves, not the assembled book: they
            # are what lint_kit reads and what the next build is assembled from,
            # so a fix applied only to translated_book.tex is overwritten by it.
            for fragment in sorted((kit_dir / TRANSLATED_DIRNAME).glob("*.tex")):
                _autofix(fragment, fragment.read_text(encoding="utf-8"))
        report = lint_kit(kit_dir)
        print(f"linting {kit_dir}")

    print(report.render())

    if report.needs_review:
        print("\nFix the findings above, then re-run. The build is gated on this.")
    return report.exit_code


def _autofix(path: Path, text: str) -> str:
    fixed, count = apply_auto_fixes(text)
    if count:
        # newline="\n": on Windows the default rewrites every line ending as
        # CRLF, turning a three-line repair into a whole-file diff.
        path.write_text(fixed, encoding="utf-8", newline="\n")
        print(f"auto-fixed {count} finding(s) in {path}")
    return fixed


if __name__ == "__main__":
    sys.exit(main())
