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

``--json`` prints a structured summary instead of the report -- counts by
severity, every review and auto-fix finding with its ``translated/NNN.tex``
and line, info findings counted by check -- and writes the full report to
``<kit>/logs/lint.txt`` for when the detail is needed.

Per-book additions live in the kit: ``known_commands.txt`` (commands that are
real LaTeX this book uses) and ``lint_waivers.json`` (findings judged false
positives, each with its reason; a waived finding is shown as info).

Exits non-zero when anything needs review, so it can gate the build.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    say = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

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
            text = _autofix(path, text, say)
        lint = lint_latex if args.tex is not None else lint_markdown
        report = lint(text, base_dir=path.parent)
        say(f"linting {path}")
        log_dir = path.parent
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
                _autofix(fragment, fragment.read_text(encoding="utf-8"), say)
        report = lint_kit(kit_dir)
        say(f"linting {kit_dir}")
        log_dir = kit_dir / "logs"

    rendered = report.render()
    if args.json:
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "lint.txt"
        log.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({**report.summary(), "log": str(log)}, indent=2, ensure_ascii=False))
        return report.exit_code

    print(rendered)
    if report.needs_review:
        print("\nFix the findings above, then re-run. The build is gated on this.")
    return report.exit_code


def _autofix(path: Path, text: str, say=print) -> str:  # noqa: ANN001
    fixed, count = apply_auto_fixes(text)
    if count:
        # newline="\n": on Windows the default rewrites every line ending as
        # CRLF, turning a three-line repair into a whole-file diff.
        path.write_text(fixed, encoding="utf-8", newline="\n")
        say(f"auto-fixed {count} finding(s) in {path}")
    return fixed


if __name__ == "__main__":
    sys.exit(main())
