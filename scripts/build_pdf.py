#!/usr/bin/env python3
"""
Assemble a translated kit and build the PDF.

    python scripts/build_pdf.py <kit-dir>
    python scripts/build_pdf.py <kit-dir> --no-lint
    python scripts/build_pdf.py <kit-dir> --engine xelatex
    python scripts/build_pdf.py <kit-dir> --json --publish

Concatenates the saved LaTeX replies, restores the real image filenames from
``image_map.json``, wraps the result in ``assets/preamble.tex``, lints it, and
runs the LaTeX engine twice.

The lint gate is on by default and refuses to build when anything needs
review. A broken figure, an undefined environment or an untranslated chunk is
far cheaper to fix now than to find one compile error at a time in a 700-page
book.

Every attempt -- gated, failed or built -- is recorded in
``<kit>/build_result.json``, which ``--publish`` (and ``scripts/publish.py``)
reads: a failed build publishes the run log alone, never a PDF. ``--json``
prints a structured summary (the lint counts, assembly problems, and every
LaTeX error located as ``translated/NNN.tex:L``) instead of the report, and
writes the full human-readable report to ``<kit>/logs/build.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.build import DEFAULT_ENGINE, BuildResult, compile_pdf  # noqa: E402
from src.kit import assemble  # noqa: E402
from src.lint import apply_auto_fixes, lint_kit  # noqa: E402
from src.publish import BUILD_RESULT_NAME, publish  # noqa: E402


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
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="afterwards, copy the deliverables to data/output/<book>/ (the run log "
        "alone if the build failed)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not kit_dir.is_dir():
        print(f"error: {kit_dir} is not a directory", file=sys.stderr)
        return 1

    report_text = io.StringIO()
    summary: dict = {"kit": str(kit_dir)}
    with contextlib.redirect_stdout(report_text) if args.json else contextlib.nullcontext():
        code, result = _build(kit_dir, args, summary)

    _record(kit_dir, result, summary)
    if args.publish:
        published = publish(kit_dir.parent)
        summary["published"] = published.to_json()
        if not args.json:
            for path in published.copied:
                print(f"published {path}")
            for problem in published.problems:
                print(f"WARNING: {problem}")

    if args.json:
        log = kit_dir / "logs" / "build.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(report_text.getvalue(), encoding="utf-8")
        summary["log"] = str(log)
        summary["exit_code"] = code
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return code


def _build(kit_dir: Path, args: argparse.Namespace, summary: dict) -> tuple[int, BuildResult]:
    # Lint the fragments, not the assembled file: the preamble legitimately
    # contains the \documentclass and \begin{document} that are defects in a
    # translated chunk.
    report = lint_kit(kit_dir)
    print(f"linting {kit_dir}")
    print(report.render())
    summary["lint"] = report.summary()

    if report.needs_review and not args.no_lint:
        message = "Not building: fix the lint findings, or pass --no-lint to build anyway."
        print(f"\n{message}", file=sys.stderr)
        summary.update(stage="lint", succeeded=False)
        return 1, BuildResult(False, message)

    try:
        assembled, problems = assemble(kit_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        summary.update(stage="assemble", succeeded=False)
        return 1, BuildResult(False, str(exc))

    print(f"\nassembled {assembled} ({len(assembled.read_text(encoding='utf-8')):,} chars)")
    for problem in problems:
        print(f"  WARNING: {problem}")
    summary["assembly_problems"] = problems

    if problems and not args.no_lint:
        message = "Not building: resolve the assembly warnings, or pass --no-lint."
        print(f"\n{message}", file=sys.stderr)
        summary.update(stage="assemble", succeeded=False)
        return 1, BuildResult(False, message)

    if args.fix:
        fixed, count = apply_auto_fixes(assembled.read_text(encoding="utf-8"))
        if count:
            assembled.write_text(fixed, encoding="utf-8")
            print(f"auto-fixed {count} finding(s) in {assembled}")

    result = compile_pdf(assembled, engine=args.engine)
    summary.update(stage="compile", **result.to_json())
    if not result.succeeded:
        print(f"\nerror: {result.message}", file=sys.stderr)
        return 1, result
    summary["pdf"] = str(assembled.with_suffix(".pdf"))
    print(f"\n{result.message}")
    return 0, result


def _record(kit_dir: Path, result: BuildResult, summary: dict) -> None:
    """Every attempt leaves its outcome where publishing can read it."""
    record = {
        "succeeded": result.succeeded,
        "message": result.message[:2000],
        "stage": summary.get("stage"),
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (kit_dir / BUILD_RESULT_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
