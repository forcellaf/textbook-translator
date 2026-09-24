#!/usr/bin/env python3
"""
Turn a parsed book into a translation kit.

    python scripts/prepare_kit.py <work-dir>

Stages, in order -- the order matters:

  1. normalize   merge split headings; report anything else short
  2. profile     one cached LLM call: glossary + a level per heading
  3. re-level    apply that classification (pure, no LLM)
  4. structure   parts / chapters / readings: DeepSeek labels the whole outline
                 from the numbering under each heading, code cross-checks it
                 and follows up on conflicts. Headings become final \\part /
                 \\chapter / \\chapter* commands, so the translator never guesses
  5. tables      HTML -> pipe tables; warn and skip degenerate ones
  6. tokenize    image paths -> IMG_nnnn
  7. chunk       ~30,000 chars on paragraph boundaries

Normalizing before profiling is not optional: headings are matched by their
exact text, so classifying ``## 第13章`` and then merging it into
``## 第13章 电势`` would leave the classification unusable.

Reads ``<work-dir>/merged.md`` and writes ``<work-dir>/kit/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.config import KIT_CHUNK_CHARS  # noqa: E402
from src.kit import build_kit  # noqa: E402
from src.latex import load_preamble, preamble_first_chapter  # noqa: E402
from src.normalize import normalize  # noqa: E402
from src.profiler import apply_heading_levels, profile_book  # noqa: E402
from src.structure import (  # noqa: E402
    STRUCTURE_FILENAME,
    analyze_structure,
    apply_structure,
    render_outline,
)
from src.tables import convert_html_tables  # noqa: E402

NORMALIZED_NAME = "normalized.md"
PREPARED_NAME = "prepared.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument("--kit-dir", type=Path, default=None, help="default: <work-dir>/kit")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=None,
        help=f"chunk budget in source characters (default: {KIT_CHUNK_CHARS:,})",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the LLM call; headings keep the levels the parser gave them",
    )
    parser.add_argument("--force-profile", action="store_true", help="ignore the cached profile")
    parser.add_argument(
        "--force-structure",
        action="store_true",
        help="ask DeepSeek again about headings it already decided (manual "
        "entries in structure.json are always kept)",
    )
    parser.add_argument(
        "--fix-math-spacing",
        action="store_true",
        help="also trim whitespace inside inline $ delimiters (off by default: the "
        "current parser produces none)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    work_dir: Path = args.work_dir
    merged_path = work_dir / "merged.md"
    if not merged_path.exists():
        print(f"error: {merged_path} not found -- run scripts/parse_book.py first", file=sys.stderr)
        return 1

    text = merged_path.read_text(encoding="utf-8")
    print(f"source: {merged_path} ({len(text):,} chars)\n")

    # ── 1. normalize ─────────────────────────────────────────────────────
    result = normalize(text, fix_math_spacing=args.fix_math_spacing)
    print(f"normalize: {result.summary()}")
    for change in result.changes:
        print(f"  L{change.line:>6} {change.kind}: {change.before!r} -> {change.after!r}")
    review_notes = [n for n in result.notes if n.kind != "short-heading"]
    for note in review_notes:
        print(f"  L{note.line:>6} {note.kind}: {note.text!r} -- {note.detail}")
    short = [n for n in result.notes if n.kind == "short-heading"]
    if short:
        print(f"  {len(short)} short heading(s) left unchanged (use -v to list them)")
        for note in short if args.verbose else []:
            print(f"  L{note.line:>6} short-heading: {note.text!r}")
    print(f"  diagnostics: {result.diagnostics}")

    normalized_path = work_dir / NORMALIZED_NAME
    normalized_path.write_text(result.text, encoding="utf-8")
    text = result.text

    # ── 2-3. profile + re-level ──────────────────────────────────────────
    profile = None
    if args.no_profile:
        print("\nprofile: skipped (--no-profile); heading levels left as parsed")
    else:
        profile = profile_book(normalized_path, work_dir, force=args.force_profile)
        print(
            f"\nprofile: {profile.subject}/{profile.subfield or '-'}, "
            f"{len(profile.glossary)} glossary term(s), "
            f"{len(profile.heading_levels)} classified heading(s)"
        )
        text, counts = apply_heading_levels(text, profile)
        print(
            f"  headings: {counts['relevelled']} re-levelled, "
            f"{counts['unchanged']} unchanged, {counts['demoted']} demoted to body text"
        )

    # ── 4. structure ─────────────────────────────────────────────────────
    llm = None
    if not args.no_profile:
        from src.llm.factory import get_llm

        try:
            llm = get_llm()
        except Exception as exc:  # noqa: BLE001 - rules alone still decide most headings
            print(f"\nstructure: no LLM available ({exc}); using the numbering rules only")
    report = analyze_structure(
        text,
        llm=llm,
        cache_path=work_dir / STRUCTURE_FILENAME,
        force=args.force_structure,
        first_chapter=preamble_first_chapter(load_preamble()),
    )
    kinds = report.counts()
    print(
        f"\nstructure: {kinds['part']} part(s), {kinds['chapter']} chapter(s), "
        f"{kinds['reading']} reading(s), {kinds['other']} front/back matter, "
        f"{kinds['undecided']} undecided; {report.llm_calls} DeepSeek call(s)"
    )
    print(render_outline(report))
    for warning in report.warnings:
        print(f"  WARNING: {warning}")
    text, rewritten = apply_structure(text, report)
    print(
        f"  rewrote: {rewritten['summary item']} summary item(s) back to paragraphs, "
        f"{rewritten['unnumbered section']} section(s) of unnumbered chapters as \\section*"
    )
    print(f"  check the outline above; correct it in {work_dir / STRUCTURE_FILENAME}")

    # ── 5. tables ────────────────────────────────────────────────────────
    text, reports = convert_html_tables(text)
    converted = sum(1 for r in reports if r.converted)
    print(f"\ntables: {converted} of {len(reports)} converted to pipe tables")
    for report in reports:
        if not report.converted:
            print(f"  L{report.line:>6} table {report.index} left as HTML: {report.reason}")

    (work_dir / PREPARED_NAME).write_text(text, encoding="utf-8")

    # ── 6-7. tokenize + chunk ────────────────────────────────────────────
    kit_dir = args.kit_dir or (work_dir / "kit")
    kit = build_kit(
        text,
        kit_dir,
        images_src=work_dir / "images",
        profile=profile,
        **({"max_chars": args.chunk_chars} if args.chunk_chars else {}),
    )

    budget = args.chunk_chars or KIT_CHUNK_CHARS
    print(
        f"\nkit: {kit.chunk_count} chunk(s) of up to {budget:,} chars, "
        f"{len(kit.image_map)} image token(s) in {kit_dir}"
    )
    print(f"  paste {kit_dir / 'system_prompt.txt'} as the system prompt")
    print(f"  send  {kit_dir / 'chunks'}/NNN.md one at a time, one fresh conversation each")
    print(f"  save each reply (LaTeX) to {kit_dir / 'translated'}/NNN.tex")
    print(f"\nnext: python scripts/lint_translation.py {kit_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
