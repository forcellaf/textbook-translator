#!/usr/bin/env python3
"""
Turn a parsed book into a translation kit.

    python scripts/prepare_kit.py <work-dir>
    python scripts/prepare_kit.py <work-dir> --classify-dry-run --json

Stages, in order -- the order matters:

  1. normalize   merge split headings; report anything else short
  2. profile     one cached LLM call: glossary + a level per heading
  3. re-level    apply that classification (pure, no LLM)
  4. structure   parts / chapters / readings: Claude labels the whole outline
                 from the numbering under each heading, code cross-checks it
                 and follows up on conflicts. Headings become final \\part /
                 \\chapter / \\chapter* commands, so the translator never guesses
  5. tables      HTML -> pipe tables; warn and skip degenerate ones
  6. figures     kit/figures.json: each image's source number and width; Claude
                 vision labels the uncaptioned ones, and a confident decoration
                 is dropped from the text before the translator sees it
  7. tokenize    image paths -> IMG_nnnn
  8. chunk       ~30,000 chars on paragraph boundaries

Normalizing before profiling is not optional: headings are matched by their
exact text, so classifying ``## 第13章`` and then merging it into
``## 第13章 电势`` would leave the classification unusable.

Reads ``<work-dir>/merged.md`` and writes ``<work-dir>/kit/``.

No-API mode (``LLM_BACKEND=exchange``): an LLM call with no answer yet becomes
a request file under ``<work-dir>/llm/``. The run stops after the first stage
that left requests -- profile, structure or figures, in that order, because
each one's prompts depend on the decisions before it -- writes
``llm/pending.json`` and exits with code 3. Answer the requests and re-run;
answered calls are free on the next run.

``--json`` prints a structured digest instead of the report (merged headings,
the outline with every ``<-- CHECK`` row, warnings, figure and table counts,
the chunk count, pending requests) and writes the full report to
``<work-dir>/logs/prepare_kit.txt``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.config import KIT_CHUNK_CHARS  # noqa: E402
from src.figures import (  # noqa: E402
    FIGURES_NAME,
    MODE_CLASSIFY,
    MODE_DRY_RUN,
    MODE_OFF,
    REVIEW_NAME,
    figures_step,
    strip_dropped_images,
    write_review,
)
from src.kit import build_kit  # noqa: E402
from src.latex import load_preamble, preamble_first_chapter, preamble_for  # noqa: E402
from src.llm.exchange import (  # noqa: E402
    EXIT_PENDING,
    has_pending,
    pending_requests,
    pending_summary,
    write_pending,
)
from src.normalize import normalize  # noqa: E402
from src.page_range import PAGE_RANGE_NAME, load_page_range  # noqa: E402
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
        help="ask Claude again about headings it already decided (manual and "
        "agent entries in structure.json are always kept)",
    )
    parser.add_argument(
        "--fix-math-spacing",
        action="store_true",
        help="also trim whitespace inside inline $ delimiters (off by default: the "
        "current parser produces none)",
    )
    classify = parser.add_mutually_exclusive_group()
    classify.add_argument(
        "--no-classify",
        action="store_true",
        help="no vision calls: keep every uncaptioned image and list it for review",
    )
    classify.add_argument(
        "--classify-dry-run",
        action="store_true",
        help="classify uncaptioned images and fill in the review sheet, but drop nothing",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="ignore the labels cached in figures.json and ask the vision model again "
        "(after a prompt or model change); manual and agent entries are never touched",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON digest instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.reclassify and args.no_classify:
        parser.error("--reclassify needs the vision model; drop --no-classify")

    configure_logging(args.verbose)

    summary: dict = {"work_dir": str(args.work_dir)}
    report = io.StringIO()
    with contextlib.redirect_stdout(report) if args.json else contextlib.nullcontext():
        code = _prepare(args, summary)

    if args.json:
        log = args.work_dir / "logs" / "prepare_kit.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(report.getvalue(), encoding="utf-8")
        summary.update(
            exit_code=code,
            pending=pending_requests(args.work_dir),
            log=str(log),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return code


def _prepare(args: argparse.Namespace, summary: dict) -> int:
    work_dir: Path = args.work_dir
    merged_path = work_dir / "merged.md"
    if not merged_path.exists():
        print(f"error: {merged_path} not found -- run scripts/parse_book.py first", file=sys.stderr)
        summary["error"] = f"{merged_path} not found"
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
    summary["normalize"] = {
        "merged_headings": [
            {"line": c.line, "kind": c.kind, "before": c.before, "after": c.after}
            for c in result.changes
        ],
        "notes": [
            {"line": n.line, "kind": n.kind, "text": n.text, "detail": n.detail}
            for n in review_notes
        ],
        "short_headings": len(short),
    }

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
            + (f" [by {profile.by}]" if profile.by else "")
        )
        text, counts = apply_heading_levels(text, profile)
        print(
            f"  headings: {counts['relevelled']} re-levelled, "
            f"{counts['unchanged']} unchanged, {counts['demoted']} demoted to body text"
        )
        summary["profile"] = {
            "subject": profile.subject,
            "subfield": profile.subfield,
            "glossary_terms": len(profile.glossary),
            "classified_headings": len(profile.heading_levels),
            "by": profile.by,
            "headings": dict(counts),
        }
        if has_pending(work_dir):
            return _stop_for_answers(work_dir, "profile", summary)

    # ── 4. structure ─────────────────────────────────────────────────────
    llm = None
    if not args.no_profile:
        from src.config import ROLE_STRUCTURE
        from src.llm.factory import get_llm

        try:
            llm = get_llm(ROLE_STRUCTURE, work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001 - rules alone still decide most headings
            print(f"\nstructure: no LLM available ({exc}); using the numbering rules only")
    preamble_chapter = preamble_first_chapter(load_preamble(preamble_for(work_dir / "kit")))
    report = analyze_structure(
        text,
        llm=llm,
        cache_path=work_dir / STRUCTURE_FILENAME,
        force=args.force_structure,
        first_chapter=preamble_chapter,
    )
    kinds = report.counts()
    print(
        f"\nstructure: {kinds['part']} part(s), {kinds['chapter']} chapter(s), "
        f"{kinds['reading']} reading(s), {kinds['other']} front/back matter, "
        f"{kinds['undecided']} undecided; {report.llm_calls} LLM call(s)"
    )
    outline = render_outline(report)
    print(outline)
    warnings = list(report.warnings) + _page_range_chapter_warning(work_dir, preamble_chapter)
    for warning in warnings:
        print(f"  WARNING: {warning}")
    text, rewritten = apply_structure(text, report)
    print(
        f"  rewrote: {rewritten['summary item']} summary item(s) back to paragraphs, "
        f"{rewritten['unnumbered section']} section(s) of unnumbered chapters as \\section*"
    )
    print(f"  check the outline above; correct it in {work_dir / STRUCTURE_FILENAME}")
    summary["structure"] = {
        "counts": {k: kinds[k] for k in ("part", "chapter", "reading", "other", "undecided")},
        "llm_calls": report.llm_calls,
        "outline": outline.split("\n"),
        "check_rows": [
            {"line": d.line, "title": d.title, "kind": d.kind, "number": d.number, "flag": d.flag}
            for d in report.divisions
            if d.flag
        ],
        "warnings": warnings,
        "preamble_first_chapter": preamble_chapter,
    }
    if has_pending(work_dir):
        return _stop_for_answers(work_dir, "structure", summary)

    # ── 5. tables ────────────────────────────────────────────────────────
    text, tables = convert_html_tables(text)
    converted = sum(1 for r in tables if r.converted)
    print(f"\ntables: {converted} of {len(tables)} converted to pipe tables")
    for table in tables:
        if not table.converted:
            print(f"  L{table.line:>6} table {table.index} left as HTML: {table.reason}")
    summary["tables"] = {
        "total": len(tables),
        "converted": converted,
        "left_as_html": [
            {"line": t.line, "index": t.index, "reason": t.reason} for t in tables if not t.converted
        ],
    }

    # ── 6. figures ───────────────────────────────────────────────────────
    kit_dir = args.kit_dir or (work_dir / "kit")
    mode = (
        MODE_OFF
        if args.no_classify
        else MODE_DRY_RUN
        if args.classify_dry_run
        else MODE_CLASSIFY
    )
    figures, note = figures_step(
        text, work_dir=work_dir, kit_dir=kit_dir, mode=mode, reclassify=args.reclassify
    )
    print(f"\nfigures ({mode}): {figures.render()}")
    if note:
        print(f"  {note}")
    text, removed = strip_dropped_images(text, figures.records)
    if removed:
        print(f"  removed {removed} dropped image reference(s) from the text")
    print(f"  correct any of them in {kit_dir / FIGURES_NAME} (set \"by\": \"manual\" or \"agent\")")
    records = list(figures.records.values())
    summary["figures"] = {
        "mode": mode,
        "images": len(records),
        "captioned": sum(1 for r in records if r.number),
        "panels": sum(1 for r in records if r.panel_of),
        "candidates": sum(1 for r in records if not r.number and not r.panel_of),
        "dropped": len(figures.dropped),
        "would_drop": len(figures.would_drop),
        "flagged": len(figures.flagged),
        "default_width": len(figures.defaulted),
        "labels": dict(figures.label_counts()),
        "classifier_calls": figures.calls,
        "awaiting_answers": figures.pending,
        "note": note,
        "review_sheet": str(kit_dir / REVIEW_NAME),
    }
    if has_pending(work_dir):
        return _stop_for_answers(work_dir, "figures", summary)
    write_pending(work_dir)  # nothing pending: clears a stale manifest

    (work_dir / PREPARED_NAME).write_text(text, encoding="utf-8")

    # ── 7-8. tokenize + chunk ────────────────────────────────────────────
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
    review = write_review(kit_dir, figures.records, kit.image_map, figures.order)
    print(f"  figure review sheet: {review}")
    print(f"  translate {kit_dir / 'chunks'}/NNN.md with {kit_dir / 'system_prompt.txt'} as the")
    print(f"  system prompt, one fresh conversation each, into {kit_dir / 'translated'}/NNN.tex")
    print(f"\nnext: python scripts/lint_translation.py {kit_dir}")
    summary["kit"] = {
        "dir": str(kit_dir),
        "chunks": kit.chunk_count,
        "chunk_chars": budget,
        "image_tokens": len(kit.image_map),
    }
    return 0


def _page_range_chapter_warning(work_dir: Path, preamble_chapter: int) -> list[str]:
    """The page-range finder read the first chapter's number off the page;
    if the preamble starts elsewhere, every chapter is misnumbered."""
    path = work_dir / PAGE_RANGE_NAME
    if not path.exists():
        return []
    try:
        page_range = load_page_range(path)
    except (FileNotFoundError, ValueError) as exc:
        return [f"{path} is unusable: {exc}"]
    first = page_range.first_chapter_number
    if first is None or first == preamble_chapter:
        return []
    return [
        f"{PAGE_RANGE_NAME} says the first chapter is {first} (printed on the page) but the "
        f"preamble starts at {preamble_chapter}: create kit/preamble.tex from "
        f"assets/preamble.tex with \\setcounter{{chapter}}{{{first - 1}}}"
    ]


def _stop_for_answers(work_dir: Path, stage: str, summary: dict) -> int:
    """End a no-API round: record the requests, say so, exit 3."""
    write_pending(work_dir)
    summary["stopped_after"] = stage
    print(f"\nstopped after {stage}: {pending_summary(work_dir)}")
    return EXIT_PENDING


if __name__ == "__main__":
    sys.exit(main())
