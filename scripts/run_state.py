#!/usr/bin/env python3
"""
Read and change an unattended run's state (``<work-dir>/run_state.json``).

    python scripts/run_state.py <work-dir> show [--json]
    python scripts/run_state.py <work-dir> set stage translate
    python scripts/run_state.py <work-dir> set mode claude-code
    python scripts/run_state.py <work-dir> set backend exchange
    python scripts/run_state.py <work-dir> set chunk 012 translated
    python scripts/run_state.py <work-dir> set chunk 013 failed --error "lost 2 image tokens"
    python scripts/run_state.py <work-dir> sync-chunks
    python scripts/run_state.py <work-dir> build ok|failed [--message "..."]
    python scripts/run_state.py <work-dir> decide "chapter 12 starts on p16" --evidence "..."

The orchestrator changes run state only through this script, so every write
is validated and atomic (see ``src/run_state.py``). ``set chunk`` takes
several chunk numbers at once. Each command prints one line, or JSON with
``--json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401  (puts src on sys.path)

from src.run_state import CHUNK_STATUSES, STAGES, load, save  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("work_dir", type=Path, help="this book's data/work/<book_name>/ directory")
    parser.add_argument("--json", action="store_true", help="print the state summary as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("show", help="print the state summary")

    setter = commands.add_parser("set", help="set the stage, mode, backend or a chunk's status")
    setter.add_argument("field", choices=["stage", "mode", "backend", "chunk"])
    setter.add_argument("values", nargs="+", help="the value; for chunk: NNN [NNN ...] STATUS")
    setter.add_argument("--error", default="", help="for a failed chunk: why")

    sync = commands.add_parser("sync-chunks", help="add every chunk of the kit as pending/translated")
    sync.add_argument("--kit-dir", type=Path, default=None, help="default: <work-dir>/kit")

    build = commands.add_parser("build", help="record a build attempt")
    build.add_argument("outcome", choices=["ok", "failed"])
    build.add_argument("--message", default="")

    decide = commands.add_parser("decide", help="record a decision and its evidence")
    decide.add_argument("what")
    decide.add_argument("--evidence", default="")
    decide.add_argument("--stage", default="")
    decide.add_argument("--by", default="agent")

    args = parser.parse_args(argv)
    work_dir: Path = args.work_dir
    if not work_dir.is_dir():
        print(f"error: {work_dir} is not a directory", file=sys.stderr)
        return 1

    state = load(work_dir)
    try:
        if args.command == "set":
            line = _set(state, args.field, args.values, args.error)
        elif args.command == "sync-chunks":
            state.sync_chunks(args.kit_dir or work_dir / "kit")
            line = f"{len(state.chunks)} chunk(s) tracked"
        elif args.command == "build":
            count = state.record_build(args.outcome == "ok", args.message)
            line = f"build attempt {count}: {args.outcome}"
        elif args.command == "decide":
            state.decide(args.what, stage=args.stage, evidence=args.evidence, by=args.by)
            line = f"decision {len(state.decisions)} recorded"
        else:
            line = ""
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command != "show":
        save(work_dir, state)
    if args.json or args.command == "show":
        print(json.dumps(state.summary(), indent=2, ensure_ascii=False))
    else:
        print(line)
    return 0


def _set(state, field: str, values: list[str], error: str) -> str:  # noqa: ANN001
    if field == "chunk":
        if len(values) < 2:
            raise ValueError(f"set chunk needs NNN [NNN ...] STATUS ({', '.join(CHUNK_STATUSES)})")
        *chunks, status = values
        for chunk in chunks:
            state.set_chunk(chunk, status, error)
        return f"{len(chunks)} chunk(s) -> {status}"
    if len(values) != 1:
        raise ValueError(f"set {field} takes one value")
    value = values[0]
    if field == "stage":
        state.set_stage(value)
        return f"stage -> {value} (of {', '.join(STAGES)})"
    setattr(state, field, value)
    return f"{field} -> {value}"


if __name__ == "__main__":
    sys.exit(main())
