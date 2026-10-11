#!/usr/bin/env python3
"""
API mode: translate a kit's untranslated chunks as one Message Batch.

    python scripts/translate_kit.py <kit-dir>
    python scripts/translate_kit.py <kit-dir> --chunks 012,015 --json
    python scripts/translate_kit.py <kit-dir> --wait-minutes 9

Needs ``ANTHROPIC_API_KEY``. Every chunk without ``translated/NNN.tex`` (or
the ``--chunks`` given) goes up in a single batch on the ``translate`` role's
model (``CLAUDE_MODEL_TRANSLATE``, default Claude Haiku 5.5), with the kit's
``system_prompt.txt`` cached across the batch. Replies are written to
``translated/NNN.tex``; a reply cut off at ``max_tokens`` or missing the
``% END-OF-CHUNK`` sentinel is kept as ``NNN.failed.tex`` and reported.

Resumable: the batch id is saved in ``<kit>/translate_batch.json``, and a
re-run polls the same batch rather than submitting another. ``--wait-minutes``
bounds one run's polling. No-API mode does not use this script: there, one
``chunk-translator`` agent translates each chunk.

Exit codes: 0 every submitted chunk translated, 1 some failed (or an error),
4 the batch is still running -- re-run to keep polling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src import config  # noqa: E402
from src.batch_translate import run  # noqa: E402

EXIT_STILL_RUNNING = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, help="the translation kit folder")
    parser.add_argument("--chunks", default=None, help="comma-separated chunk numbers to (re)send")
    parser.add_argument(
        "--force", action="store_true", help="submit a new batch even if one is in flight"
    )
    parser.add_argument("--wait-minutes", type=float, default=9.0, help="poll at most this long")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not (kit_dir / "chunks").is_dir() or not (kit_dir / "system_prompt.txt").exists():
        print(f"error: {kit_dir} is not a prepared kit", file=sys.stderr)
        return 1
    if not config.ANTHROPIC_API_KEY:
        print(
            "error: ANTHROPIC_API_KEY is not set. This is the API-mode translator; in no-API "
            "mode each chunk is translated by a chunk-translator agent.",
            file=sys.stderr,
        )
        return 1

    import anthropic

    chunks = (
        [f"{int(c):03d}" if c.strip().isdigit() else c.strip() for c in args.chunks.split(",")]
        if args.chunks
        else None
    )
    try:
        outcome = run(
            kit_dir,
            client=anthropic.Anthropic(max_retries=config.ANTHROPIC_MAX_RETRIES),
            chunks=chunks,
            force=args.force,
            wait_minutes=args.wait_minutes,
        )
    except anthropic.AnthropicError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    code = (
        EXIT_STILL_RUNNING
        if outcome.state == "running"
        else 1
        if outcome.failed
        else 0
    )
    if args.json:
        print(json.dumps({**outcome.to_json(), "exit_code": code}, indent=2, ensure_ascii=False))
    elif outcome.state == "nothing-to-do":
        print("every chunk already has a translation")
    elif outcome.state == "running":
        print(f"batch {outcome.batch_id} still running ({outcome.counts}); re-run to keep polling")
    else:
        print(f"translated {len(outcome.translated)} chunk(s): {', '.join(outcome.translated)}")
        for chunk, reason in sorted(outcome.failed.items()):
            print(f"  FAILED {chunk}: {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
