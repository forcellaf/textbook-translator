#!/usr/bin/env python3
"""
API mode launcher: translate one book with the Claude Agent SDK.

    uv sync --group agent
    uv run --group agent python scripts/run_agent_sdk.py data/input/physics.pdf [book_name]

The same orchestrator as no-API mode -- the ``/translate-book`` skill, the
same subagents in ``.claude/agents/``, the same project settings -- run by
the Agent SDK on an ``ANTHROPIC_API_KEY`` instead of by Claude Code on a
subscription. The differences are in the environment:

* ``LLM_BACKEND=anthropic``: stage 2's LLM calls (profile, structure, figure
  classification) are made by Python through the Messages API, so the
  ``stage2-preparer`` sees no pending exchange requests;
* translation runs as one Message Batch through ``scripts/translate_kit.py``
  (the skill picks this from the backend) instead of one subagent per chunk.

The run settings overlay (``.claude/run-settings.json``) denies edits to the
tool itself for the duration of the run. Every SDK message is appended to
``logs/translate-<book>-<timestamp>.jsonl``; the deliverables land in
``data/output/<book>/``.

The SDK is an optional dependency group (``agent``) so that no-API mode does
not need it.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _book_name(pdf: Path) -> str:
    return pdf.stem.lower().replace(" ", "_")


def _record(message: object) -> dict:
    """A JSON-able record of one SDK message, for the transcript."""
    if dataclasses.is_dataclass(message):
        data = dataclasses.asdict(message)  # type: ignore[arg-type]
    else:
        data = {"repr": repr(message)}
    return {"type": type(message).__name__, **data}


async def run(pdf: Path, book: str) -> int:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    except ImportError:
        print(
            "error: the Agent SDK is not installed; run `uv sync --group agent`",
            file=sys.stderr,
        )
        return 1

    options = ClaudeAgentOptions(
        model="claude-opus-5-5",
        effort="high",
        cwd=str(ROOT),
        # "project" loads .claude/settings.json, CLAUDE.md, .claude/skills
        # and .claude/agents -- the same skill and subagents as no-API mode.
        setting_sources=["project"],
        skills="all",
        settings=str(ROOT / ".claude" / "run-settings.json"),
        permission_mode="dontAsk",
        env={"LLM_BACKEND": "anthropic"},
    )

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    transcript = logs / f"translate-{book}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"translating {pdf} as '{book}'; transcript: {transcript}", file=sys.stderr)

    outcome = 1
    with transcript.open("w", encoding="utf-8") as log:
        async for message in query(prompt=f"/translate-book {pdf} {book}", options=options):
            log.write(json.dumps(_record(message), ensure_ascii=False, default=str) + "\n")
            log.flush()
            if isinstance(message, ResultMessage):
                outcome = 0 if message.subtype == "success" and not message.is_error else 1
                if message.result:
                    print(message.result)
    print(f"done (exit {outcome}); output in data/output/{book}/", file=sys.stderr)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("book_name", nargs="?", default=None)
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.path.insert(0, str(ROOT))
        from src import config  # loads .env

        if not config.ANTHROPIC_API_KEY:
            print(
                "error: ANTHROPIC_API_KEY is not set. API mode needs it; for a Claude Code "
                "subscription use scripts/translate_unattended.sh instead.",
                file=sys.stderr,
            )
            return 1
        os.environ["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
    return asyncio.run(run(args.pdf, args.book_name or _book_name(args.pdf)))


if __name__ == "__main__":
    sys.exit(main())
