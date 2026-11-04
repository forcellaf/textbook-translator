#!/usr/bin/env python3
"""
Figures with Chinese text inside: list them, render the English versions, report.

    python scripts/figure_text.py <kit-dir> tasks [--json]
    python scripts/figure_text.py <kit-dir> render <stem> [--json]
    python scripts/figure_text.py <kit-dir> report [--json]

The classifier (``src/figures.py``) records, for every kept image, whether
Chinese text is printed inside it. The ``diagram-translator`` agent redraws
each such figure in TikZ, overlays English labels on the scan, or adds an
English legend under it (see ``src/figure_text.py``). This script never
calls a model.

``tasks`` writes ``<kit>/figure_text/tasks.json``: one task per image, with
its figure number, the book width to draw at, and the translated and source
text around it (for the book's own terms). ``todo`` lists the stems still to
do; ``not_asked`` the kept images nobody asked about.

``render <stem>`` compiles ``<kit>/figure_text/<stem>.json`` (writing the
``.tex`` for an overlay or legend; taking the agent's for TikZ) into
``<stem>.pdf`` and a ``<stem>.preview.png`` for the agent to look at. Exit 1
when it did not compile; the errors are in the JSON.

``report`` re-lists the tasks with their outcome for the run log (with
``--json`` the text goes to ``<kit>/logs/figure_text.txt``). Assembly uses an
English PDF only when its spec is accepted and unchanged since a successful
render; everything else falls back to the original scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.figure_text import build_tasks, figure_text_report, render  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, help="the translation kit folder")
    parser.add_argument("command", choices=["tasks", "render", "report"])
    parser.add_argument("stem", nargs="?", help="for render: the image's file name without extension")
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not kit_dir.is_dir():
        print(f"error: {kit_dir} is not a directory", file=sys.stderr)
        return 1

    if args.command == "tasks":
        payload = build_tasks(kit_dir)
        if args.json:
            # The contexts are for the agent, which reads tasks.json itself.
            brief = [
                {k: v for k, v in t.items() if k not in ("translated_context", "source_context")}
                for t in payload["tasks"]
            ]
            print(json.dumps({**payload, "tasks": brief}, indent=2, ensure_ascii=False))
        else:
            for task in payload["tasks"]:
                print(f"{task['status']:<8} {task['stem']}  figure {task['number'] or '-'}")
            print(f"{len(payload['todo'])} to do of {len(payload['tasks'])}")
        return 0

    if args.command == "render":
        if not args.stem:
            parser.error("render needs the image's stem")
        result = render(kit_dir, args.stem)
        if args.json:
            print(json.dumps(result.to_json(), indent=2, ensure_ascii=False))
        else:
            print(f"{args.stem}: {'ok' if result.ok else 'FAILED'}")
            for error in result.errors:
                print(f"  {error}")
            if result.preview:
                print(f"  preview: {result.preview}")
        return 0 if result.ok else 1

    summary, text = figure_text_report(kit_dir)
    if args.json:
        log_dir = kit_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "figure_text.txt"
        log.write_text(text, encoding="utf-8")
        print(json.dumps({**summary, "log": str(log)}, indent=2, ensure_ascii=False))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
