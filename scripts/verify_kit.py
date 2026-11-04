#!/usr/bin/env python3
"""
Collect the meaning verifiers' results, and prepare their page checks.

    python scripts/verify_kit.py <kit-dir> report [--json]
    python scripts/verify_kit.py <kit-dir> pages [--json] [--work-dir <dir>]

The agents do the reading (see ``src/verify.py``): ``chunk-verifier`` writes
``<kit>/verify/NNN.json`` and fixes the translation in place;
``page-verifier`` writes ``<kit>/verify/NNN.pages.json``. This script never
calls a model.

``report`` validates every result file, looks each quoted excerpt up in the
file it claims to come from, and prints what the run log needs: chunks not
verified yet, every fix, every finding left, page checks pending and decided.
With ``--json`` the full report goes to ``<kit>/logs/verify.txt``. It exits
1 only when a result file must be rewritten (``problems``); findings never
fail it -- verification is reported in the log, it does not gate the build.

``pages`` finds, for every pending page check, the page(s) of the parsed PDF
showing the source excerpt (through MinerU's ``content_list.json``), renders
them to ``<kit>/verify/pages/pNNNN.png`` (original physical page numbers),
and writes ``<kit>/verify/page_checks.json``: the tasks, grouped by chunk,
that the page verifiers are spawned with. Renders are local; no quota is
spent. ``--work-dir`` defaults to the kit's parent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import configure_logging  # noqa: E402  (must precede src imports)

from src.verify import prepare_page_checks, verification_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kit_dir", type=Path, help="the translation kit folder")
    parser.add_argument("command", choices=["report", "pages"])
    parser.add_argument("--work-dir", type=Path, default=None, help="default: the kit's parent")
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    kit_dir: Path = args.kit_dir
    if not kit_dir.is_dir():
        print(f"error: {kit_dir} is not a directory", file=sys.stderr)
        return 1

    if args.command == "pages":
        work_dir: Path = args.work_dir or kit_dir.parent
        payload = prepare_page_checks(kit_dir, work_dir)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            for task in payload["tasks"]:
                pages = ", ".join(str(p) for p in task["pages"]) or "-"
                print(
                    f"chunk {task['chunk']} finding {task['finding']}: p. {pages} "
                    f"({task['locate']}) {task['note']}".rstrip()
                )
            print(f"{len(payload['tasks'])} page check(s) in {len(payload['chunks'])} chunk(s)")
        return 0

    summary, text = verification_report(kit_dir)
    exit_code = 1 if summary["problems"] else 0
    if args.json:
        log_dir = kit_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "verify.txt"
        log.write_text(text, encoding="utf-8")
        print(json.dumps({**summary, "log": str(log)}, indent=2, ensure_ascii=False))
    else:
        print(text, end="")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
