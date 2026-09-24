"""Put the project root on ``sys.path`` so ``scripts/*.py`` can ``import src``,
and make the console able to print what these scripts have to print.

``scripts/`` is deliberately not a package: these are things you run, not
things you import. Each script imports this module first.

Why the encoding fix is here and not in one script
--------------------------------------------------
Everything this pipeline reports on is full of characters a Windows console's
default cp1252 cannot encode: Chinese headings, Greek letters, a LaTeX log
quoting the character that broke the build. Printing one of those raises
``UnicodeEncodeError``, which kills the script *in place of* the message it
was trying to show.

That is not a cosmetic failure. A real build died on "Unicode character
ε not set up for use with LaTeX", tried to print that error, and
crashed printing it -- so the run ended with no PDF, no error, and no clue.
The diagnostic must always outlive the thing it is diagnosing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        # Already-wrapped or redirected streams may not support this. `errors`
        # cannot be set independently, so there is no partial fallback worth
        # attempting; the scripts still run, they just cannot print every
        # character.
        pass


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
