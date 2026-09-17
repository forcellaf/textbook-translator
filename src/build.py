"""
Assembled LaTeX -> PDF.

``translated_book.tex`` is a complete document: ``assets/preamble.tex``, the
translated body, ``\\end{document}``. There is no conversion step left, so
this module is a thin, careful wrapper around the LaTeX engine.

The three non-obvious decisions:

* **pdflatex, not xelatex.** The preamble uses ``inputenc``, and a finished
  book contains no residual Chinese. If one does, the preamble carries a
  commented two-line swap to fontspec + xeCJK; compile that with
  ``engine="xelatex"``.
* **Two passes.** ``\\tableofcontents`` writes the ``.toc`` on the first pass
  and typesets it on the second. One pass produces a book with an empty
  contents page and no error.
* **``-interaction=nonstopmode``.** Translated LaTeX contains whatever the
  model wrote. An undefined macro stops an interactive run and waits forever
  for keyboard input that an automated build will never provide.

Failures come back as a value, not an exception: the caller almost always
wants to print the error lines and move on, and a LaTeX log is tens of
thousands of lines of which about six matter.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENGINE = "pdflatex"

# A full textbook is a long run; 30 minutes is generous but finite.
BUILD_TIMEOUT_SECONDS = 1800

# The lines worth reading in a LaTeX log: the error itself ('!'), the source
# line it points at ('l.123 ...'), and the fatal/emergency endings.
_LOG_ERROR_RE = re.compile(r"^(!.*|l\.\d+.*|.*Fatal error.*|.*Emergency stop.*)$", re.MULTILINE)
_MAX_ERROR_LINES = 40


def log_excerpt(log_path: Path) -> str:
    """Pull just the error lines out of a LaTeX log.

    A ``.log`` for a real book runs to tens of thousands of lines; what the
    caller needs is the handful starting with '!' and their line references.
    Falls back to the tail when the log has no recognizable error lines,
    since something still went wrong.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"(no readable log at {log_path})"

    matches = _LOG_ERROR_RE.findall(log_text)
    if not matches:
        return "\n".join(log_text.splitlines()[-_MAX_ERROR_LINES:])
    return "\n".join(matches[:_MAX_ERROR_LINES])


def build_pdf(
    tex_path: Path,
    *,
    engine: str = DEFAULT_ENGINE,
    passes: int = 2,
    timeout_seconds: int = BUILD_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Compile ``tex_path`` to a PDF beside it.

    Runs in the ``.tex`` file's own directory so the preamble's
    ``\\graphicspath{{images/}{./}}`` finds the figures.

    Returns:
        ``(succeeded, message)``. A missing engine, a missing source file, a
        timeout and a LaTeX error all come back as ``(False, <explanation>)``
        rather than raising -- see the module docstring.
    """
    if not tex_path.exists():
        return False, f"LaTeX source not found: {tex_path}"
    if shutil.which(engine) is None:
        return (
            False,
            f"LaTeX engine {engine!r} is not on PATH. Install TeX Live or MiKTeX "
            f"(and make sure {engine} is on PATH), or pass a different engine.",
        )

    work_dir = tex_path.parent
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    log_path = tex_path.with_suffix(".log")
    passes = max(1, passes)

    for attempt in range(1, passes + 1):
        logger.info("Running %s (pass %d/%d) on %s", engine, attempt, passes, tex_path.name)
        try:
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"{engine} timed out after {timeout_seconds // 60} minutes on pass "
                f"{attempt}/{passes}.",
            )
        except OSError as exc:
            return False, f"Could not run {engine}: {exc}"

        if result.returncode != 0:
            return (
                False,
                f"{engine} failed on pass {attempt}/{passes} (exit "
                f"{result.returncode}):\n{log_excerpt(log_path)}",
            )

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        return (
            False,
            f"{engine} reported success but no PDF was produced:\n{log_excerpt(log_path)}",
        )

    logger.info("Wrote %s", pdf_path)
    return True, f"PDF written to {pdf_path}"


__all__ = ["BUILD_TIMEOUT_SECONDS", "DEFAULT_ENGINE", "build_pdf", "log_excerpt"]
