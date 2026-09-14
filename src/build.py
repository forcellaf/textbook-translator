"""
Assembled Markdown -> PDF, via pandoc and XeLaTeX.

The flag set below is the one that produced a complete 723-page book; the
non-obvious parts:

* ``-V classoption=openany`` -- without it the ``book`` class starts every
  chapter on a recto page, padding the output with blank versos.
* ``-H assets/header.tex`` -- figure sizing and float placement, see that file.
* ``--resource-path`` -- image references are relative to the markdown, and
  pandoc resolves them relative to its own working directory otherwise.

pandoc is invoked as an external binary on purpose: it is the one dependency
this project does not install through Python.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from src.config import ASSETS_DIR

logger = logging.getLogger(__name__)

HEADER_TEX = ASSETS_DIR / "header.tex"

# A full textbook is a long XeLaTeX run; 30 minutes is generous but finite.
_BUILD_TIMEOUT_SECONDS = 1800

# Excerpt of pandoc's stderr on failure. LaTeX prints its error near the end
# of a very long log, so the tail is what matters.
_LOG_TAIL_CHARS = 3000


class BuildError(RuntimeError):
    """Raised when pandoc is missing, times out, or fails to build the PDF."""


def pandoc_command(
    markdown_path: Path,
    output_path: Path,
    *,
    resource_dirs: list[Path] | None = None,
    cjk_font: str | None = None,
    header: Path | None = None,
) -> list[str]:
    """Build the pandoc argv. Separated out so tests can assert on the flags
    without needing pandoc installed."""
    header = header or HEADER_TEX
    resources = resource_dirs or [markdown_path.parent, markdown_path.parent / "images"]

    command = [
        "pandoc",
        str(markdown_path),
        "-o",
        str(output_path),
        "--pdf-engine=xelatex",
        "--toc",
        "-V",
        "documentclass=book",
        # Removes the blank verso before every chapter.
        "-V",
        "classoption=openany",
        "-V",
        "geometry:margin=2.5cm",
    ]
    if cjk_font:
        # Only needed while source-script text survives in the output; a fully
        # translated book does not need a CJK font at all.
        command += ["-V", f"CJKmainfont={cjk_font}"]
    if header.exists():
        command += ["-H", str(header)]
    command += ["--resource-path", _resource_path(resources)]
    return command


def _resource_path(dirs: list[Path]) -> str:
    """pandoc's --resource-path separator is ':' on POSIX and ';' on Windows."""
    return os.pathsep.join(str(d) for d in dirs)


def build_pdf(
    markdown_path: Path,
    output_path: Path | None = None,
    *,
    cjk_font: str | None = None,
) -> Path:
    """Run pandoc over ``markdown_path`` and return the PDF path.

    Raises:
        BuildError: If pandoc is not installed, times out, or exits non-zero.
            The tail of its output is included -- LaTeX reports its error near
            the end of a very long log.
    """
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown not found: {markdown_path}")
    if shutil.which("pandoc") is None:
        raise BuildError(
            "pandoc is not on PATH. Install it from "
            "https://pandoc.org/installing.html (it is the one dependency this "
            "project does not install through Python)."
        )

    output_path = output_path or markdown_path.with_suffix(".pdf")
    command = pandoc_command(markdown_path, output_path, cjk_font=cjk_font)
    logger.info("Running: %s", " ".join(command))

    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        raise BuildError(
            f"pandoc timed out after {_BUILD_TIMEOUT_SECONDS // 60} minutes"
        ) from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-_LOG_TAIL_CHARS:]
        raise BuildError(
            f"pandoc exited {proc.returncode}. Last {_LOG_TAIL_CHARS} characters "
            f"of its output:\n{tail}"
        )

    logger.info("Wrote %s", output_path)
    return output_path


__all__ = ["HEADER_TEX", "BuildError", "build_pdf", "pandoc_command"]
