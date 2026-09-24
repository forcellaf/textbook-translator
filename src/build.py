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

``-halt-on-error`` keeps a failed build fast, but it means one error per
build: a real book needed four builds to find four defects. So when a build
fails, ``collect_errors`` compiles once more *without* it and reports every
error in the book, each mapped back to the translated fragment and line it
came from.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENGINE = "pdflatex"

# A full textbook is a long run; 30 minutes is generous but finite.
BUILD_TIMEOUT_SECONDS = 1800

# The lines worth reading in a LaTeX log: the error itself ('!'), the source
# line it points at ('l.123 ...'), and the fatal/emergency endings.
_LOG_ERROR_RE = re.compile(r"^(!.*|l\.\d+.*|.*Fatal error.*|.*Emergency stop.*)$", re.MULTILINE)
_MAX_ERROR_LINES = 40

# The comment `src.kit.assemble` writes above each translated fragment:
# `% >>> translated/008.tex:3` -- the fragment, and the line its body starts on.
_CHUNK_MARKER_PREFIX = "% >>> "
_CHUNK_MARKER_RE = re.compile(r"^% >>> (\S+):(\d+)$")

# One error in a LaTeX log: the `! ...` message, then (within a few lines)
# `l.<n> <text read so far>`. The "==> Fatal error occurred" and "Emergency
# stop" lines are consequences of an earlier error, not errors of their own.
_LOG_LINE_RE = re.compile(r"^l\.(\d+) ?(.*)$")
_LOG_LINE_LOOKAHEAD = 12
_MAX_ERRORS = 60


def chunk_marker(label: str, first_line: int) -> str:
    """The comment placed above a fragment in the assembled book."""
    return f"{_CHUNK_MARKER_PREFIX}{label}:{first_line}"


@dataclass(frozen=True)
class LatexError:
    """One error from a LaTeX log, located in the file it came from."""

    message: str
    location: str  # "translated/008.tex:39", or "translated_book.tex:12" in the preamble
    context: str = ""  # the source text LaTeX had read when it stopped

    def render(self) -> str:
        tail = f"  <- {self.context}" if self.context else ""
        return f"  {self.location}: {self.message}{tail}"


def _locate(tex_lines: list[str], tex_name: str, line: int) -> str:
    """Map a line of the assembled book back to its translated fragment."""
    for index in range(min(line, len(tex_lines)) - 1, -1, -1):
        match = _CHUNK_MARKER_RE.match(tex_lines[index])
        if match:
            # `index` is 0-based and the fragment body starts one line below
            # its marker, so the marker sits at 1-based line `index + 1`.
            return f"{match.group(1)}:{int(match.group(2)) + line - index - 2}"
    return f"{tex_name}:{line}"


def parse_log_errors(log_text: str, tex_path: Path) -> list[LatexError]:
    """Every distinct error in ``log_text``, located via the chunk markers
    in ``tex_path``."""
    try:
        tex_lines = tex_path.read_text(encoding="utf-8").split("\n")
    except OSError:
        tex_lines = []

    log_lines = log_text.split("\n")
    errors: list[LatexError] = []
    seen: set[tuple[str, str]] = set()
    for index, line in enumerate(log_lines):
        if not line.startswith("! ") or "==> Fatal error" in line or "Emergency stop" in line:
            continue
        message = line[2:].strip()
        location, context = tex_path.name, ""
        for follow in log_lines[index + 1 : index + 1 + _LOG_LINE_LOOKAHEAD]:
            found = _LOG_LINE_RE.match(follow)
            if found:
                location = _locate(tex_lines, tex_path.name, int(found.group(1)))
                context = found.group(2).strip()
                break
        if (message, location) in seen:
            continue
        seen.add((message, location))
        errors.append(LatexError(message, location, context))
    return errors


def collect_errors(
    tex_path: Path,
    *,
    engine: str = DEFAULT_ENGINE,
    timeout_seconds: int = BUILD_TIMEOUT_SECONDS,
) -> list[LatexError]:
    """Compile once without ``-halt-on-error`` and return every error.

    Runs in the ``.tex`` file's directory (for ``\\graphicspath``) but writes
    its log and aux files to a temporary directory, so the failed build's
    own files are left exactly as they were. Best-effort: anything that goes
    wrong here returns an empty list, and the caller falls back to the
    failed build's own log.
    """
    with tempfile.TemporaryDirectory(prefix="latex-probe-") as out_dir:
        command = [engine, "-interaction=nonstopmode", f"-output-directory={out_dir}"]
        if engine == "pdflatex":
            command.append("-draftmode")  # no PDF is written, which is faster
        command.append(tex_path.name)
        logger.info("Collecting every error: one %s run without -halt-on-error", engine)
        try:
            subprocess.run(
                command,
                cwd=tex_path.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            log_text = (Path(out_dir) / tex_path.with_suffix(".log").name).read_text(
                encoding="utf-8", errors="replace"
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
    return parse_log_errors(log_text, tex_path)


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
    collect_all_errors: bool = True,
) -> tuple[bool, str]:
    """Compile ``tex_path`` to a PDF beside it.

    Runs in the ``.tex`` file's own directory so the preamble's
    ``\\graphicspath{{images/}{./}}`` finds the figures. On a LaTeX error,
    ``collect_all_errors`` runs one more pass to list every error rather
    than the first (see ``collect_errors``).

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
            failed = f"{engine} failed on pass {attempt}/{passes} (exit {result.returncode})"
            errors = (
                collect_errors(tex_path, engine=engine, timeout_seconds=timeout_seconds)
                if collect_all_errors
                else []
            )
            if not errors:
                return False, f"{failed}:\n{log_excerpt(log_path)}"
            shown = "\n".join(error.render() for error in errors[:_MAX_ERRORS])
            more = (
                f"\n  ... and {len(errors) - _MAX_ERRORS} more" if len(errors) > _MAX_ERRORS else ""
            )
            return (
                False,
                f"{failed}. Every error in the book, from one run without "
                f"-halt-on-error ({len(errors)}):\n{shown}{more}",
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
