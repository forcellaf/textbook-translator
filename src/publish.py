"""
Publish a run's deliverables to ``data/output/<book>/``.

The kit (``data/work/<book>/kit/``) is a workshop: chunks, fragments, logs,
images, aux files. What a reader wants is three files, in one obvious place:

``<book>_translated.pdf``  the book
``<book>_translated.tex``  the assembled LaTeX -- small, and enough to fix a
                           typo and rebuild without re-running anything
``run_log.md``             every decision, assumption and uncertainty of the
                           unattended run that produced them

Files are *copied*, never moved: the build still needs everything in the
kit, and a later rebuild publishes again.

A failed build publishes the run log alone. A PDF from a failed build is
never published -- and a PDF published by an *earlier* successful run is
renamed ``*.stale.pdf`` rather than left beside a log that says the build
failed, so nobody mistakes it for this run's result. Nothing is deleted.

Whether the build succeeded is read from ``kit/build_result.json``, which
``scripts/build_pdf.py`` writes after every attempt, and cross-checked
against the files: the PDF must exist and be newer than the ``.tex`` it was
built from.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DATA_OUTPUT

RUN_LOG_NAME = "run_log.md"
BUILD_RESULT_NAME = "build_result.json"
ASSEMBLED_TEX = "translated_book.tex"
ASSEMBLED_PDF = "translated_book.pdf"


@dataclass
class PublishResult:
    out_dir: Path
    built: bool
    copied: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "out_dir": str(self.out_dir),
            "built": self.built,
            "copied": [str(p) for p in self.copied],
            "stale": [str(p) for p in self.stale],
            "problems": self.problems,
        }


def build_succeeded(kit_dir: Path) -> tuple[bool, str]:
    """``(succeeded, why)`` for the kit's last build."""
    record = kit_dir / BUILD_RESULT_NAME
    pdf, tex = kit_dir / ASSEMBLED_PDF, kit_dir / ASSEMBLED_TEX
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, f"no readable {record.name}: the book was not built by build_pdf.py"
    if not data.get("succeeded"):
        return False, f"the last build failed ({str(data.get('message', ''))[:200]})"
    if not pdf.exists():
        return False, f"{pdf} is missing"
    if tex.exists() and pdf.stat().st_mtime < tex.stat().st_mtime:
        return False, f"{pdf.name} is older than {tex.name}: it is not from the last build"
    return True, "built"


def publish(work_dir: Path, *, output_root: Path = DATA_OUTPUT) -> PublishResult:
    """Copy the deliverables of ``work_dir`` to ``output_root/<book>/``."""
    book = work_dir.name
    kit = work_dir / "kit"
    out = output_root / book
    out.mkdir(parents=True, exist_ok=True)

    built, why = build_succeeded(kit)
    result = PublishResult(out, built)

    log = work_dir / RUN_LOG_NAME
    if log.exists():
        shutil.copy2(log, out / RUN_LOG_NAME)
        result.copied.append(out / RUN_LOG_NAME)
    else:
        result.problems.append(f"{log} does not exist: no run log was published")

    targets = {
        kit / ASSEMBLED_PDF: out / f"{book}_translated.pdf",
        kit / ASSEMBLED_TEX: out / f"{book}_translated.tex",
    }
    if built:
        for source, target in targets.items():
            if source.exists():
                shutil.copy2(source, target)
                result.copied.append(target)
    else:
        result.problems.append(f"not publishing the PDF: {why}")
        for target in targets.values():
            if target.exists():
                stale = target.with_name(f"{target.stem}.stale{target.suffix}")
                target.replace(stale)
                result.stale.append(stale)
    return result


__all__ = ["BUILD_RESULT_NAME", "RUN_LOG_NAME", "PublishResult", "build_succeeded", "publish"]
