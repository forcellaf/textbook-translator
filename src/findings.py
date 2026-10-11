"""The finding type every check reports, split three ways.

Lives apart from ``src.lint`` so that check modules (``src.fidelity``) can
report findings without importing the lint module that runs them.

Severity is how a finding should be acted on:

* **review** -- someone must look (in an unattended run, the orchestrator).
  Gates the build.
* **auto-fix** -- provably safe; ``src.lint.apply_auto_fixes`` repairs it.
* **info** -- context, not a problem. A waived finding becomes info and
  keeps its reason in the message, so it stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How a finding should be acted on."""

    AUTO_FIX = "auto-fix"  # provably safe; `apply_auto_fixes` handles it
    REVIEW = "review"  # someone must look; gates the build
    INFO = "info"  # context, not a problem


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    message: str
    line: int = 0  # 1-based line in `file` (or the linted text), 0 when not line-specific
    context: str = ""
    # The kit file the line is in, e.g. "translated/012.tex"; empty when the
    # finding is about the whole book or a lone file.
    file: str = ""

    @property
    def location(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}" if self.line else self.file
        return f"L{self.line}" if self.line else "-"

    def render(self) -> str:
        tail = f"\n        {self.context}" if self.context else ""
        return f"  [{self.severity.value:>8}] {self.location:>8}  {self.check}: {self.message}{tail}"

    def to_json(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "context": self.context,
        }


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)

    def of(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def needs_review(self) -> bool:
        return any(f.severity is Severity.REVIEW for f in self.findings)

    @property
    def exit_code(self) -> int:
        """Non-zero when anything needs review, so this can gate the build."""
        return 1 if self.needs_review else 0

    def render(self) -> str:
        lines: list[str] = []
        for severity in (Severity.REVIEW, Severity.AUTO_FIX, Severity.INFO):
            group = self.of(severity)
            if not group:
                continue
            lines.append(f"\n{severity.value.upper()} ({len(group)}):")
            lines.extend(f.render() for f in group)

        review = len(self.of(Severity.REVIEW))
        auto = len(self.of(Severity.AUTO_FIX))
        lines.append(
            f"\n{review} finding(s) need review, {auto} auto-fixable, "
            f"{len(self.of(Severity.INFO))} informational."
        )
        if not review:
            lines.append("Nothing blocking; the build can go ahead.")
        return "\n".join(lines)

    def summary(self) -> dict:
        """The machine-readable form: counts, then every review and auto-fix
        finding, then the info ones grouped by check (they are the noisy
        ones, and an orchestrator reading this wants the blocking items)."""
        by_check: dict[str, int] = {}
        for finding in self.of(Severity.INFO):
            by_check[finding.check] = by_check.get(finding.check, 0) + 1
        return {
            "counts": {
                "review": len(self.of(Severity.REVIEW)),
                "auto_fix": len(self.of(Severity.AUTO_FIX)),
                "info": len(self.of(Severity.INFO)),
            },
            "needs_review": self.needs_review,
            "review": [f.to_json() for f in self.of(Severity.REVIEW)],
            "auto_fix": [f.to_json() for f in self.of(Severity.AUTO_FIX)],
            "info_by_check": by_check,
            "files_with_review": sorted({f.file for f in self.of(Severity.REVIEW) if f.file}),
        }


__all__ = ["Finding", "LintReport", "Severity"]
