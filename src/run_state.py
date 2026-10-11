"""
``data/work/<book>/run_state.json``: where an unattended run is, so it can resume.

An unattended translation runs for hours across several stages, any of which
can be interrupted -- a crash, a closed laptop, a session limit. The
orchestrator reads this file first and resumes from ``stage``, re-checking
that the last completed stage's outputs exist before trusting it. It holds:

``stage``           the stage being worked on (``STAGES``)
``chunks``          per chunk: ``pending`` / ``translated`` / ``failed``, the
                    attempts made and the last error -- what decides which
                    chunks the next translation pass sends out
``build_attempts``  each build's outcome, so the five-attempt limit survives
                    a restart
``decisions``       every decision so far, with its evidence: the raw
                    material of the run log

It is small and machine-written on purpose. The orchestrator changes it
through ``scripts/run_state.py``, never by editing JSON by hand, so a
half-written edit cannot corrupt the one file a resume depends on: every
write goes to a temporary file first and replaces the old one atomically.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_NAME = "run_state.json"

STAGES: tuple[str, ...] = (
    "preflight",
    "page-range",
    "parse",
    "prepare",
    "translate",
    "lint",
    "build",
    "publish",
    "done",
    "stopped",
)
CHUNK_STATUSES: tuple[str, ...] = ("pending", "translated", "failed", "gap")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ChunkState:
    status: str = "pending"
    attempts: int = 0
    last_error: str = ""


@dataclass
class RunState:
    book: str
    stage: str = "preflight"
    mode: str = ""  # "claude-code" (no API) | "agent-sdk" (API)
    backend: str = ""  # "exchange" | "anthropic"
    started: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    chunks: dict[str, ChunkState] = field(default_factory=dict)
    build_attempts: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    # ── Changes ─────────────────────────────────────────────────────────

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
        self.stage = stage

    def set_chunk(self, chunk: str, status: str, error: str = "") -> ChunkState:
        """Record a chunk's outcome. A translation attempt -- one that ends
        ``translated`` or ``failed`` -- counts toward ``attempts``."""
        if status not in CHUNK_STATUSES:
            raise ValueError(
                f"unknown chunk status {status!r}; expected one of {', '.join(CHUNK_STATUSES)}"
            )
        state = self.chunks.setdefault(_chunk_id(chunk), ChunkState())
        if status in ("translated", "failed"):
            state.attempts += 1
        state.status = status
        state.last_error = error if status == "failed" else ""
        return state

    def record_build(self, succeeded: bool, message: str = "") -> int:
        self.build_attempts.append({"at": _now(), "succeeded": succeeded, "message": message})
        return len(self.build_attempts)

    def decide(self, what: str, *, stage: str = "", evidence: str = "", by: str = "agent") -> None:
        self.decisions.append(
            {"at": _now(), "stage": stage or self.stage, "what": what, "evidence": evidence, "by": by}
        )

    def sync_chunks(self, kit_dir: Path) -> None:
        """Add every chunk of the kit; a chunk with a saved translation and
        no recorded state counts as translated."""
        for source in sorted((kit_dir / "chunks").glob("*.md")):
            if source.stem in self.chunks:
                continue
            done = (kit_dir / "translated" / f"{source.stem}.tex").exists()
            self.chunks[source.stem] = ChunkState(status="translated" if done else "pending")

    # ── Reading ─────────────────────────────────────────────────────────

    def chunks_with(self, *statuses: str) -> list[str]:
        return sorted(name for name, c in self.chunks.items() if c.status in statuses)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for chunk in self.chunks.values():
            counts[chunk.status] = counts.get(chunk.status, 0) + 1
        return {
            "book": self.book,
            "stage": self.stage,
            "mode": self.mode,
            "backend": self.backend,
            "started": self.started,
            "updated": self.updated,
            "chunks": counts,
            "failed": {
                name: {"attempts": c.attempts, "last_error": c.last_error}
                for name, c in sorted(self.chunks.items())
                if c.status == "failed"
            },
            "build_attempts": len(self.build_attempts),
            "last_build": self.build_attempts[-1] if self.build_attempts else None,
            "decisions": len(self.decisions),
        }

    def to_json(self) -> dict:
        data = asdict(self)
        data["chunks"] = {name: asdict(c) for name, c in sorted(self.chunks.items())}
        return data


def _chunk_id(chunk: str) -> str:
    """``12``, ``012`` and ``012.md`` all name chunk ``012``."""
    stem = Path(str(chunk)).stem
    return f"{int(stem):03d}" if stem.isdigit() else stem


def state_path(work_dir: Path) -> Path:
    return work_dir / STATE_NAME


def load(work_dir: Path) -> RunState:
    """The book's run state, or a fresh one named after the work dir."""
    path = state_path(work_dir)
    if not path.exists():
        return RunState(book=work_dir.name)
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = {name: ChunkState(**value) for name, value in (data.pop("chunks", {}) or {}).items()}
    known = {k: v for k, v in data.items() if k in RunState.__dataclass_fields__}
    return RunState(**known, chunks=chunks)


def save(work_dir: Path, state: RunState) -> Path:
    """Write atomically: a temporary file, then a replace."""
    state.updated = _now()
    path = state_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return path


__all__ = [
    "CHUNK_STATUSES",
    "STAGES",
    "STATE_NAME",
    "ChunkState",
    "RunState",
    "load",
    "save",
    "state_path",
]
