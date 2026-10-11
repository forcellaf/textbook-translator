"""
API mode's translator: the kit's untranslated chunks as one Message Batch.

No-API mode translates a kit with one ``chunk-translator`` agent per chunk.
API mode does the same job in bulk: every chunk without a saved translation
goes up as one request in a single Message Batch (half the price of live
calls, and no rate-limit pacing), and each reply is written to
``kit/translated/NNN.tex`` -- the same files, so lint, build and the rest of
the pipeline cannot tell the two modes apart.

Every request is built by ``src.llm.providers.anthropic.build_request`` for
the ``translate`` role: the kit's ``system_prompt.txt`` (identical for every
chunk, so it carries ``cache_control`` and is cached across the batch) and
the chunk as the user message.

Truncation is checked twice, as the brief requires of the API path: a reply
whose ``stop_reason`` is ``max_tokens`` fails, and so does a reply that does
not end with the ``% END-OF-CHUNK`` sentinel. A failed reply is kept as
``translated/NNN.failed.tex`` for inspection, never as ``NNN.tex``, so lint
keeps reporting the chunk as missing.

Resumable, like the parse: the batch id is saved in
``kit/translate_batch.json`` when the batch is created, and a re-run polls
that batch instead of submitting another one.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROLE_TRANSLATE, role_config
from src.lint import SENTINEL, has_sentinel
from src.llm.providers.anthropic import build_request, reply_text

logger = logging.getLogger(__name__)

STATE_NAME = "translate_batch.json"
FAILED_SUFFIX = ".failed.tex"


@dataclass
class BatchOutcome:
    state: str  # nothing-to-do | running | collected
    batch_id: str | None = None
    submitted: list[str] = field(default_factory=list)
    translated: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "state": self.state,
            "batch_id": self.batch_id,
            "submitted": self.submitted,
            "translated": self.translated,
            "failed": self.failed,
            "counts": self.counts,
        }


def untranslated(kit_dir: Path) -> list[str]:
    """Chunks with no ``translated/NNN.tex``."""
    return [
        source.stem
        for source in sorted((kit_dir / "chunks").glob("*.md"))
        if not (kit_dir / "translated" / f"{source.stem}.tex").exists()
    ]


def _load_state(kit_dir: Path) -> dict:
    try:
        return json.loads((kit_dir / STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(kit_dir: Path, state: dict) -> None:
    (kit_dir / STATE_NAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def submit(kit_dir: Path, chunks: list[str], client) -> str:  # noqa: ANN001
    """Create one batch for ``chunks`` and record it before anything else."""
    cfg = role_config(ROLE_TRANSLATE)
    system_prompt = (kit_dir / "system_prompt.txt").read_text(encoding="utf-8")
    requests = [
        {
            "custom_id": chunk,
            "params": build_request(
                cfg,
                system_prompt,
                (kit_dir / "chunks" / f"{chunk}.md").read_text(encoding="utf-8"),
            ),
        }
        for chunk in chunks
    ]
    batch = client.messages.batches.create(requests=requests)
    _save_state(
        kit_dir,
        {
            "batch_id": batch.id,
            "state": "submitted",
            "chunks": chunks,
            "model": cfg.model,
            "effort": cfg.effort,
            "submitted_at": _now(),
        },
    )
    logger.info("Submitted batch %s: %d chunk(s) on %s", batch.id, len(chunks), cfg.model)
    return batch.id


def collect(kit_dir: Path, batch_id: str, client) -> BatchOutcome:  # noqa: ANN001
    """Write every succeeded, complete reply; record why the others failed."""
    model = role_config(ROLE_TRANSLATE).model
    outcome = BatchOutcome("collected", batch_id)
    translated_dir = kit_dir / "translated"
    translated_dir.mkdir(exist_ok=True)
    for entry in client.messages.batches.results(batch_id):
        chunk = entry.custom_id
        result = entry.result
        if result.type != "succeeded":
            error = getattr(getattr(result, "error", None), "error", None)
            outcome.failed[chunk] = f"{result.type}: {getattr(error, 'message', '') or ''}".strip()
            continue
        message = result.message
        partial = "".join(b.text for b in message.content if b.type == "text")
        try:
            text = reply_text(message, model)
        except RuntimeError as exc:  # truncated, refused, empty
            outcome.failed[chunk] = str(exc)
            _keep_failed(translated_dir, chunk, partial)
            continue
        if not has_sentinel(text):
            outcome.failed[chunk] = f"the reply does not end with {SENTINEL}: cut off"
            _keep_failed(translated_dir, chunk, text)
            continue
        (translated_dir / f"{chunk}.tex").write_text(text, encoding="utf-8", newline="\n")
        stale = translated_dir / f"{chunk}{FAILED_SUFFIX}"
        if stale.exists():
            stale.unlink()
        outcome.translated.append(chunk)
    outcome.translated.sort()
    return outcome


def _keep_failed(directory: Path, chunk: str, text: str) -> None:
    if text.strip():
        (directory / f"{chunk}{FAILED_SUFFIX}").write_text(text, encoding="utf-8", newline="\n")


def run(
    kit_dir: Path,
    *,
    client,  # noqa: ANN001
    chunks: list[str] | None = None,
    force: bool = False,
    wait_minutes: float = 9.0,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> BatchOutcome:
    """Submit (or resume) the batch, poll for up to ``wait_minutes``, collect."""
    state = _load_state(kit_dir)
    if state.get("state") == "submitted" and not force:
        batch_id = state["batch_id"]
        submitted = list(state.get("chunks", []))
        logger.info("Resuming batch %s", batch_id)
    else:
        todo = chunks if chunks is not None else untranslated(kit_dir)
        if not todo:
            return BatchOutcome("nothing-to-do")
        batch_id = submit(kit_dir, todo, client)
        submitted = todo

    deadline = time.monotonic() + wait_minutes * 60
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts.to_dict() if batch.request_counts else {}
        if batch.processing_status == "ended":
            break
        if time.monotonic() >= deadline:
            return BatchOutcome("running", batch_id, submitted, counts=counts)
        sleep(poll_seconds)

    outcome = collect(kit_dir, batch_id, client)
    outcome.submitted, outcome.counts = submitted, counts
    state.update(
        batch_id=batch_id,
        state="collected",
        chunks=submitted,
        collected_at=_now(),
        translated=outcome.translated,
        failed=outcome.failed,
    )
    _save_state(kit_dir, state)
    return outcome


__all__ = ["FAILED_SUFFIX", "STATE_NAME", "BatchOutcome", "collect", "run", "submit", "untranslated"]
