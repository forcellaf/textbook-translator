"""The exchange: LLM calls answered by an agent instead of an API.

No-API mode runs on a Claude Code subscription with no Anthropic API key, so
Python cannot call a model. It writes down what it *would* have sent, and a
subagent answers it:

    data/work/<book>/llm/
    ├── requests/<id>.json     written by Python
    ├── responses/<id>.txt     written by an agent: the model's raw reply, nothing else
    ├── images/<sha>.<ext>     copies of the images a classify request shows
    └── pending.json           manifest of unanswered requests, grouped by role

A request carries exactly what `src.llm.providers.anthropic.build_request`
would send -- the system prompt, the user text, the images, the role's model
and effort -- so an answer written by an agent and an answer from the API are
interchangeable, and the existing parsing code handles both the same way. A
malformed reply takes the same path a malformed API reply takes (a figure is
kept and flagged, a profile falls back to generic). Because the response file
then exists, the same request is answered from it on every later run: a bad
answer can never cause an endless re-request. Delete the response file to
ask again.

Collect, don't fail fast
------------------------
A missing answer raises `ExchangePending`, which is *not* a ``RuntimeError``
or a ``ValueError``: the retry loops and fail-fast paths built for the API
must neither retry it nor treat it as fatal. Callers catch it where they
would otherwise have used the answer, carry on, and the script reports every
pending request of the run at once. Figure classification, for instance,
enumerates every candidate in one pass and writes every request together.

Each script run is one round. Answers are cached, so a re-run reaches the
point where it stopped for free and asks only what is new. `write_pending`
records the round's requests in ``pending.json``, and the script exits with
`EXIT_PENDING` (3), a code no script used before.

Stable ids
----------
An id is the first 16 hex digits of a SHA-256 over the role, model, effort,
system prompt, user text and image hashes. Everything that goes into a
prompt is deterministic (the profiler's sample, the structure outline table,
the classifier's context window), so a re-run computes the same ids and finds
the answers. A changed prompt is a new id, so an answer to an old prompt is
never reused for a new one -- and ``--reclassify`` / ``--force-profile`` /
``--force-structure`` only discard the cached *decisions*; an unchanged
prompt still finds its old answer here.

No retries here: retrying is the orchestrator's job.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import role_config
from src.llm.base import BaseLLM, VisionLLM, image_mime_type

logger = logging.getLogger(__name__)

LLM_DIRNAME = "llm"
REQUESTS_DIRNAME = "requests"
RESPONSES_DIRNAME = "responses"
IMAGES_DIRNAME = "images"
PENDING_NAME = "pending.json"

# The exit code of a script run that left requests for an agent to answer.
# 0 is success, 1 a failure, 2 argparse's usage error, 4 a remote job still
# running (`scripts/parse_book.py`, `scripts/translate_kit.py`).
EXIT_PENDING = 3

_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}


class ExchangePending(Exception):
    """No answer yet for this request; it has been written and recorded.

    Deliberately a plain ``Exception``: see the module docstring.
    """

    def __init__(self, request_id: str, role: str) -> None:
        super().__init__(f"exchange request {request_id} ({role}) is awaiting an answer")
        self.request_id = request_id
        self.role = role


@dataclass(frozen=True)
class _Pending:
    request_id: str
    role: str


# Requests left unanswered in this process, per exchange directory. A script
# run is one process, so this is "the current round".
_PENDING: dict[Path, dict[str, _Pending]] = defaultdict(dict)


def exchange_dir(work_dir: Path) -> Path:
    return work_dir / LLM_DIRNAME


def request_id(
    role: str, model: str, effort: str, system: str, user: str, image_hashes: list[str]
) -> str:
    payload = json.dumps(
        [role, model, effort, system, user, image_hashes], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ExchangeLLM(BaseLLM, VisionLLM):
    """One role's model, answered through request and response files."""

    def __init__(self, role: str, work_dir: Path, *, reply_format: str = "json") -> None:
        self.config = role_config(role)
        self.work_dir = work_dir
        self.root = exchange_dir(work_dir)
        # Every prompt the pipeline sends through the exchange asks for JSON.
        self.reply_format = reply_format

    @property
    def role(self) -> str:
        return self.config.role

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.0) -> str:
        return self._exchange(system_prompt, user_text, [])

    def generate_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[tuple[bytes, str]],
        temperature: float = 0.0,
    ) -> str:
        return self._exchange(system_prompt, user_text, images)

    def _exchange(self, system: str, user: str, images: list[tuple[bytes, str]]) -> str:
        cfg = self.config
        hashes = [hashlib.sha256(data).hexdigest() for data, _ in images]
        rid = request_id(cfg.role, cfg.model, cfg.effort, system, user, hashes)

        response = self.root / RESPONSES_DIRNAME / f"{rid}.txt"
        if response.exists():
            # Returned as-is: the caller parses it exactly as it would an API
            # reply, malformed or not.
            return response.read_text(encoding="utf-8")

        request = self.root / REQUESTS_DIRNAME / f"{rid}.json"
        if not request.exists():
            request.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "id": rid,
                "role": cfg.role,
                "model": cfg.model,
                "effort": cfg.effort,
                "system": system,
                "user": user,
                "images": [
                    self._save_image(data, mime, digest)
                    for (data, mime), digest in zip(images, hashes)
                ],
                "reply_format": self.reply_format,
                "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            request.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        _PENDING[self.root.resolve()][rid] = _Pending(rid, cfg.role)
        raise ExchangePending(rid, cfg.role)

    def _save_image(self, data: bytes, mime: str, digest: str) -> str:
        """Copy an image where the answering agent can Read it; the path is
        relative to the work dir, as the request schema says."""
        ext = _EXTENSIONS.get(image_mime_type(data) or mime, "bin")
        path = self.root / IMAGES_DIRNAME / f"{digest[:16]}.{ext}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path.relative_to(self.work_dir).as_posix()


def pending_requests(work_dir: Path) -> dict[str, list[str]]:
    """This run's unanswered request ids, by role."""
    by_role: dict[str, list[str]] = defaultdict(list)
    for entry in _PENDING.get(exchange_dir(work_dir).resolve(), {}).values():
        by_role[entry.role].append(entry.request_id)
    return {role: sorted(ids) for role, ids in sorted(by_role.items())}


def has_pending(work_dir: Path) -> bool:
    return bool(_PENDING.get(exchange_dir(work_dir).resolve()))


def write_pending(work_dir: Path) -> int:
    """Write ``pending.json`` for this run and return the request count.

    With nothing pending the file is removed, so its presence always means
    there is work for an agent -- a stale manifest from an earlier round
    never survives a run that needed no answers.
    """
    root = exchange_dir(work_dir)
    manifest = root / PENDING_NAME
    by_role = pending_requests(work_dir)
    count = sum(len(ids) for ids in by_role.values())
    if not count:
        if manifest.exists():
            manifest.unlink()
        return 0
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": count,
        "by_role": by_role,
        "requests_dir": (root / REQUESTS_DIRNAME).as_posix(),
        "responses_dir": (root / RESPONSES_DIRNAME).as_posix(),
        "note": "answer each request by writing the raw reply to responses/<id>.txt, "
        "then re-run the same command",
    }
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return count


def pending_summary(work_dir: Path) -> str:
    """The short report a script prints before exiting with `EXIT_PENDING`."""
    by_role = pending_requests(work_dir)
    counts = ", ".join(f"{role} {len(ids)}" for role, ids in by_role.items())
    return (
        f"exchange: {sum(len(ids) for ids in by_role.values())} request(s) awaiting an "
        f"answer ({counts}); manifest {exchange_dir(work_dir) / PENDING_NAME}. Answer them "
        f"and re-run (exit code {EXIT_PENDING})."
    )


def reset() -> None:
    """Forget this process's pending requests (tests, and long-lived callers)."""
    _PENDING.clear()


__all__ = [
    "EXIT_PENDING",
    "LLM_DIRNAME",
    "PENDING_NAME",
    "ExchangeLLM",
    "ExchangePending",
    "exchange_dir",
    "has_pending",
    "pending_requests",
    "pending_summary",
    "request_id",
    "reset",
    "write_pending",
]
