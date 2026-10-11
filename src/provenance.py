"""Who made a per-book decision, and whether the pipeline may overwrite it.

``structure.json``, ``kit/figures.json``, ``book_profile.json`` and
``page_range.json`` record each decision's author in a ``"by"`` field. Most
authors are the pipeline itself (``llm``, ``rules``, ``caption``,
``classifier`` ...), and a re-run is free to decide again. Two are not:

``manual``
    A human edited the entry.
``agent``
    The unattended orchestrator decided it -- the decisions the README used
    to leave to a human (a ``<-- CHECK`` row, a flagged figure). It is kept
    apart from ``manual`` so the run log and the outline can say which
    decisions no person has looked at.

Both are *protected*: no re-run, ``--force-*`` or ``--reclassify`` flag
overwrites them. Between the two, ``manual`` wins -- a person's correction
outranks the agent's -- wherever two records could compete for one decision
(`stronger`), and the agent is told never to edit an entry a person marked
``manual``.
"""

from __future__ import annotations

MANUAL = "manual"
AGENT = "agent"
PROTECTED: tuple[str, ...] = (MANUAL, AGENT)

_RANK = {MANUAL: 2, AGENT: 1}


def is_protected(by: object) -> bool:
    """Whether an entry with this ``by`` must survive every re-run."""
    return by in PROTECTED


def stronger(a: str | None, b: str | None) -> str | None:
    """The author whose decision wins when two compete: manual, then agent,
    then whichever came first."""
    return a if _RANK.get(a or "", 0) >= _RANK.get(b or "", 0) else b


__all__ = ["AGENT", "MANUAL", "PROTECTED", "is_protected", "stronger"]
