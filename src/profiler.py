"""
Book-level profiling: one LLM call per book, before any translation happens.

Why this exists
---------------
Chunks are translated independently, so nothing except a shared prompt keeps
them consistent. Without a profile, the same term comes back three different
ways across three chapters, and the model has to guess the book's subject,
level and register from whatever 3,000-token window it happens to be looking
at. ``profile_book`` reads a sample of the whole book once, extracts that
shared context (subject, audience, register, notation conventions and -- most
importantly -- a glossary), and ``profile_to_prompt_block`` renders it into a
compact fragment that is prepended to every chunk's system prompt.

Failure policy: profiling is an *optimization*, never a gate. Any failure
(unreachable LLM, unparseable JSON, unreadable file) is logged as a warning
and degrades to ``BookProfile.generic()``. A profiling hiccup must not kill a
multi-hour translation run. Failed profiles are deliberately not cached --
otherwise one bad night would stick to the book forever.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import SOURCE_LANG, TARGET_LANG
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "book_profile.json"

# ── Sampling budget ─────────────────────────────────────────────────────────
# One call, ~24k chars of input: enough context to characterize a textbook
# without paying to send the whole book.
_HEAD_CHARS = 6_000  # front matter / TOC / preface usually states level + audience
_SLICE_CHARS = 3_000
_SLICE_COUNT = 5
_SAMPLE_BUDGET = _HEAD_CHARS + _SLICE_CHARS * _SLICE_COUNT

_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")

_GLOSSARY_MIN = 15
_GLOSSARY_MAX = 40


@dataclass(frozen=True)
class BookProfile:
    """Book-wide translation context, shared by every chunk of one book.

    Every field has a usable default, so a partially-parsed (or entirely
    failed) profile is still a valid object -- callers never have to
    None-check individual fields.
    """

    subject: str = "general"
    subfield: str = ""
    education_level: str = "university"
    audience: str = "students"
    register: str = "formal academic prose"
    notation_notes: str = ""
    structural_elements: tuple[str, ...] = ()
    glossary: tuple[tuple[str, str], ...] = ()  # (source_term, target_term)
    latex_documentclass: str = "book"
    latex_packages: tuple[str, ...] = ()
    summary: str = ""

    @classmethod
    def generic(cls) -> "BookProfile":
        """The fallback profile: all defaults, no glossary, no assumptions."""
        return cls()

    def to_json(self) -> str:
        return json.dumps(
            {
                "subject": self.subject,
                "subfield": self.subfield,
                "education_level": self.education_level,
                "audience": self.audience,
                "register": self.register,
                "notation_notes": self.notation_notes,
                "structural_elements": list(self.structural_elements),
                "glossary": [{"source": s, "target": t} for s, t in self.glossary],
                "latex_documentclass": self.latex_documentclass,
                "latex_packages": list(self.latex_packages),
                "summary": self.summary,
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "BookProfile":
        """Build a profile from an arbitrary dict, tolerating LLM sloppiness.

        Accepts missing keys, wrong types, glossary entries in either
        ``{"source": ..., "target": ...}`` or ``["源", "source"]`` form, and
        bare strings where a list is expected. Anything unusable is skipped
        rather than raising -- a half-parsed profile still beats none.
        """
        if not isinstance(data, dict):
            return cls.generic()

        defaults = cls()
        return cls(
            subject=_as_str(data, "subject", defaults.subject),
            subfield=_as_str(data, "subfield", defaults.subfield),
            education_level=_as_str(data, "education_level", defaults.education_level),
            audience=_as_str(data, "audience", defaults.audience),
            register=_as_str(data, "register", defaults.register),
            notation_notes=_as_str(data, "notation_notes", defaults.notation_notes),
            structural_elements=_as_str_tuple(data.get("structural_elements")),
            glossary=_as_glossary(data.get("glossary")),
            latex_documentclass=_as_str(
                data, "latex_documentclass", defaults.latex_documentclass
            ),
            latex_packages=_as_str_tuple(data.get("latex_packages")),
            summary=_as_str(data, "summary", defaults.summary),
        )


# ── from_dict coercion helpers ──────────────────────────────────────────────


def _as_str(data: dict, key: str, default: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return default


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a list-ish field. A bare string becomes a 1-tuple (models
    routinely answer ``"amsmath"`` where a list was asked for)."""
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, (list, tuple)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _as_glossary(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    entries: list[tuple[str, str]] = []
    for item in value:
        pair: tuple[str, str] | None = None
        if isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
            if isinstance(source, str) and isinstance(target, str):
                pair = (source.strip(), target.strip())
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            source, target = item
            if isinstance(source, str) and isinstance(target, str):
                pair = (source.strip(), target.strip())

        if pair and pair[0] and pair[1]:
            entries.append(pair)
        else:
            logger.debug("Skipping malformed glossary entry: %r", item)

    return tuple(entries)


# ── Sampling ────────────────────────────────────────────────────────────────


def _slice_starts(text: str, head_end: int) -> list[int]:
    """Pick ``_SLICE_COUNT`` evenly-spaced offsets in the book's body, each
    snapped forward to the nearest Markdown heading when one is close by, so
    samples begin at a section boundary instead of mid-sentence."""
    headings = [m.start() for m in _HEADING_RE.finditer(text)]
    span = len(text) - head_end
    starts: list[int] = []

    for i in range(1, _SLICE_COUNT + 1):
        target = head_end + int(span * i / (_SLICE_COUNT + 1))
        window = target + _SLICE_CHARS  # only snap forward within one slice
        snapped = next((h for h in headings if target <= h < window), target)
        if not starts or snapped >= starts[-1] + _SLICE_CHARS // 2:
            starts.append(snapped)

    return starts


def _sample_text(text: str) -> str:
    """Front matter plus interior slices, capped at ``_SAMPLE_BUDGET`` chars."""
    if len(text) <= _SAMPLE_BUDGET:
        return text

    parts = [text[:_HEAD_CHARS]]
    for start in _slice_starts(text, _HEAD_CHARS):
        parts.append(text[start : start + _SLICE_CHARS])

    return "\n\n[...]\n\n".join(parts)


# ── Prompting ───────────────────────────────────────────────────────────────


def _build_profile_prompt(source_lang: str, target_lang: str) -> str:
    return (
        f"You are a publishing analyst preparing a {source_lang}-to-{target_lang} "
        "translation of a textbook. You are given excerpts sampled from across one "
        "book (front matter first, then interior slices separated by [...]).\n\n"
        "Return ONLY a single JSON object -- no prose, no explanation, no code "
        "fences -- with exactly these keys:\n"
        '  "subject": broad field, e.g. "mathematics", "physics", "law"\n'
        '  "subfield": narrower area, e.g. "real analysis"\n'
        '  "education_level": e.g. "high school", "undergraduate", "graduate"\n'
        '  "audience": who reads this book\n'
        f'  "register": how the {target_lang} prose should read, e.g. '
        '"formal academic prose"\n'
        '  "notation_notes": conventions a translator must preserve (symbol usage, '
        "numbering style, units)\n"
        '  "structural_elements": list of recurring block types, e.g. '
        '["definition", "theorem", "proof", "exercise"]\n'
        f'  "glossary": list of {_GLOSSARY_MIN}-{_GLOSSARY_MAX} objects '
        '{"source": "<term as it appears in the book>", '
        f'"target": "<the {target_lang} term to always use>"}}\n'
        '  "latex_documentclass": one of "book", "report", "article"\n'
        '  "latex_packages": extra LaTeX packages this content needs, e.g. '
        '["amsthm", "siunitx"]\n'
        '  "summary": two sentences on what the book covers\n\n'
        "The glossary is by far the most important field. The book is translated "
        "in independent chunks, so any term you omit will be rendered "
        "inconsistently across chapters. Prioritize terms that have several "
        "plausible translations, plus names of recurring structures, and give the "
        "single translation that must be used every time."
    )


def profile_to_prompt_block(profile: BookProfile, *, max_glossary: int = 40) -> str:
    """Render ``profile`` as a compact prompt fragment.

    Kept terse on purpose: this text is prepended to *every* chunk's system
    prompt, so its length is multiplied by the number of chunks in the book.
    """
    lines = ["BOOK CONTEXT", f"- Subject: {profile.subject}"]
    if profile.subfield:
        lines.append(f"- Subfield: {profile.subfield}")
    lines.append(f"- Level / audience: {profile.education_level}, {profile.audience}")
    lines.append(f"- Register: {profile.register}")
    if profile.notation_notes:
        lines.append(f"- Notation: {profile.notation_notes}")
    if profile.structural_elements:
        lines.append(f"- Recurring blocks: {', '.join(profile.structural_elements)}")

    glossary = profile.glossary[:max_glossary]
    if glossary:
        lines.append("")
        lines.append(
            "Required terminology (use these exactly and consistently, every "
            "time the term appears):"
        )
        lines.extend(f"  {source} -> {target}" for source, target in glossary)

    return "\n".join(lines)


# ── Response parsing ────────────────────────────────────────────────────────


def _extract_json_object(raw: str) -> dict:
    """Pull a JSON object out of a model response.

    Models wrap JSON in ``` fences, prefix it with "Here is the profile:", or
    both. Strip fences first, then fall back to slicing between the first
    ``{`` and the last ``}``.
    """
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _load_cached(cache_path: Path) -> BookProfile | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable cached profile %s: %s", cache_path, exc)
        return None
    return BookProfile.from_dict(data)


def profile_book(
    merged_md_path: Path,
    work_dir: Path,
    *,
    llm: BaseLLM | None = None,
    source_lang: str = SOURCE_LANG,
    target_lang: str = TARGET_LANG,
    force: bool = False,
) -> BookProfile:
    """Profile the book at ``merged_md_path``, caching to ``work_dir``.

    Args:
        merged_md_path: The book's merged Markdown.
        work_dir: Directory holding this book's artifacts; the profile is
            cached at ``work_dir/book_profile.json``.
        llm: Provider to use. Defaults to the configured one.
        source_lang: Language the book is written in.
        target_lang: Language it will be translated into.
        force: Re-profile even when a cached profile exists.

    Returns:
        The parsed ``BookProfile``, or ``BookProfile.generic()`` if anything
        went wrong. Never raises.
    """
    cache_path = work_dir / PROFILE_FILENAME

    if not force and cache_path.exists():
        cached = _load_cached(cache_path)
        if cached is not None:
            logger.info("Using cached book profile: %s", cache_path)
            return cached

    try:
        text = merged_md_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"{merged_md_path} is empty")

        if llm is None:
            from src.llm.factory import get_llm

            llm = get_llm()

        raw = llm.generate(
            _build_profile_prompt(source_lang, target_lang),
            _sample_text(text),
            temperature=0.2,
        )
        profile = BookProfile.from_dict(_extract_json_object(raw))
    except Exception as exc:  # noqa: BLE001 - profiling must never be fatal
        logger.warning(
            "Book profiling failed (%s); falling back to a generic profile. "
            "Translation will continue with reduced terminology consistency.",
            exc,
        )
        # Deliberately NOT cached: a transient failure must not become sticky.
        return BookProfile.generic()

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(profile.to_json(), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache book profile to %s: %s", cache_path, exc)

    logger.info(
        "Profiled book as %s/%s (%s), %d glossary term(s)",
        profile.subject,
        profile.subfield or "-",
        profile.education_level,
        len(profile.glossary),
    )
    return profile


__all__ = [
    "PROFILE_FILENAME",
    "BookProfile",
    "profile_book",
    "profile_to_prompt_block",
]
