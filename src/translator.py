"""
The translation stage: Markdown in, translated Markdown or LaTeX out.

Layout
------
``chunk_markdown``   splits a document on paragraph boundaries into
                     token-budgeted pieces.
``translate_chunk``  translates one piece, with two *separate* resilience
                     layers (see below).
``translate_markdown`` runs a whole document through those two.
``translate_book``   runs a whole book chapter-by-chapter with on-disk
                     checkpoints, so an interrupted multi-hour run resumes
                     instead of restarting.

The two resilience layers, and why they must not compound
---------------------------------------------------------
1. *Transient* failures (network blip, rate limit) are retried inside a
   single ``llm.generate`` call by tenacity, with exponential backoff. When
   that whole schedule is exhausted, the failure is permanent as far as this
   module is concerned and ``TranslationError`` propagates immediately.
2. *Successful but bad* output (empty, implausibly short, still in the source
   language, structurally invalid LaTeX) is re-asked up to
   ``MAX_HEAL_ATTEMPTS`` times, quoting the specific problem back to the
   model.

These are nested the only way round that works: the heal loop never catches
``TranslationError``. If it did, a dead API key would cost
``API_MAX_RETRIES x (MAX_HEAL_ATTEMPTS + 1)`` attempts and several minutes of
backoff before reporting the obvious -- instead of one backoff schedule.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from pathlib import Path

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.chapter_splitter import split_into_chapters
from src.config import (
    API_MAX_RETRIES,
    MAX_CHUNK_TOKENS,
    MAX_HEAL_ATTEMPTS,
    SOURCE_LANG,
    SOURCE_RESIDUE_THRESHOLD,
    TARGET_LANG,
)
from src.latex import assemble_document, validate_fragment
from src.llm.base import BaseLLM
from src.profiler import BookProfile, profile_book, profile_to_prompt_block

logger = logging.getLogger(__name__)

OUTPUT_FORMATS: tuple[str, ...] = ("markdown", "latex")

CHAPTERS_DIRNAME = "chapters"
TRANSLATED_CHAPTERS_DIRNAME = "translated_chapters"
TRANSLATED_MERGED_NAME = "translated_merged.md"
TRANSLATED_TEX_NAME = "translated_book.tex"

_FORMAT_SUFFIXES = {"markdown": ".md", "latex": ".tex"}

# No tokenizer ships with this project, so chunk budgets are estimated from
# character count. The ratio is skewed toward the CJK end (CJK is roughly one
# token per character, Markdown/English roughly one per four): over-estimating
# tokens costs an extra chunk boundary, under-estimating costs a rejected
# request mid-book.
CHARS_PER_TOKEN = 1.5

# "Implausibly short" heuristic: a real translation of a 600-char source is
# never 22 chars. Only applied above _SHORT_SOURCE_FLOOR -- short paragraphs
# legitimately compress a lot (a 40-char CJK sentence can be 30 chars of
# English).
_MIN_LENGTH_RATIO = 0.30
_SHORT_SOURCE_FLOOR = 200

# Floor for the source-residue check. Below this many non-whitespace
# characters a percentage is noise: one legitimately kept proper noun in a
# two-line chunk is already well past any sane threshold.
_MIN_RESIDUE_SAMPLE = 80

# Backoff schedule for transient API failures: ~1+2+4+8s across the default
# five attempts. Module-level so tests can neutralize the sleeps.
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=60)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")
_FULL_FENCE_RE = re.compile(r"\A```[a-zA-Z0-9_+-]*\n(.*)\n```\Z", re.DOTALL)


class TranslationError(RuntimeError):
    """Raised when a chunk cannot be translated after the full retry schedule."""


# ── Chunking ────────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` from its character count."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def chunk_markdown(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` estimated tokens.

    Splits **only** on blank-line paragraph boundaries, never inside a
    paragraph, so Markdown tables, list blocks and display math survive
    intact.

    A single paragraph that exceeds the budget on its own is emitted whole
    rather than cut: one oversized request either succeeds or fails loudly,
    whereas a table sliced down the middle produces plausible-looking,
    permanently corrupted output.

    Returns ``[]`` for empty or whitespace-only input.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip("\n") for p in _PARAGRAPH_SPLIT_RE.split(text)]
    paragraphs = [p for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = estimate_tokens(paragraph)

        if current and current_tokens + tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(paragraph)
        current_tokens += tokens

        if current_tokens >= max_tokens:
            if len(current) == 1 and tokens > max_tokens:
                logger.warning(
                    "Paragraph of ~%d estimated tokens exceeds the %d-token budget; "
                    "keeping it whole to avoid corrupting its content.",
                    tokens,
                    max_tokens,
                )
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ── Prompts ─────────────────────────────────────────────────────────────────

_MARKDOWN_RULES = """\
Rules:
- Output ONLY the translated Markdown. No preamble, no commentary, no code
  fences around the whole answer, no notes about what you did.
- Preserve the Markdown structure exactly: heading levels, list markers,
  table pipes, block quotes, emphasis, and blank-line paragraph breaks.
- Preserve every formula verbatim, inline ($...$) and display ($$...$$)
  alike. Translate surrounding prose, never the mathematics itself: do not
  "fix", simplify, re-derive or re-number formulas.
- Keep image references exactly as given, including the path:
  ![alt](path/to/image.jpg) -- translate the alt text only.
- Translate every piece of prose. Do not leave source-language text in the
  output, and do not add content that is not in the source."""

_LATEX_RULES = """\
Output LaTeX body content only.

Hard constraints -- the preamble is generated separately and yours would
conflict with it:
- NEVER emit \\documentclass, \\usepackage, \\begin{document} or
  \\end{document}.
- NEVER wrap the answer in ``` code fences, and never add commentary.

Structure mapping:
- Markdown '# '   -> \\chapter{...}
- Markdown '## '  -> \\section{...}
- Markdown '### ' -> \\subsection{...}
- Bulleted lists -> itemize; numbered lists -> enumerate.
- Tables -> tabular inside a table float, using booktabs rules (\\toprule,
  \\midrule, \\bottomrule).
- ![alt](path) -> a figure float with \\includegraphics{path}, keeping the
  path EXACTLY as given, and the translated alt text as the \\caption.

Mathematics:
- Preserve all math verbatim. Never "fix", simplify or re-derive a formula.
- Inline $...$ stays inline. Display math becomes an equation environment or
  \\[ ... \\].

Escaping:
- In prose, escape &, %, #, _ and any literal $ as \\&, \\%, \\#, \\_, \\$.
- Every \\begin must have a matching \\end, and braces must balance."""


def build_system_prompt(
    *,
    source_lang: str,
    target_lang: str,
    profile: BookProfile | None = None,
    output_format: str = "markdown",
) -> str:
    """Assemble the system prompt for one chunk translation."""
    rules = _LATEX_RULES if output_format == "latex" else _MARKDOWN_RULES
    parts = [
        f"You are an expert {source_lang}-to-{target_lang} translator working on "
        "one excerpt of a textbook. Translate the excerpt the user sends you.",
        rules,
    ]
    if profile is not None:
        parts.append(profile_to_prompt_block(profile))
    return "\n\n".join(parts)


def _heal_prompt(system_prompt: str, problem: str) -> str:
    """Re-ask prompt that names the actual defect.

    Quoting the specific problem matters: "try again" gets the same answer
    back, whereas "\\begin{equation} is never closed" gets it fixed.
    """
    return (
        f"{system_prompt}\n\n"
        "IMPORTANT -- your previous answer to this exact excerpt was rejected: "
        f"{problem}.\n"
        "Produce a corrected, complete translation of the excerpt below. Fix "
        "that specific problem, translate the entire excerpt, and follow all "
        "the rules above."
    )


# ── Single-chunk translation ────────────────────────────────────────────────


def _strip_wrapping_fence(text: str) -> str:
    """Remove a code fence wrapping the *entire* response.

    Only when the whole answer is one fenced block -- a document containing
    legitimate fenced code blocks must keep them.
    """
    match = _FULL_FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


# ── Source-language residue ─────────────────────────────────────────────────
#
# A chunk can come back full-length and structurally perfect while still being
# the untranslated source: the model does the Markdown-to-LaTeX conversion and
# silently skips the translation half of the job, sometimes emitting a heading
# twice (once per language) before continuing in the source language. The
# length heuristic sees a full-length answer and the LaTeX validator sees
# well-formed \section{} blocks, so measuring how much source-only script
# survives is the only thing that catches it.

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
    "kana": ((0x3040, 0x30FF), (0x31F0, 0x31FF)),
    "hangul": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)),
    "latin": ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x400, 0x52F),),
    "greek": ((0x370, 0x3FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x600, 0x6FF), (0x750, 0x77F)),
    "hebrew": ((0x590, 0x5FF),),
    "devanagari": ((0x900, 0x97F),),
    "thai": ((0xE00, 0xE7F),),
}

# Which scripts each language is written in. This exists to answer exactly one
# question -- does the source use a script the target does not? -- so a
# language absent from this map disables the check instead of guessing. A
# wrong guess bills an extra API call on every chunk of the book.
#
# Han is listed for Japanese and Korean deliberately: both legitimately contain
# Han characters, so Chinese -> Japanese leaves nothing source-exclusive and
# the check correctly turns itself off rather than flagging valid output.
_LANGUAGE_SCRIPTS: dict[str, frozenset[str]] = {
    "chinese": frozenset({"han"}),
    "simplified chinese": frozenset({"han"}),
    "traditional chinese": frozenset({"han"}),
    "mandarin": frozenset({"han"}),
    "zh": frozenset({"han"}),
    "japanese": frozenset({"han", "kana"}),
    "ja": frozenset({"han", "kana"}),
    "korean": frozenset({"hangul", "han"}),
    "ko": frozenset({"hangul", "han"}),
    "english": frozenset({"latin"}),
    "en": frozenset({"latin"}),
    "french": frozenset({"latin"}),
    "fr": frozenset({"latin"}),
    "german": frozenset({"latin"}),
    "de": frozenset({"latin"}),
    "spanish": frozenset({"latin"}),
    "es": frozenset({"latin"}),
    "italian": frozenset({"latin"}),
    "portuguese": frozenset({"latin"}),
    "dutch": frozenset({"latin"}),
    "polish": frozenset({"latin"}),
    "turkish": frozenset({"latin"}),
    "vietnamese": frozenset({"latin"}),
    "indonesian": frozenset({"latin"}),
    "malay": frozenset({"latin"}),
    "russian": frozenset({"cyrillic"}),
    "ru": frozenset({"cyrillic"}),
    "ukrainian": frozenset({"cyrillic"}),
    "greek": frozenset({"greek"}),
    "arabic": frozenset({"arabic"}),
    "persian": frozenset({"arabic"}),
    "farsi": frozenset({"arabic"}),
    "hebrew": frozenset({"hebrew"}),
    "hindi": frozenset({"devanagari"}),
    "thai": frozenset({"thai"}),
}

# Regions that must not count towards residue: a formula with a CJK subscript
# and an image path with a CJK filename are both correct output. Ordered
# longest-delimiter-first so $$...$$ is consumed before the inline $...$ rule
# can bite into it.
_NON_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(
        r"\\begin\{(equation|align|gather|multline|eqnarray|displaymath|math)\*?\}"
        r".*?\\end\{\1\*?\}",
        re.DOTALL,
    ),
    re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.DOTALL),
    re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}"),
)

# ![alt](path) -> ![alt]: the path is not prose, the alt text is and must be
# translated like any other caption.
_MARKDOWN_IMAGE_PATH_RE = re.compile(r"(!\[[^\]]*\])\([^)]*\)")


def _script_of(char: str) -> str | None:
    """Name of the script ``char`` belongs to, or None if it is in none of
    the tracked ranges (digits, punctuation, symbols)."""
    code = ord(char)
    for script, ranges in _SCRIPT_RANGES.items():
        if any(low <= code <= high for low, high in ranges):
            return script
    return None


def _strip_non_prose(text: str) -> str:
    """Remove math and image paths, leaving the text that had to be translated."""
    for pattern in _NON_PROSE_PATTERNS:
        text = pattern.sub(" ", text)
    return _MARKDOWN_IMAGE_PATH_RE.sub(r"\1", text)


def _source_residue(output: str, source_lang: str, target_lang: str) -> float | None:
    """Fraction of ``output``'s prose still written in a source-only script.

    Returns None when the check does not apply: when either language's script
    is unknown, when the source uses no script the target lacks (English ->
    French, Chinese -> Japanese), or when there is too little text to measure.
    """
    source_scripts = _LANGUAGE_SCRIPTS.get(source_lang.strip().lower())
    target_scripts = _LANGUAGE_SCRIPTS.get(target_lang.strip().lower())
    if source_scripts is None or target_scripts is None:
        return None

    source_only = source_scripts - target_scripts
    if not source_only:
        return None

    prose = [char for char in _strip_non_prose(output) if not char.isspace()]
    if len(prose) < _MIN_RESIDUE_SAMPLE:
        return None

    residue = sum(1 for char in prose if _script_of(char) in source_only)
    return residue / len(prose)


def _diagnose(
    source: str,
    output: str,
    output_format: str,
    *,
    source_lang: str = SOURCE_LANG,
    target_lang: str = TARGET_LANG,
) -> str | None:
    """Return a human-readable description of what's wrong with ``output``,
    or None if it looks usable."""
    stripped = output.strip()

    if not stripped or (
        len(source) > _SHORT_SOURCE_FLOOR and len(stripped) < _MIN_LENGTH_RATIO * len(source)
    ):
        return (
            f"output was empty or implausibly short ({len(stripped)} chars "
            f"for {len(source)} chars of input)"
        )

    residue = _source_residue(stripped, source_lang, target_lang)
    if residue is not None and residue > SOURCE_RESIDUE_THRESHOLD:
        return f"output is still {residue:.0%} {source_lang} - the text was not translated"

    if output_format == "latex":
        issues = validate_fragment(stripped)
        if issues:
            return "invalid LaTeX: " + "; ".join(issues)

    return None


def _generate_with_retry(llm: BaseLLM, system_prompt: str, user_text: str) -> str:
    """One LLM call, with the full transient-failure retry schedule.

    ``GeminiProvider`` raises ``RuntimeError`` for every API-level failure, so
    that is what is retried. Exhausting the schedule converts to
    ``TranslationError``, which callers must let propagate rather than fold
    into a heal loop.
    """
    retryer = Retrying(
        stop=stop_after_attempt(API_MAX_RETRIES),
        wait=_RETRY_WAIT,
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    try:
        return retryer(llm.generate, system_prompt, user_text, 0.3)
    except TranslationError:
        raise
    except RuntimeError as exc:
        raise TranslationError(
            f"LLM call failed after {API_MAX_RETRIES} attempt(s): {exc}"
        ) from exc


def translate_chunk(
    chunk: str,
    *,
    llm: BaseLLM,
    source_lang: str = SOURCE_LANG,
    target_lang: str = TARGET_LANG,
    profile: BookProfile | None = None,
    output_format: str = "markdown",
) -> str:
    """Translate one chunk, healing bad-but-successful responses.

    Raises:
        TranslationError: if the API itself keeps failing. This propagates
            out of the heal loop immediately and on purpose -- see the module
            docstring.
    """
    system_prompt = build_system_prompt(
        source_lang=source_lang,
        target_lang=target_lang,
        profile=profile,
        output_format=output_format,
    )

    # Layer 1 (transient failures) happens inside this call; a permanent
    # failure raises TranslationError here and never reaches the heal loop.
    result = _strip_wrapping_fence(_generate_with_retry(llm, system_prompt, chunk))
    problem = _diagnose(
        chunk, result, output_format, source_lang=source_lang, target_lang=target_lang
    )

    # Layer 2: the API is answering, the answer is just unusable.
    for attempt in range(1, MAX_HEAL_ATTEMPTS + 1):
        if problem is None:
            return result
        logger.warning(
            "Chunk translation problem (heal attempt %d/%d): %s",
            attempt,
            MAX_HEAL_ATTEMPTS,
            problem,
        )
        result = _strip_wrapping_fence(
            _generate_with_retry(llm, _heal_prompt(system_prompt, problem), chunk)
        )
        problem = _diagnose(
            chunk, result, output_format, source_lang=source_lang, target_lang=target_lang
        )

    if problem is not None:
        # Returning imperfect text beats aborting a multi-hour book run over
        # one stubborn chunk; the error log says which chunk to review.
        logger.error(
            "Giving up healing after %d attempt(s) (%s); keeping the last response.",
            MAX_HEAL_ATTEMPTS,
            problem,
        )
    return result


# ── Document translation ────────────────────────────────────────────────────


def _validate_output_format(output_format: str) -> None:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unknown output_format {output_format!r}; expected one of {list(OUTPUT_FORMATS)}"
        )


def translate_markdown(
    markdown_text: str,
    target_lang: str = TARGET_LANG,
    *,
    source_lang: str = SOURCE_LANG,
    llm: BaseLLM | None = None,
    on_chunk_done: Callable[[int, int, str], None] | None = None,
    profile: BookProfile | None = None,
    output_format: str = "markdown",
) -> str:
    """Translate a whole Markdown document, chunk by chunk.

    Args:
        markdown_text: The source document.
        target_lang: Language to translate into.
        source_lang: Language the document is written in.
        llm: Provider to use. Defaults to the configured one.
        on_chunk_done: Progress callback, called as
            ``(index, total, translated_chunk)`` with a 1-based index.
        profile: Book context to prepend to every chunk's system prompt.
        output_format: ``"markdown"`` or ``"latex"``.

    Returns:
        The translated document. ``""`` for empty input.
    """
    _validate_output_format(output_format)

    chunks = chunk_markdown(markdown_text)
    if not chunks:
        return ""

    if llm is None:
        from src.llm.factory import get_llm

        llm = get_llm()

    translated: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        logger.info("Translating chunk %d/%d (%d chars)", index, len(chunks), len(chunk))
        piece = translate_chunk(
            chunk,
            llm=llm,
            source_lang=source_lang,
            target_lang=target_lang,
            profile=profile,
            output_format=output_format,
        )
        translated.append(piece)
        if on_chunk_done is not None:
            on_chunk_done(index, len(chunks), piece)

    return "\n\n".join(translated)


# ── Book translation ────────────────────────────────────────────────────────


def _checkpoint_path(checkpoint_dir: Path, chapter_path: Path, output_format: str) -> Path:
    """Checkpoint path for one chapter, carrying a format-specific suffix.

    The suffix is what stops a run with ``output_format="latex"`` from
    happily reusing Markdown checkpoints written by an earlier run (and
    assembling a .tex file half-full of Markdown).
    """
    return checkpoint_dir / f"{chapter_path.stem}{_FORMAT_SUFFIXES[output_format]}"


def _is_fresh(checkpoint: Path, source: Path) -> bool:
    try:
        return checkpoint.exists() and checkpoint.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def translate_book(
    merged_md_path: Path,
    work_dir: Path,
    *,
    target_lang: str = TARGET_LANG,
    source_lang: str = SOURCE_LANG,
    llm: BaseLLM | None = None,
    resume: bool = True,
    output_format: str = "markdown",
    profile: BookProfile | None = None,
    use_profile: bool = True,
    title: str | None = None,
) -> Path:
    """Translate a whole book, one chapter at a time, with checkpoints.

    Chapters are translated into ``work_dir/translated_chapters/`` and only
    then concatenated, so an interrupted run resumes from the last completed
    chapter rather than re-paying for the whole book.

    Profiling is lazy: the set of chapters actually needing work is computed
    first, and ``profile_book`` is only called when that set is non-empty.
    Re-running a fully-translated book therefore costs zero LLM calls.

    Args:
        merged_md_path: The book's merged Markdown.
        work_dir: Directory for this book's artifacts.
        target_lang: Language to translate into.
        source_lang: Language the book is written in.
        llm: Provider to use. Defaults to the configured one, resolved lazily.
        resume: Reuse checkpoints that are newer than their source chapter.
        output_format: ``"markdown"`` or ``"latex"``.
        profile: Pre-computed book profile; skips the profiling call.
        use_profile: Set False to translate without any book context.
        title: Document title for LaTeX output. Defaults to the work
            directory's name.

    Returns:
        Path to ``translated_merged.md`` or ``translated_book.tex``.

    Raises:
        ValueError: For an unknown ``output_format`` (checked before any work).
        TranslationError: If the LLM fails permanently on some chunk.
    """
    _validate_output_format(output_format)

    chapter_result = split_into_chapters(merged_md_path, work_dir / CHAPTERS_DIRNAME)
    checkpoint_dir = work_dir / TRANSLATED_CHAPTERS_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        (chapter, _checkpoint_path(checkpoint_dir, chapter.file_path, output_format))
        for chapter in chapter_result.chapters
    ]
    pending = [
        (chapter, checkpoint)
        for chapter, checkpoint in checkpoints
        if not (resume and _is_fresh(checkpoint, chapter.file_path))
    ]

    if pending:
        if llm is None:
            from src.llm.factory import get_llm

            llm = get_llm()
        if profile is None and use_profile:
            profile = profile_book(
                merged_md_path,
                work_dir,
                llm=llm,
                source_lang=source_lang,
                target_lang=target_lang,
            )
    else:
        logger.info("All %d chapter(s) already translated; nothing to do.", len(checkpoints))

    for position, (chapter, checkpoint) in enumerate(pending, start=1):
        logger.info(
            "Translating chapter %d/%d: %s", position, len(pending), chapter.file_path.name
        )
        source_text = chapter.file_path.read_text(encoding="utf-8")
        translated = translate_markdown(
            source_text,
            target_lang,
            source_lang=source_lang,
            llm=llm,
            profile=profile,
            output_format=output_format,
        )
        checkpoint.write_text(translated, encoding="utf-8")

    bodies = [
        checkpoint.read_text(encoding="utf-8")
        for _, checkpoint in checkpoints
        if checkpoint.exists()
    ]

    if output_format == "markdown":
        output_path = work_dir / TRANSLATED_MERGED_NAME
        output_path.write_text("\n\n".join(bodies).strip() + "\n", encoding="utf-8")
        logger.info("Wrote translated Markdown: %s", output_path)
        return output_path

    return assemble_document(
        bodies,
        profile or BookProfile.generic(),
        title or work_dir.name,
        work_dir / TRANSLATED_TEX_NAME,
        target_lang=target_lang,
    )


__all__ = [
    "OUTPUT_FORMATS",
    "TranslationError",
    "build_system_prompt",
    "chunk_markdown",
    "estimate_tokens",
    "translate_book",
    "translate_chunk",
    "translate_markdown",
]
