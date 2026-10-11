"""
Build and assemble a translation kit.

A "kit" is a self-contained folder handed to the translating model one chunk
at a time, plus everything needed to put the replies back together:

    <kit>/chunks/001.md ...      source chunks, image paths already tokenised
    <kit>/translated/001.tex ... the saved replies (you fill this in)
    <kit>/images/                the figures
    <kit>/image_map.json         IMG_nnnn -> real filename
    <kit>/source_clean.md        the full normalized source, for comparison
    <kit>/system_prompt.txt      the per-chunk system prompt
    <kit>/translated_book.tex    the assembled book, once the replies are in

The source chunks stay Markdown -- that is what MinerU produces -- and the
replies are LaTeX, which is why the two directories carry different suffixes.

Image tokenising is not parser-related and does not go away
------------------------------------------------------------
MinerU names images by a 64-character content hash. Models drop and transpose
characters in long opaque strings: measured at **7.7%** on this project's
book, which across 578 images is ~45 broken figures, each one silent until
the PDF build. Short ``IMG_nnnn`` tokens survive, and any that do not are
caught by ``src.lint`` because the token set is checkable and the hash set is
not.

Chunk size
----------
The binding constraint is the translating model's **output** limit, not its
context window: it has to emit a full translation of everything it is given.
See ``KIT_CHUNK_CHARS`` in ``src.config`` for the measured expansion ratios
behind the 30,000-character default.

Output is LaTeX
---------------
Replies are LaTeX body fragments, pasted into VSCode and compiled directly.
The model's known failure modes there -- invented environments, preamble
commands, source numbers repeated into headings LaTeX numbers itself -- are
each an explicit check in ``src.lint``, because every one of them looks
correct on the page and only fails at compile time.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.build import chunk_marker
from src.config import KIT_CHUNK_CHARS
from src.cover import cover_latex, find_book_meta, load_book_meta
from src.figures import DEFAULT_WIDTH, FIGURES_NAME, FigureRecord, load_figures
from src.latex import assemble_document, build_system_prompt, preamble_for
from src.lint import (
    IMG_TOKEN_RE,
    TRANSLATED_SUFFIX,
    _braced_argument,
    latex_image_refs,
    strip_sentinel,
    strip_wrapping_fence,
)
from src.profiler import BookProfile

logger = logging.getLogger(__name__)

CHUNKS_DIRNAME = "chunks"
TRANSLATED_DIRNAME = "translated"
IMAGES_DIRNAME = "images"
IMAGE_MAP_NAME = "image_map.json"
SOURCE_CLEAN_NAME = "source_clean.md"
SYSTEM_PROMPT_NAME = "system_prompt.txt"
ASSEMBLED_NAME = "translated_book.tex"

_MD_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
_HTML_IMAGE_RE = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")

# Restore only a token that fills a LaTeX argument, a markdown link target or
# an src attribute entirely. A bare `IMG_0001` in prose is the model talking
# about a figure, not referencing one, and rewriting it into a path would
# corrupt the text.
#
# The braced form is the one that matters now: a translated figure is
# `\bookfig{IMG_0042}{caption}`. The markdown and HTML forms are kept because
# the same function restores paths into source markdown as well.
_RESTORE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<=\{)(IMG_\d{4})(?=\})"),
    re.compile(r"(?<=\()(IMG_\d{4})(?=\))"),
    re.compile(r'(?<=src=")(IMG_\d{4})(?=")'),
)


@dataclass(frozen=True)
class KitResult:
    kit_dir: Path
    chunk_paths: tuple[Path, ...]
    image_map: dict[str, str]
    source_chars: int

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_paths)


# ── Image tokenising ────────────────────────────────────────────────────────


def tokenize_images(text: str) -> tuple[str, dict[str, str]]:
    """Replace every image path with an ``IMG_nnnn`` token.

    Returns ``(new_text, {token: original_path})``. A path used twice gets
    the same token both times, so the map stays a clean bijection and the
    round trip through ``restore_images`` is exact.
    """
    mapping: dict[str, str] = {}
    by_path: dict[str, str] = {}

    def token_for(path: str) -> str:
        if path not in by_path:
            token = f"IMG_{len(by_path):04d}"
            by_path[path] = token
            mapping[token] = path
        return by_path[path]

    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{token_for(match.group(2))}{match.group(3)}"

    text = _MD_IMAGE_RE.sub(replace, text)
    text = _HTML_IMAGE_RE.sub(replace, text)
    return text, mapping


def restore_images(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Put the real image paths back. Returns ``(new_text, unknown_tokens)``.

    A token with no entry in ``mapping`` is left exactly as it is and
    reported: the model corrupted it, and writing a guessed path would turn a
    findable defect into a silently wrong figure.
    """
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        path = mapping.get(token)
        if path is None:
            unknown.append(token)
            return token
        return path

    for pattern in _RESTORE_RES:
        text = pattern.sub(replace, text)
    return text, unknown


# ── Chunking ────────────────────────────────────────────────────────────────


def chunk_text(text: str, max_chars: int = KIT_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, on blank lines only.

    Never cuts inside a paragraph: a table or display-math block sliced down
    the middle produces plausible-looking, permanently corrupted output,
    whereas one oversized request either succeeds or fails loudly. A single
    paragraph over budget is therefore emitted whole, with a warning.
    """
    if not text.strip():
        return []

    paragraphs = [p for p in (p.strip("\n") for p in _PARAGRAPH_SPLIT_RE.split(text)) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        length = len(paragraph)
        if current and size + length > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        if not current and length > max_chars:
            logger.warning(
                "A single %d-character paragraph exceeds the %d-character chunk "
                "budget; keeping it whole rather than corrupting its content.",
                length,
                max_chars,
            )
        current.append(paragraph)
        size += length + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── Building ────────────────────────────────────────────────────────────────


def build_kit(
    text: str,
    kit_dir: Path,
    *,
    images_src: Path | None = None,
    profile: BookProfile | None = None,
    max_chars: int = KIT_CHUNK_CHARS,
) -> KitResult:
    """Write a translation kit for ``text`` into ``kit_dir``.

    Args:
        text: The normalized, table-converted, re-levelled markdown.
        kit_dir: Directory to create. Existing ``chunks/`` content is
            replaced so a re-run never leaves stale chunk numbers behind.
        images_src: Directory of figures to copy into the kit.
        profile: Book profile, folded into the system prompt.
        max_chars: Chunk budget.
    """
    kit_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = kit_dir / CHUNKS_DIRNAME
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir()
    (kit_dir / TRANSLATED_DIRNAME).mkdir(exist_ok=True)

    tokenized, image_map = tokenize_images(text)
    chunks = chunk_text(tokenized, max_chars)

    chunk_paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        path = chunks_dir / f"{index:03d}.md"
        path.write_text(chunk, encoding="utf-8")
        chunk_paths.append(path)

    (kit_dir / SOURCE_CLEAN_NAME).write_text(tokenized, encoding="utf-8")
    (kit_dir / IMAGE_MAP_NAME).write_text(
        json.dumps(image_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (kit_dir / SYSTEM_PROMPT_NAME).write_text(
        build_system_prompt(profile), encoding="utf-8"
    )

    if images_src is not None and images_src.is_dir():
        target = kit_dir / IMAGES_DIRNAME
        target.mkdir(exist_ok=True)
        copied = 0
        for image in images_src.iterdir():
            if image.is_file():
                shutil.copy2(image, target / image.name)
                copied += 1
        logger.info("Copied %d image(s) into %s", copied, target)

    logger.info(
        "Kit ready: %d chunk(s), %d image token(s), %d source chars in %s",
        len(chunk_paths),
        len(image_map),
        len(text),
        kit_dir,
    )
    return KitResult(kit_dir, tuple(chunk_paths), image_map, len(text))


# ── Assembling ──────────────────────────────────────────────────────────────


def assemble(kit_dir: Path) -> tuple[Path, list[str]]:
    """Concatenate the saved translations into a compilable ``.tex``.

    The fragments are joined in chunk order, real image filenames are
    restored from ``image_map.json``, and the result is wrapped in
    ``assets/preamble.tex``. The output lands in ``kit_dir``, next to
    ``images/``, because the preamble's ``\\graphicspath`` looks for figures
    there relative to the ``.tex``.

    Returns ``(assembled_path, problems)``. Problems are described, not
    raised on -- ``src.lint`` is what decides whether the build may proceed.
    """
    chunks_dir = kit_dir / CHUNKS_DIRNAME
    translated_dir = kit_dir / TRANSLATED_DIRNAME
    problems: list[str] = []

    sources = sorted(chunks_dir.glob("*.md"))
    if not sources:
        raise FileNotFoundError(f"No source chunks in {chunks_dir}")

    parts: list[str] = []
    for source_path in sources:
        target_path = translated_dir / f"{source_path.stem}{TRANSLATED_SUFFIX}"
        if not target_path.exists():
            problems.append(f"chunk {source_path.stem} has no translation saved")
            continue
        raw = target_path.read_text(encoding="utf-8")
        # The `% END-OF-CHUNK` sentinel has done its job once lint has seen it.
        body = strip_sentinel(strip_wrapping_fence(raw)).strip()
        # A comment naming the fragment and the line its body starts on, so
        # an error at line 6,532 of the assembled book can be reported as
        # translated/008.tex:39 -- see `src.build.collect_errors`.
        first_line = raw[: raw.find(body)].count("\n") + 1 if body else 1
        label = f"{TRANSLATED_DIRNAME}/{target_path.name}"
        parts.append(f"{chunk_marker(label, first_line)}\n{body}")

    merged = "\n\n".join(parts)

    map_path = kit_dir / IMAGE_MAP_NAME
    if map_path.exists():
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        figures_path = kit_dir / FIGURES_NAME
        if figures_path.exists():
            merged, figure_problems = apply_figures(merged, mapping, load_figures(figures_path))
            problems += figure_problems
        merged, unknown = restore_images(merged, mapping)
        if unknown:
            problems.append(
                f"{len(unknown)} unknown image token(s) left as-is: "
                f"{sorted(set(unknown))[:5]}"
            )
        still_tokenized = set(IMG_TOKEN_RE.findall(merged))
        if still_tokenized:
            problems.append(
                f"{len(still_tokenized)} image token(s) were not in a figure "
                "argument and stayed as text"
            )

    # The cover page, from book_meta.json. Its absence is reported, never
    # a problem: problems gate the build, and a missing cover is cosmetic.
    meta, notes = load_book_meta(find_book_meta(kit_dir))
    for note in notes:
        logger.warning("Cover: %s", note)

    out_path = assemble_document(
        [merged],
        kit_dir / ASSEMBLED_NAME,
        preamble_path=preamble_for(kit_dir),
        cover=cover_latex(meta) if meta else "",
    )
    logger.info("Assembled %s (%d chars of translated body)", out_path, len(merged))
    return out_path, problems


# ── Figures ─────────────────────────────────────────────────────────────────

_BOOKFIG_RE = re.compile(r"\\bookfig(two)?(?![A-Za-z])")

# The pair never exceeds the text width: two minipages plus the
# 0.04\textwidth gap \bookfigtwox puts between them.
_PAIR_BUDGET = 0.96


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _parse_bookfig(text: str, match: re.Match[str]) -> tuple[list[str], int] | None:
    """The arguments of the ``\\bookfig`` / ``\\bookfigtwo`` at ``match``.

    Returns ``(arguments, end)`` or None when the call is malformed, which is
    left untouched for lint to report. An optional ``[width]`` is skipped:
    ``figures.json`` decides the width, whatever the model wrote.
    """
    count = 5 if match.group(1) else 2
    index = _skip_space(text, match.end())
    if not match.group(1) and index < len(text) and text[index] == "[":
        close = text.find("]", index)
        if close == -1:
            return None
        index = _skip_space(text, close + 1)
    arguments: list[str] = []
    for _ in range(count):
        index = _skip_space(text, index)
        if index >= len(text) or text[index] != "{":
            return None
        found = _braced_argument(text, index)
        if found is None:
            return None
        argument, index = found
        arguments.append(argument)
    return arguments, index


def _in_comment(text: str, index: int) -> bool:
    line_start = text.rfind("\n", 0, index) + 1
    return re.search(r"(?<!\\)%", text[line_start:index]) is not None


def _width(value: float) -> str:
    return f"{value:.2f}"


def apply_figures(
    text: str, mapping: dict[str, str], records: dict[str, FigureRecord]
) -> tuple[str, list[str]]:
    """Rewrite the model's figure calls from ``figures.json``.

    * ``\\bookfig{IMG_0042}{caption}`` -> ``\\bookfigx{0.42}{IMG_0042}{18.3}{caption}``
      (the token is restored to a path afterwards, like every other one);
    * a width the model emitted, ``\\bookfig[0.85]{...}``, is ignored;
    * a dropped image's call is removed entirely;
    * ``\\bookfigtwo`` becomes ``\\bookfigtwox``, each panel at its own
      width, with the number of whichever panel carries one. If only one
      panel was dropped, both are kept and the pair is reported.

    Captions are brace-matched, never regex-matched: they contain math
    with nested groups. Returns ``(text, problems)``.
    """
    problems: list[str] = []
    missing: list[str] = []
    pieces: list[str] = []
    cursor = 0
    rewritten = dropped = 0

    def record_for(token: str) -> FigureRecord | None:
        path = mapping.get(token.strip())
        if path is None:
            return None  # a corrupted token: restore_images reports it
        record = records.get(Path(path).name)
        if record is None:
            missing.append(token.strip())
        return record

    for match in _BOOKFIG_RE.finditer(text):
        if match.start() < cursor or _in_comment(text, match.start()):
            continue
        parsed = _parse_bookfig(text, match)
        if parsed is None:
            continue
        arguments, end = parsed

        if not match.group(1):
            token, caption = arguments
            if token.strip() not in mapping:
                continue
            record = record_for(token) or FigureRecord()
            if not record.keep:
                replacement = ""
                dropped += 1
            else:
                replacement = (
                    f"\\bookfigx{{{_width(record.width)}}}{{{token.strip()}}}"
                    f"{{{record.number or ''}}}{{{caption}}}"
                )
                rewritten += 1
        else:
            first, first_label, second, second_label, caption = arguments
            if first.strip() not in mapping or second.strip() not in mapping:
                continue
            a = record_for(first) or FigureRecord()
            b = record_for(second) or FigureRecord()
            if not a.keep and not b.keep:
                replacement = ""
                dropped += 1
            else:
                if a.keep != b.keep:
                    gone = first if not a.keep else second
                    problems.append(
                        f"\\bookfigtwo{{{first.strip()}}}...{{{second.strip()}}}: {gone.strip()} "
                        "is marked dropped but its pair is kept, so both were kept -- "
                        "review it in figures.json"
                    )
                if a.number and b.number and a.number != b.number:
                    problems.append(
                        f"\\bookfigtwo{{{first.strip()}}}...{{{second.strip()}}} pairs two "
                        f"numbered figures ({a.number}, {b.number}); printed as {b.number}"
                    )
                number = b.number or a.number or ""
                scale = min(1.0, _PAIR_BUDGET / (a.width + b.width))
                replacement = (
                    f"\\bookfigtwox{{{_width(a.width * scale)}}}{{{first.strip()}}}{{{first_label}}}"
                    f"{{{_width(b.width * scale)}}}{{{second.strip()}}}{{{second_label}}}"
                    f"{{{number}}}{{{caption}}}"
                )
                rewritten += 1

        pieces.append(text[cursor : match.start()])
        pieces.append(replacement)
        cursor = end

    pieces.append(text[cursor:])
    if missing:
        problems.append(
            f"{len(missing)} figure(s) have no {FIGURES_NAME} record and were set at the "
            f"default width ({DEFAULT_WIDTH}) with no number: {sorted(set(missing))[:5]}"
        )
    logger.info("Figures: %d rewritten from %s, %d dropped", rewritten, FIGURES_NAME, dropped)
    return "".join(pieces), problems


def verify_images(text: str, kit_dir: Path) -> list[str]:
    """Every image reference in ``text`` that does not resolve on disk.

    Handles both sides of the pipeline: markdown image syntax for the source,
    ``\\bookfig`` / ``\\includegraphics`` for the translation. Called before
    packaging because a silent path mismatch once produced a zip with zero
    usable figures -- and the zip looked fine.
    """
    refs = [m.group(2) for m in _MD_IMAGE_RE.finditer(text)]
    refs += [m.group(2) for m in _HTML_IMAGE_RE.finditer(text)]
    refs += latex_image_refs(text)

    missing: list[str] = []
    for ref in dict.fromkeys(refs):
        if ref.startswith(("http://", "https://", "data:")):
            continue
        if (kit_dir / ref).exists() or (kit_dir / IMAGES_DIRNAME / Path(ref).name).exists():
            continue
        missing.append(ref)
    return missing


def missing_kit_images(kit_dir: Path) -> list[str]:
    """Every path in ``image_map.json`` with no file behind it.

    The map is what has to be checked, not ``source_clean.md``: that file is
    tokenised, so its references are ``IMG_nnnn`` rather than paths. The
    map's *values* are the real filenames the assembled book will point at.
    """
    map_path = kit_dir / IMAGE_MAP_NAME
    if not map_path.exists():
        # No map means no tokenising happened; fall back to the source's own
        # references.
        source_path = kit_dir / SOURCE_CLEAN_NAME
        if not source_path.exists():
            return []
        return verify_images(source_path.read_text(encoding="utf-8"), kit_dir)

    mapping: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for path in dict.fromkeys(mapping.values()):
        if path.startswith(("http://", "https://", "data:")):
            continue
        if (kit_dir / path).exists() or (kit_dir / IMAGES_DIRNAME / Path(path).name).exists():
            continue
        missing.append(path)
    return missing


def package_kit(kit_dir: Path, archive_path: Path | None = None) -> Path:
    """Zip ``kit_dir`` for handoff, refusing to if any figure is missing.

    Raises:
        FileNotFoundError: If an image reference does not resolve. Shipping a
            kit whose figures are already broken just moves the failure to
            someone who cannot diagnose it -- a silent mismatch once produced
            a zip with zero usable figures.
    """
    missing = missing_kit_images(kit_dir)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by {IMAGE_MAP_NAME} do not "
            f"resolve under {kit_dir}: {missing[:5]}. Re-run the parse, or copy "
            "the images in, before packaging."
        )

    archive_path = archive_path or kit_dir.with_suffix(".zip")
    made = shutil.make_archive(
        str(archive_path.with_suffix("")), "zip", root_dir=kit_dir.parent, base_dir=kit_dir.name
    )
    logger.info("Packaged %s", made)
    return Path(made)


__all__ = [
    "ASSEMBLED_NAME",
    "IMAGE_MAP_NAME",
    "KitResult",
    "apply_figures",
    "assemble",
    "build_kit",
    "chunk_text",
    "missing_kit_images",
    "package_kit",
    "restore_images",
    "tokenize_images",
    "verify_images",
]
