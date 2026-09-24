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
from src.latex import assemble_document, build_system_prompt
from src.lint import IMG_TOKEN_RE, TRANSLATED_SUFFIX, latex_image_refs, strip_wrapping_fence
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
        body = strip_wrapping_fence(raw).strip()
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

    out_path = assemble_document([merged], kit_dir / ASSEMBLED_NAME)
    logger.info("Assembled %s (%d chars of translated body)", out_path, len(merged))
    return out_path, problems


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
    "assemble",
    "build_kit",
    "chunk_text",
    "missing_kit_images",
    "package_kit",
    "restore_images",
    "tokenize_images",
    "verify_images",
]
