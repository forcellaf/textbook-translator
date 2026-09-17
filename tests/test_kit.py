"""Tests for src.kit (image tokenising, chunking, assembly, packaging).

The round-trip test is the important one. MinerU names images by a
64-character content hash, and the translating model corrupted those at a
measured 7.7% -- roughly 45 broken figures across 578 images, each silent
until the PDF build. Tokenising is what makes that failure detectable, so the
round trip has to be exact -- including through the LaTeX form the tokens
come back in, ``\\bookfig{IMG_0042}{caption}``.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from src.config import KIT_CHUNK_CHARS
from src.kit import (
    ASSEMBLED_NAME,
    assemble,
    build_kit,
    chunk_text,
    missing_kit_images,
    package_kit,
    restore_images,
    tokenize_images,
    verify_images,
)
from src.profiler import BookProfile

HASH_A = "6a4664edf518e3474010602a246bff85109b125b602b19133d3bf76d48010c6d.jpg"
HASH_B = "ce7af3353cafb1af097e9398bdbd331c09d518d8a1d5817e6cf1cf8bca4d1732.jpg"


# ── Tokenising round trip ───────────────────────────────────────────────────


def test_tokenise_restore_round_trip_preserves_every_path_exactly() -> None:
    text = (
        f"Prose one.\n\n![figure 12.1](images/{HASH_A})\n\n"
        f"Prose two.\n\n![](images/{HASH_B})\n\n"
        f'Prose three.\n\n<img src="images/{HASH_A}">\n'
    )
    tokenized, mapping = tokenize_images(text)

    assert HASH_A not in tokenized and HASH_B not in tokenized
    assert "IMG_0000" in tokenized and "IMG_0001" in tokenized

    restored, unknown = restore_images(tokenized, mapping)
    assert restored == text
    assert unknown == []


def test_the_round_trip_survives_the_translation_into_latex() -> None:
    """The real path: a token goes out in markdown image syntax and comes
    back as a \\bookfig argument. Restoration has to find it there."""
    source = f"![图 12.3 电场分布](images/{HASH_A})\n\n![](images/{HASH_B})\n"
    tokenized, mapping = tokenize_images(source)

    translated = (
        "\\bookfig{IMG_0000}{Field distribution}\n\n"
        "\\bookfigtwo{IMG_0000}{(a)}{IMG_0001}{(b)}{Both halves}\n\n"
        "\\includegraphics[width=0.5\\textwidth]{IMG_0001}\n"
    )
    restored, unknown = restore_images(translated, mapping)

    assert unknown == []
    assert "IMG_" not in restored
    assert restored.count(f"images/{HASH_A}") == 2
    assert restored.count(f"images/{HASH_B}") == 2


def test_a_repeated_path_reuses_one_token() -> None:
    text = f"![a](images/{HASH_A})\n\n![b](images/{HASH_A})\n"
    tokenized, mapping = tokenize_images(text)

    assert len(mapping) == 1
    assert tokenized.count("IMG_0000") == 2


def test_alt_text_is_left_alone_for_the_translator() -> None:
    text = f"![电场强度分布图](images/{HASH_A})\n"
    tokenized, _ = tokenize_images(text)

    assert "电场强度分布图" in tokenized


def test_an_unknown_token_is_left_as_is_and_reported() -> None:
    """A corrupted token must stay findable, not become a guessed path."""
    restored, unknown = restore_images("![a](IMG_9999)\n", {"IMG_0000": f"images/{HASH_A}"})

    assert "IMG_9999" in restored
    assert unknown == ["IMG_9999"]


def test_a_token_mentioned_in_prose_is_not_rewritten() -> None:
    """`IMG_0000` in a sentence is the model talking about a figure, not
    referencing one; rewriting it would corrupt the text."""
    mapping = {"IMG_0000": f"images/{HASH_A}"}
    text = "As IMG_0000 shows, the field is radial.\n"
    restored, unknown = restore_images(text, mapping)

    assert restored == text
    assert unknown == []


def test_no_images_is_a_clean_no_op() -> None:
    text = "Prose with $x$ and no figures.\n"
    tokenized, mapping = tokenize_images(text)

    assert tokenized == text
    assert mapping == {}


# ── Chunking ────────────────────────────────────────────────────────────────


def test_chunks_split_only_on_paragraph_boundaries() -> None:
    paragraphs = [f"Paragraph {i} " + "x" * 200 for i in range(20)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=600)

    assert len(chunks) > 1
    # Nothing was cut mid-paragraph: every original paragraph survives whole.
    rejoined = "\n\n".join(chunks)
    for paragraph in paragraphs:
        assert paragraph in rejoined


def test_a_table_block_is_never_cut_in_half() -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    text = f"Intro.\n\n{table}\n\nOutro."
    chunks = chunk_text(text, max_chars=30)

    assert any(table in chunk for chunk in chunks)


def test_an_oversized_paragraph_is_emitted_whole_with_a_warning(caplog) -> None:
    giant = "x" * 500
    chunks = chunk_text(giant, max_chars=100)

    assert chunks == [giant]
    assert any("exceeds" in r.message for r in caplog.records)


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_the_budget_when_they_can() -> None:
    text = "\n\n".join("y" * 90 for _ in range(10))
    for chunk in chunk_text(text, max_chars=200):
        assert len(chunk) <= 200


# ── Building and assembling ─────────────────────────────────────────────────


def _seed_kit(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    (images / HASH_A).write_bytes(b"\x89PNG-a")
    (images / HASH_B).write_bytes(b"\x89PNG-b")

    text = (
        f"# Chapter\n\n{'a' * 300}\n\n![one](images/{HASH_A})\n\n"
        f"{'b' * 300}\n\n![two](images/{HASH_B})\n"
    )
    build_kit(text, tmp_path / "kit", images_src=images, max_chars=400)
    return tmp_path / "kit"


def test_build_kit_writes_every_expected_artifact(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)

    assert sorted(p.name for p in (kit / "chunks").glob("*.md")) == ["001.md", "002.md"]
    assert (kit / "source_clean.md").exists()
    assert (kit / "system_prompt.txt").exists()
    assert (kit / "translated").is_dir()
    assert (kit / "images" / HASH_A).read_bytes() == b"\x89PNG-a"

    mapping = json.loads((kit / "image_map.json").read_text(encoding="utf-8"))
    assert sorted(mapping) == ["IMG_0000", "IMG_0001"]


def test_the_kit_system_prompt_is_the_asset_with_the_glossary_folded_in(
    tmp_path: Path,
) -> None:
    """What the user pastes into AI Studio is the tracked prompt asset, not
    something assembled from string literals in the code."""
    build_kit(
        "# Chapter\n\nBody.\n",
        tmp_path / "kit",
        profile=BookProfile(subject="physics", glossary=(("电荷", "electric charge"),)),
    )

    prompt = (tmp_path / "kit" / "system_prompt.txt").read_text(encoding="utf-8")

    assert "\\bookfig{IMG_0042}" in prompt  # straight from the asset
    assert "电荷 -> electric charge" in prompt
    assert "{{BOOK_CONTEXT_AND_GLOSSARY}}" not in prompt


def test_the_default_chunk_budget_is_the_configured_one(tmp_path: Path) -> None:
    """30,000 source characters: the measured worst-case expansion (2.87x)
    on a LaTeX-emitting run has to stay inside the model's output limit."""
    assert KIT_CHUNK_CHARS == 30_000

    paragraphs = "\n\n".join("x" * 5_000 for _ in range(12))
    build_kit(paragraphs, tmp_path / "kit")

    for chunk in (tmp_path / "kit" / "chunks").glob("*.md"):
        assert len(chunk.read_text(encoding="utf-8")) <= KIT_CHUNK_CHARS


def test_rebuilding_a_kit_does_not_leave_stale_chunks(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)
    (kit / "chunks" / "099.md").write_text("stale", encoding="utf-8")

    build_kit("# Short\n\nOne paragraph.\n", kit, max_chars=400)

    assert [p.name for p in (kit / "chunks").glob("*.md")] == ["001.md"]


def _translate(kit: Path, wrap: Callable[[str], str] = lambda body: body) -> None:
    """Save a plausible LaTeX reply for every chunk, keeping its tokens."""
    for index, chunk in enumerate(sorted((kit / "chunks").glob("*.md")), start=1):
        tokens = re.findall(r"IMG_\d{4}", chunk.read_text(encoding="utf-8"))
        parts = [f"\\section{{Part {index}}}"]
        parts += [f"\\bookfig{{{token}}}{{Figure {index}}}" for token in tokens]
        (kit / "translated" / f"{chunk.stem}.tex").write_text(
            wrap("\n\n".join(parts)), encoding="utf-8"
        )


def test_assemble_restores_real_paths_and_wraps_the_preamble(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)
    _translate(kit)

    assembled, problems = assemble(kit)
    tex = assembled.read_text(encoding="utf-8")

    assert problems == []
    assert HASH_A in tex and HASH_B in tex
    assert "IMG_0000" not in tex
    assert assembled.name == ASSEMBLED_NAME == "translated_book.tex"
    # Compilable as it stands, and sitting next to images/ as \graphicspath
    # expects.
    assert tex.count("\\begin{document}") == 1
    assert tex.count("\\end{document}") == 1
    assert (assembled.parent / "images").is_dir()


def test_assemble_reports_a_chunk_with_no_translation_saved(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)
    (kit / "translated" / "001.tex").write_text("\\section{One}", encoding="utf-8")

    _, problems = assemble(kit)
    assert any("002" in p for p in problems)


def test_assemble_strips_a_wrapping_latex_fence(tmp_path: Path) -> None:
    """AI Studio wraps the reply in ```latex regardless of the prompt."""
    kit = _seed_kit(tmp_path)
    _translate(kit, wrap=lambda body: f"```latex\n{body}\n```")

    assembled, _ = assemble(kit)
    assert "```" not in assembled.read_text(encoding="utf-8")


# ── Packaging ───────────────────────────────────────────────────────────────


def test_verify_images_finds_a_reference_with_no_file(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)
    missing = verify_images(f"![a](images/{HASH_A})\n\n![b](images/nope.jpg)\n", kit)

    assert missing == ["images/nope.jpg"]


def test_verify_images_checks_bookfig_references_too(tmp_path: Path) -> None:
    """After assembly the references are LaTeX, not markdown."""
    kit = _seed_kit(tmp_path)
    text = (
        f"\\bookfig{{images/{HASH_A}}}{{There}}\n\n"
        "\\bookfig[0.9]{images/gone.jpg}{Not there}\n"
    )

    assert verify_images(text, kit) == ["images/gone.jpg"]


def test_the_preambles_own_macro_definition_is_not_a_missing_figure(
    tmp_path: Path,
) -> None:
    """\\bookfig is defined as \\includegraphics{#2}; `#2` is not a path."""
    kit = _seed_kit(tmp_path)
    _translate(kit)
    assembled, _ = assemble(kit)

    assert verify_images(assembled.read_text(encoding="utf-8"), kit) == []


def test_missing_kit_images_checks_the_map_not_the_tokenised_source(tmp_path: Path) -> None:
    """source_clean.md references IMG_nnnn, so only image_map.json's values
    can be checked against the disk."""
    kit = _seed_kit(tmp_path)
    assert missing_kit_images(kit) == []

    (kit / "images" / HASH_A).unlink()
    assert missing_kit_images(kit) == [f"images/{HASH_A}"]


def test_packaging_refuses_when_a_figure_is_missing(tmp_path: Path) -> None:
    """A silent mismatch once produced a zip with zero usable figures."""
    kit = _seed_kit(tmp_path)
    (kit / "images" / HASH_A).unlink()

    with pytest.raises(FileNotFoundError, match="do not resolve"):
        package_kit(kit)


def test_packaging_succeeds_when_every_figure_resolves(tmp_path: Path) -> None:
    kit = _seed_kit(tmp_path)
    archive = package_kit(kit)

    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert any(name.endswith("image_map.json") for name in names)
    assert any(HASH_A in name for name in names)
