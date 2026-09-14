"""
CLI entry point: PDF -> Markdown (MinerU cloud API) -> translated Markdown.

Usage
-----
    uv run python -m src.main <pdf_file> [--lang English]

This is the fully automated path, for a book you are happy to hand to the
translating model unattended. For the reviewed path -- normalize, classify
headings, convert tables, lint, then build -- use the four scripts instead:

    scripts/parse_book.py <pdf> <work-dir>
    scripts/prepare_kit.py <work-dir>
    scripts/lint_translation.py <kit-dir>
    scripts/build_pdf.py <kit-dir>

Pipeline
--------
    1. parse_pdf(pdf, work_dir)       -> data/work/{name}/merged.md
    2. translate_markdown(md, lang)   -> translated markdown text
    3. write data/output/{name}_translated.md

Translation is implemented in `src/translator.py`. This entry point runs the
whole-document path (`translate_markdown`), which chunks the Markdown and
translates it chunk by chunk. For a full book, prefer
`src.translator.translate_book`, which adds per-chapter checkpointing,
book-level profiling (`src.profiler`) and optional LaTeX output
(`src.latex`).

The passthrough fallback below is kept as a safety net: if `src.translator`
cannot be imported at all, the parse step still produces output, with a clear
warning that the text was left untranslated.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import DATA_INPUT, DATA_OUTPUT, DATA_WORK
from src.mineru_api import parse_pdf

logger = logging.getLogger(__name__)


def _get_translate_fn():
    """Import translate_markdown lazily; fall back to a passthrough with a
    warning if src.translator can't be imported (e.g. a missing optional
    dependency), so the parse step still yields output."""
    try:
        from src.translator import translate_markdown

        return translate_markdown
    except ImportError:
        logger.warning(
            "src.translator.translate_markdown not available yet; "
            "skipping translation and passing the parsed markdown through as-is."
        )

        def _passthrough(markdown_text: str, target_lang: str = "English") -> str:
            return markdown_text

        return _passthrough


def run(pdf_filename: str, target_lang: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pdf_path = DATA_INPUT / pdf_filename
    work_dir = DATA_WORK / pdf_path.stem.replace(" ", "_")

    logger.info("Step 1/3: Parsing '%s' with the MinerU cloud API...", pdf_filename)
    try:
        result = parse_pdf(pdf_path, work_dir)
    except Exception as exc:
        logger.error("Parsing failed: %s", exc)
        sys.exit(1)
    if result.failed_parts:
        logger.error(
            "%d part(s) failed and are missing from %s; not translating a partial book.",
            len(result.failed_parts),
            result.merged_path,
        )
        sys.exit(1)
    md_path = result.merged_path
    logger.info("Parsed markdown ready: %s", md_path)

    logger.info("Step 2/3: Translating to %s...", target_lang)
    translate_markdown = _get_translate_fn()
    markdown_text = md_path.read_text(encoding="utf-8")
    try:
        translated = translate_markdown(markdown_text, target_lang=target_lang)
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        sys.exit(1)

    logger.info("Step 3/3: Writing output...")
    stem = md_path.parent.name
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = DATA_OUTPUT / f"{stem}_translated.md"
    output_path.write_text(translated, encoding="utf-8")
    logger.info("Done: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Parse a PDF with the MinerU cloud API and translate the resulting markdown.",
    )
    parser.add_argument("pdf_file", help="Name of the PDF in data/input/ (e.g., calculus.pdf)")
    parser.add_argument(
        "--lang",
        default="English",
        help="Target translation language (default: English).",
    )
    args = parser.parse_args()

    run(args.pdf_file, args.lang)


if __name__ == "__main__":
    main()
