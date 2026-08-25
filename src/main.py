"""
CLI entry point: PDF -> Markdown (cloud or local MinerU) -> translated Markdown.

Usage
-----
    uv run python -m src.main <pdf_file> [--mode cloud|local] [--lang zh-CN]

Pipeline
--------
    1. parse_pdf(pdf_file, mode)      -> data/work/{name}/{cloud_}full.md
    2. translate_markdown(md, lang)   -> translated markdown text
    3. write data/output/{name}_translated.md

Note: translation is not wired up yet (`src/translator.py` is still a stub
under active development). Until it lands, this pipeline falls back to
passing the parsed markdown through untranslated so the parse step can be
exercised end-to-end; a clear warning is logged when that happens.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import DATA_OUTPUT
from src.parser import parse_pdf

logger = logging.getLogger(__name__)


def _get_translate_fn():
    """Import translate_markdown lazily; fall back to a passthrough with a
    warning if src/translator.py isn't implemented yet."""
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


def run(pdf_filename: str, mode: str | None, target_lang: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Step 1/3: Parsing '%s' (mode=%s)...", pdf_filename, mode or "env default")
    try:
        md_path = parse_pdf(pdf_filename, mode=mode)
    except Exception as exc:
        logger.error("Parsing failed: %s", exc)
        if (mode or "cloud") == "cloud":
            logger.error("Tip: retry with --mode local to use the local MinerU pipeline.")
        sys.exit(1)
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
        description="Parse a PDF (via MinerU cloud or local) and translate the resulting markdown.",
    )
    parser.add_argument("pdf_file", help="Name of the PDF in data/input/ (e.g., calculus.pdf)")
    parser.add_argument(
        "--mode",
        choices=["cloud", "local"],
        default=None,
        help="MinerU parsing backend. Defaults to MINERU_MODE env var (default: cloud).",
    )
    parser.add_argument(
        "--lang",
        default="English",
        help="Target translation language (default: English).",
    )
    args = parser.parse_args()

    run(args.pdf_file, args.mode, args.lang)


if __name__ == "__main__":
    main()
