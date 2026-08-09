"""
PDF -> Markdown conversion powered by MinerU (the `magic_pdf` library).

Hardware auto-detection
------------------------
This module checks ``torch.cuda.is_available()`` the moment it is imported,
*before* any `magic_pdf` submodule that touches CUDA/paddle is loaded:

* CUDA GPU present  -> left alone, MinerU will use the GPU.
* No CUDA GPU        -> ``CUDA_VISIBLE_DEVICES`` is forced to an empty string
  so MinerU (and any paddle/torch code it pulls in) silently falls back to
  CPU. No config flag or user action is required.

API note
--------
The requirements sheet referenced MinerU's older ``UNIPipe`` /
``DiskReaderWriter`` API. The version of `magic_pdf` actually installed in
this project (1.2.2) replaced that API with ``PymuDocDataset`` /
``doc_analyze`` / ``FileBasedDataWriter`` (see
``magic_pdf/tools/common.py::do_parse`` in the installed package for the
reference implementation). This module targets that current API.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.config import DATA_INPUT, DATA_WORK

logger = logging.getLogger(__name__)

# ── Hardware auto-detection (must happen before any heavy magic_pdf import) ─
import torch  # noqa: E402  (torch is bundled with magic_pdf)

_CUDA_AVAILABLE = torch.cuda.is_available()

if _CUDA_AVAILABLE:
    logger.info("CUDA GPU detected — using GPU acceleration")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("No CUDA GPU — falling back to CPU mode")


def _ensure_mineru_device_config(use_cuda: bool) -> None:
    """Keep MinerU's own `magic-pdf.json` config in sync and readable.

    MinerU reads its GPU/CPU choice from a *separate* config file
    (``~/magic-pdf.json`` by default, or ``$MINERU_TOOLS_CONFIG_JSON``) —
    not from ``CUDA_VISIBLE_DEVICES``. Some setup tools (e.g. Windows
    PowerShell's default UTF-8 output) write that file with a UTF-8 BOM,
    which `magic_pdf`'s own `json.load(..., encoding="utf-8")` call cannot
    parse. To keep GPU/CPU selection fully automatic and crash-free, we
    normalize the file to plain UTF-8 (no BOM) and sync its "device-mode"
    to our auto-detected hardware whenever it drifts.
    """
    config_name = os.getenv("MINERU_TOOLS_CONFIG_JSON", "magic-pdf.json")
    config_path = (
        Path(config_name) if os.path.isabs(config_name) else Path.home() / config_name
    )

    if not config_path.exists():
        return  # No config yet; that's a separate MinerU setup step.

    raw = config_path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"))  # tolerate a stray BOM
    except json.JSONDecodeError:
        logger.warning("Could not parse %s; leaving it untouched", config_path)
        return

    desired_device = "cuda" if use_cuda else "cpu"
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    if has_bom or data.get("device-mode") != desired_device:
        data["device-mode"] = desired_device
        config_path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Normalized %s (device-mode=%s, BOM removed=%s)",
            config_path,
            desired_device,
            has_bom,
        )


_ensure_mineru_device_config(_CUDA_AVAILABLE)


def _quiet_mineru_logging() -> None:
    """Route MinerU's loguru output through stdlib `logging` and silence
    tqdm progress bars so they don't clutter our logs."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        from loguru import logger as loguru_logger
    except ImportError:
        return

    def _sink(message) -> None:
        record = message.record
        logging.getLogger("magic_pdf").log(record["level"].no, record["message"])

    loguru_logger.remove()
    loguru_logger.add(_sink, level="WARNING", enqueue=False)


_quiet_mineru_logging()


def parse_pdf(pdf_filename: str) -> Path:
    """
    Convert a PDF in data/input/ to Markdown using MinerU.

    Automatically uses GPU if CUDA is available, otherwise CPU.

    Args:
        pdf_filename: Name of file in data/input/ (e.g., "calculus.pdf")

    Returns:
        Path to the generated full.md

    Raises:
        FileNotFoundError: If the input PDF doesn't exist
        RuntimeError: If MinerU processing fails or produces empty output
    """
    pdf_path = DATA_INPUT / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    book_name = pdf_path.stem.replace(" ", "_")
    work_dir = DATA_WORK / book_name
    images_dir = work_dir / "images"
    md_path = work_dir / "full.md"

    # ── Idempotency: skip re-parsing if already done and up to date ────────
    if md_path.exists() and md_path.stat().st_mtime > pdf_path.stat().st_mtime:
        logger.info("Already parsed, skipping: %s", md_path)
        return md_path

    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        pdf_bytes = pdf_path.read_bytes()
        image_writer = FileBasedDataWriter(str(images_dir))
        md_writer = FileBasedDataWriter(str(work_dir))

        dataset = PymuDocDataset(pdf_bytes)

        # ── TWEAK: Completely bypass dataset.classify() ────────
        # Chinese STEM PDFs often have corrupted text layers. 
        # We force OCR to ensure the AI layout model actually runs.
        use_ocr = True 
        
        infer_result = dataset.apply(
            doc_analyze,
            ocr=use_ocr,
            formula_enable=True,
            table_enable=True,
        )

        # Force the OCR pipe mode to generate the markdown
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

        pipe_result.dump_md(md_writer, md_path.name, images_dir.name)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        # Note: the input-PDF-missing FileNotFoundError is raised earlier,
        # outside this try block, so any FileNotFoundError caught here (e.g.
        # missing MinerU model weights) is a MinerU failure, not a "file
        # doesn't exist" case, and must be wrapped like any other error.
        raise RuntimeError(
            f"MinerU failed to parse '{pdf_filename}': {exc}"
        ) from exc

    if not md_path.exists() or md_path.stat().st_size == 0:
        raise RuntimeError(
            f"MinerU produced empty or missing output for '{pdf_filename}' "
            f"at {md_path}"
        )

    return md_path