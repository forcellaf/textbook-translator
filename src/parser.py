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


def parse_pdf_local(pdf_filename: str) -> Path:
    """
    Convert a PDF in data/input/ to Markdown using the local MinerU
    (`magic_pdf`) library.

    Automatically uses GPU if CUDA is available, otherwise CPU. This path is
    CPU-slow and lower quality than the cloud API, but is kept available for
    offline use and future CUDA testing.

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
    return _parse_pdf_path_local(pdf_path, work_dir)


def _parse_pdf_path_local(pdf_path: Path, work_dir: Path) -> Path:
    """
    Core local-MinerU parse logic, decoupled from data/input/ naming
    conventions so it can be reused for both whole-book parsing
    (`parse_pdf_local`) and per-chunk parsing of a split PDF
    (`parse_book`), which lives outside data/input/.

    Args:
        pdf_path: Path to the PDF to parse (need not be in DATA_INPUT).
        work_dir: Directory to write `full.md` and `images/` into.

    Returns:
        Path to the generated full.md

    Raises:
        RuntimeError: If MinerU processing fails or produces empty output
    """
    images_dir = work_dir / "images"
    md_path = work_dir / "full.md"

    # ── Idempotency: skip re-parsing if already done and up to date ────────
    if md_path.exists() and md_path.stat().st_mtime > pdf_path.stat().st_mtime:
        logger.info("Already parsed, skipping: %s", md_path)
        return md_path

    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = pdf_path.name
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
            lang="ch"
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


def parse_pdf_cloud(pdf_filename: str) -> Path:
    """
    Convert a PDF in data/input/ to Markdown using the MinerU Cloud API (v4).

    Uploads the PDF, polls for completion, and downloads the resulting
    markdown. This is the default/primary path: faster and higher quality
    than the local CPU pipeline. Raw responses and the final markdown are
    mirrored to data/work/{filename}/ for debugging.

    Args:
        pdf_filename: Name of file in data/input/ (e.g., "calculus.pdf")

    Returns:
        Path to the generated cloud_full.md

    Raises:
        FileNotFoundError: If the input PDF doesn't exist
        RuntimeError: If MINERU_API_KEY is missing or the API call fails
        TimeoutError: If the cloud task does not complete in time
    """
    pdf_path = DATA_INPUT / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    book_name = pdf_path.stem.replace(" ", "_")
    work_dir = DATA_WORK / book_name
    return _parse_pdf_path_cloud(pdf_path, work_dir)


def _parse_pdf_path_cloud(pdf_path: Path, work_dir: Path) -> Path:
    """
    Core MinerU-Cloud parse logic, decoupled from data/input/ naming
    conventions so it can be reused for both whole-book parsing
    (`parse_pdf_cloud`) and per-chunk parsing of a split PDF
    (`parse_book`), which lives outside data/input/.

    Args:
        pdf_path: Path to the PDF to parse (need not be in DATA_INPUT).
        work_dir: Directory to write `cloud_full.md` (and debug artifacts)
            into.

    Returns:
        Path to the generated cloud_full.md

    Raises:
        RuntimeError: If MINERU_API_KEY is missing or the API call fails
        TimeoutError: If the cloud task does not complete in time
    """
    from src.mineru_client import MinerUCloudClient, MinerUCloudError

    pdf_filename = pdf_path.name
    md_path = work_dir / "cloud_full.md"

    # ── Idempotency: skip re-parsing if already done and up to date ────────
    if md_path.exists() and md_path.stat().st_mtime > pdf_path.stat().st_mtime:
        logger.info("Already parsed (cloud), skipping: %s", md_path)
        return md_path

    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        with MinerUCloudClient(work_dir=work_dir) as client:
            task_id = client.upload_pdf(str(pdf_path))
            markdown = client.wait_for_completion(
                task_id, timeout_minutes=os_env_timeout_minutes()
            )
    except MinerUCloudError as exc:
        raise RuntimeError(
            f"MinerU Cloud failed to parse '{pdf_filename}': {exc}\n"
            "Try again, or retry with --mode local to use the local pipeline."
        ) from exc
    except TimeoutError as exc:
        raise TimeoutError(
            f"{exc}\nRetry later, or use --mode local to use the local pipeline."
        ) from exc

    md_path.write_text(markdown, encoding="utf-8")

    if not md_path.exists() or md_path.stat().st_size == 0:
        raise RuntimeError(
            f"MinerU Cloud produced empty or missing output for '{pdf_filename}' "
            f"at {md_path}"
        )

    return md_path


def os_env_timeout_minutes() -> int:
    """Small indirection so timeout can be tuned via MINERU_TIMEOUT_MINUTES."""
    from src.config import MINERU_TIMEOUT_MINUTES

    return MINERU_TIMEOUT_MINUTES


def parse_pdf(pdf_filename: str, mode: str | None = None) -> Path:
    """
    Unified entry point: convert a PDF in data/input/ to Markdown, routing to
    the cloud API or the local MinerU library.

    Args:
        pdf_filename: Name of file in data/input/ (e.g., "calculus.pdf")
        mode: "cloud" or "local". Defaults to the MINERU_MODE env var
            (itself defaulting to "cloud") when not given.

    Returns:
        Path to the generated markdown file (cloud_full.md or full.md).

    Raises:
        ValueError: If mode is neither "cloud" nor "local"
        FileNotFoundError: If the input PDF doesn't exist
        RuntimeError: If parsing fails
    """
    from src.config import MINERU_MODE

    resolved_mode = (mode or MINERU_MODE or "cloud").lower()

    if resolved_mode == "cloud":
        return parse_pdf_cloud(pdf_filename)
    if resolved_mode == "local":
        return parse_pdf_local(pdf_filename)

    raise ValueError(
        f"Unknown MinerU mode '{resolved_mode}'. Expected 'cloud' or 'local'."
    )


def _parse_pdf_path(pdf_path: Path, work_dir: Path, mode: str) -> Path:
    """Dispatch a single already-resolved PDF path to the cloud or local
    MinerU backend. Shared by `parse_pdf` (via the filename-based wrappers
    above) and `parse_book`'s per-chunk parsing."""
    if mode == "cloud":
        return _parse_pdf_path_cloud(pdf_path, work_dir)
    if mode == "local":
        return _parse_pdf_path_local(pdf_path, work_dir)
    raise ValueError(f"Unknown MinerU mode '{mode}'. Expected 'cloud' or 'local'.")


def parse_book(pdf_filename: str, mode: str | None = None) -> Path:
    """
    Orchestrator entry point: convert a PDF in data/input/ to a single
    merged Markdown file, transparently splitting it first if it exceeds
    MinerU's Precision Extract API limits (<= `SPLIT_MAX_PAGES` pages,
    <= `SPLIT_MAX_SIZE_MB` MB).

    Behavior:
        * If `PDF_SPLIT_ENABLED` is False, or the PDF is within both
          limits, this simply delegates to `parse_pdf` (identical output,
          identical behavior -- no splitting/merging overhead).
        * Otherwise: the PDF is split into overlapping page-range chunks
          (`src.splitter.split_pdf`), each chunk is parsed with the same
          MinerU backend used elsewhere in this module (into
          `data/work/<book>/chunks/<chunk_stem>/`), and the per-chunk
          Markdown is merged back into `data/work/<book>/merged.md`
          (`src.merger.merge_chunks`), which is what's returned.

    Args:
        pdf_filename: Name of file in data/input/ (e.g., "calculus.pdf")
        mode: "cloud" or "local". Defaults to the MINERU_MODE env var
            (itself defaulting to "cloud") when not given.

    Returns:
        Path to the final markdown for the whole book: either the plain
        `parse_pdf` output (full.md/cloud_full.md) when no split was
        needed, or `merged.md` when it was.

    Raises:
        ValueError: If mode is neither "cloud" nor "local"
        FileNotFoundError: If the input PDF doesn't exist
        RuntimeError: If the PDF is encrypted, unreadable, or parsing fails
    """
    import pymupdf

    from src.config import (
        MINERU_MODE,
        PDF_SPLIT_ENABLED,
        SPLIT_MAX_PAGES,
        SPLIT_MAX_SIZE_MB,
        SPLIT_OVERLAP_PAGES,
    )
    from src.merger import merge_chunks
    from src.splitter import split_pdf

    pdf_path = DATA_INPUT / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    resolved_mode = (mode or MINERU_MODE or "cloud").lower()
    if resolved_mode not in ("cloud", "local"):
        raise ValueError(
            f"Unknown MinerU mode '{resolved_mode}'. Expected 'cloud' or 'local'."
        )

    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = doc.page_count
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    within_limits = page_count <= SPLIT_MAX_PAGES and size_mb <= SPLIT_MAX_SIZE_MB

    if not PDF_SPLIT_ENABLED or within_limits:
        logger.info(
            "'%s' (%d pages, %.1f MB) is within limits or splitting is disabled; "
            "parsing directly",
            pdf_filename,
            page_count,
            size_mb,
        )
        return parse_pdf(pdf_filename, mode=resolved_mode)

    logger.info(
        "'%s' (%d pages, %.1f MB) exceeds split limits (%d pages / %d MB); splitting first",
        pdf_filename,
        page_count,
        size_mb,
        SPLIT_MAX_PAGES,
        SPLIT_MAX_SIZE_MB,
    )

    book_name = pdf_path.stem.replace(" ", "_")
    book_work_dir = DATA_WORK / book_name
    chunks_dir = book_work_dir / "chunks"

    split_result = split_pdf(
        pdf_path,
        chunks_dir,
        max_pages=SPLIT_MAX_PAGES,
        max_size_mb=SPLIT_MAX_SIZE_MB,
        overlap_pages=SPLIT_OVERLAP_PAGES,
    )

    for chunk in split_result.chunks:
        chunk_work_dir = chunks_dir / chunk.output_path.stem
        logger.info(
            "Parsing chunk %d/%d (pages %d-%d)...",
            chunk.index,
            len(split_result.chunks),
            chunk.start_page,
            chunk.end_page,
        )
        _parse_pdf_path(chunk.output_path, chunk_work_dir, resolved_mode)

    return merge_chunks(list(split_result.chunks), book_work_dir)