"""
Central configuration module.

Loads environment variables from `.env`, sets up optional proxy/network
environment variables BEFORE any network-using library is imported, validates
required secrets, and exposes typed path/setting constants used across the
whole pipeline.

IMPORTANT: This module must be imported (directly or transitively) before any
LLM provider SDK or MinerU code runs, since it is responsible for setting
HTTP_PROXY / HTTPS_PROXY / HF_ENDPOINT in os.environ.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load `.env` from the project root (no-op if the file doesn't exist).
load_dotenv(PROJECT_ROOT / ".env")

DATA_INPUT = PROJECT_ROOT / "data" / "input"
DATA_WORK = PROJECT_ROOT / "data" / "work"
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Optional proxy / network setup ──────────────────────────────────────────
# Only set these if a non-empty value is provided. Leaving them unset allows
# a direct internet connection. This MUST happen before any LLM client or
# MinerU is initialized.
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "HF_ENDPOINT"):
    _value = os.getenv(_var)
    if _value:
        os.environ[_var] = _value

# ── Secrets ─────────────────────────────────────────────────────────────────
# Keys are read here but NOT validated here; see "Provider credentials" below,
# which validates only the key belonging to the active LLM_PROVIDER.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# ── Settings ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")
SOURCE_LANG: str = os.getenv("SOURCE_LANG", "Chinese")
TARGET_LANG: str = os.getenv("TARGET_LANG", "English")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "3000"))
API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "5"))
MAX_HEAL_ATTEMPTS: int = int(os.getenv("MAX_HEAL_ATTEMPTS", "3"))

# Fraction of a chunk's non-math characters that may still be written in the
# source script before the output is treated as untranslated and re-asked.
# Calibrated against a real run: the source Markdown of the physics textbook is
# ~35% CJK by character, a chunk the model passed through verbatim measured
# ~30%, and a genuinely translated chunk measures ~0% -- the handful of
# characters a legitimate proper noun or a "leave as-is" glossary term
# contributes stays under 1% of a 2000-character chunk. 4% sits an order of
# magnitude below the failure signal and several times above the legitimate
# ceiling. Raise it for a book that deliberately keeps source-script terms.
SOURCE_RESIDUE_THRESHOLD: float = float(os.getenv("SOURCE_RESIDUE_THRESHOLD", "0.04"))

# ── MinerU (PDF -> Markdown) ────────────────────────────────────────────────
# `MINERU_MODE` selects the default parsing backend for `src.parser.parse_pdf`:
#   "cloud" (default) - MinerU Cloud API (v4), fast, no local GPU/CPU cost.
#   "local"           - local `magic_pdf` library (kept for future CUDA use).
# MINERU_API_KEY is not validated here because not every invocation needs it
# (e.g. running only the local parser, or only the translator). The MinerU
# client validates it lazily, at construction time.
MINERU_MODE: str = os.getenv("MINERU_MODE", "cloud").lower()
MINERU_API_KEY: str | None = os.getenv("MINERU_API_KEY")
MINERU_API_BASE: str = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4").rstrip("/")
MINERU_MAX_FILE_MB: int = int(os.getenv("MINERU_MAX_FILE_MB", "200"))
MINERU_POLL_INTERVAL_SECONDS: int = int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "10"))
MINERU_TIMEOUT_MINUTES: int = int(os.getenv("MINERU_TIMEOUT_MINUTES", "30"))
# One of "pipeline", "vlm", "MinerU-HTML" (per MinerU v4 "precision parse" API).
MINERU_MODEL_VERSION: str = os.getenv("MINERU_MODEL_VERSION", "vlm")
# Upload to Aliyun OSS can be slow/unstable on some networks; these control
# how patiently/persistently we retry the raw PUT upload.
MINERU_UPLOAD_MAX_RETRIES: int = int(os.getenv("MINERU_UPLOAD_MAX_RETRIES", "5"))
MINERU_UPLOAD_TIMEOUT_SECONDS: float = float(os.getenv("MINERU_UPLOAD_TIMEOUT_SECONDS", "900"))

# ── DeepSeek (translation) ──────────────────────────────────────────────────
DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# ── Provider credentials ────────────────────────────────────────────────────
# Only the active provider's key is required. Validating every key would block
# a DeepSeek-only user (the common case now) on a missing Gemini key they will
# never use; validating none would defer a missing key to a confusing SDK error
# mid-run. Providers not listed here are rejected by `src.llm.factory.get_llm`.
_PROVIDER_KEYS: dict[str, str | None] = {
    "gemini": GEMINI_API_KEY,
    "deepseek": DEEPSEEK_API_KEY,
}

_active_key_name = f"{LLM_PROVIDER.upper()}_API_KEY"
if LLM_PROVIDER in _PROVIDER_KEYS and not _PROVIDER_KEYS[LLM_PROVIDER]:
    raise ValueError(
        f"{_active_key_name} is missing but LLM_PROVIDER={LLM_PROVIDER!r}. Create a "
        f"`.env` file in the project root (copy from `.env.example`) and set "
        f"{_active_key_name}=<your-key>."
    )

# ── PDF splitting (src.splitter / src.merger / src.chapter_splitter) ───────
# MinerU's Precision Extract API hard-limits uploads to <= 200 pages and
# <= 200 MB per file. Large textbooks are pre-split into overlapping chunks
# before parsing, then the resulting per-chunk Markdown is merged back
# together. See src/parser.py::parse_book for the orchestrator.
PDF_SPLIT_ENABLED: bool = os.getenv("PDF_SPLIT_ENABLED", "true").lower() in ("1", "true", "yes")
SPLIT_MAX_PAGES: int = int(os.getenv("SPLIT_MAX_PAGES", "190"))  # API hard limit 200
SPLIT_MAX_SIZE_MB: int = int(os.getenv("SPLIT_MAX_SIZE_MB", "180"))  # API hard limit 200 MB
SPLIT_OVERLAP_PAGES: int = int(os.getenv("SPLIT_OVERLAP_PAGES", "2"))
