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
ASSETS_DIR = PROJECT_ROOT / "assets"

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

# ── MinerU cloud API (PDF -> Markdown) ──────────────────────────────────────
# The only parsing backend. The local `magic_pdf` library was removed: it is
# the source of the historical parse defects (merged table cells destroyed,
# space-mangled LaTeX, welded `$$$$` delimiters) that the cloud VLM backend
# does not produce.
#
# MINERU_TOKEN is not validated here because not every invocation needs it
# (e.g. running only lint or the build). `src.mineru_api.MinerUClient`
# validates it lazily, at construction time -- the same provider-conditional
# style used for the LLM keys below.
MINERU_TOKEN: str | None = os.getenv("MINERU_TOKEN") or os.getenv("MINERU_API_KEY")
MINERU_API_BASE: str = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4").rstrip("/")
MINERU_POLL_INTERVAL_SECONDS: int = int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "10"))
MINERU_TIMEOUT_MINUTES: int = int(os.getenv("MINERU_TIMEOUT_MINUTES", "120"))
# "vlm" or "pipeline". MinerU's docs describe `pipeline` as "no hallucinations"
# and `vlm` as higher accuracy; measured on this project's physics textbook,
# `vlm` is dramatically cleaner (`$10^{-20}$` where pipeline gives
# `$1 0 ^ { - 2 0 }$`). It remains a real per-book judgement call for
# formula-dense material, so keep both selectable.
MINERU_MODEL_VERSION: str = os.getenv("MINERU_MODEL_VERSION", "vlm")
MINERU_LANGUAGE: str = os.getenv("MINERU_LANGUAGE", "ch")
# Upload to Aliyun OSS can be slow/unstable on some networks; these control
# how patiently/persistently we retry the raw PUT upload.
MINERU_UPLOAD_MAX_RETRIES: int = int(os.getenv("MINERU_UPLOAD_MAX_RETRIES", "5"))
MINERU_UPLOAD_TIMEOUT_SECONDS: float = float(os.getenv("MINERU_UPLOAD_TIMEOUT_SECONDS", "900"))
# Free-tier quota, used only to log how much of the day's budget a book eats.
MINERU_DAILY_PAGE_QUOTA: int = int(os.getenv("MINERU_DAILY_PAGE_QUOTA", "1000"))

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
# <= 200 MB per file, so large textbooks are pre-split before parsing.
#
# `src.mineru_api` splits with ZERO overlap: MinerU parses each part
# independently and its output is concatenated verbatim, so an overlap would
# duplicate whole pages rather than help. `SPLIT_OVERLAP_PAGES` still applies
# to `src.merger.merge_chunks`, which is overlap-aware and used by callers
# that want the deduplicating merge.
PDF_SPLIT_ENABLED: bool = os.getenv("PDF_SPLIT_ENABLED", "true").lower() in ("1", "true", "yes")
SPLIT_MAX_PAGES: int = int(os.getenv("SPLIT_MAX_PAGES", "190"))  # API hard limit 200
SPLIT_MAX_SIZE_MB: int = int(os.getenv("SPLIT_MAX_SIZE_MB", "180"))  # API hard limit 200 MB
SPLIT_OVERLAP_PAGES: int = int(os.getenv("SPLIT_OVERLAP_PAGES", "2"))

# ── Translation kit (src.kit) ───────────────────────────────────────────────
# The binding constraint on chunk size is the translating model's OUTPUT
# limit, not its context window: it has to emit a full translation of every
# chunk it is given.
#
# Measured on the completed book, translated output ran 1.33x to 2.87x the
# input length (mean 2.29x), and the largest single reply was 143,000
# characters (~36,000 tokens) -- occasionally over the model's output limit,
# coming back truncated for the user to spot and repair by hand. LaTeX output
# is more verbose than the Markdown those numbers were measured on (a figure
# goes from `![](IMG_0042)` to a `\bookfig{...}{...}` call with a translated
# caption, across ~800 figures), so the worst case gets worse, not better.
#
# At 30,000 characters the worst case lands near 87,000 characters ~ 22,000
# tokens, comfortably inside the limit.
KIT_CHUNK_CHARS: int = int(os.getenv("KIT_CHUNK_CHARS", "30000"))
