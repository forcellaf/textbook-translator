"""
Central configuration module.

Loads environment variables from `.env`, sets up optional proxy/network
environment variables BEFORE any network-using library is imported, validates
required secrets, and exposes typed path/setting constants used across the
whole pipeline.

IMPORTANT: This module must be imported (directly or transitively) before the
Anthropic SDK or MinerU code runs, since it is responsible for setting
HTTP_PROXY / HTTPS_PROXY / HF_ENDPOINT in os.environ.
"""

import os
from dataclasses import dataclass
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

# ── Settings ─────────────────────────────────────────────────────────────
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
# validates it lazily, at construction time. ANTHROPIC_API_KEY is the same:
# no-API mode (a Claude Code subscription) runs without one.
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

# ── Claude (every LLM call) ─────────────────────────────────────────────────
# Claude is the only LLM provider. MinerU stays as the parser: it is OCR, not
# an LLM provider.
#
# Every call site names a *role*, and each role has its own model, effort and
# max_tokens. The defaults are the decided table in the README's "Agent mode"
# section; any of them can be overridden per role with
# CLAUDE_MODEL_<ROLE> / CLAUDE_EFFORT_<ROLE> / CLAUDE_MAX_TOKENS_<ROLE>.
#
# max_tokens is generous on purpose: adaptive thinking counts toward it, and a
# reply cut off at the cap is treated as a failure, never as an answer
# (`src.llm.providers.anthropic`). Translation is the extreme case -- replies
# measured up to ~36k tokens of LaTeX, plus thinking on top.
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

ROLE_PROFILE = "profile"
ROLE_STRUCTURE = "structure"
ROLE_CLASSIFY = "classify"
ROLE_TRANSLATE = "translate"
ROLES: tuple[str, ...] = (ROLE_PROFILE, ROLE_STRUCTURE, ROLE_CLASSIFY, ROLE_TRANSLATE)

_ROLE_DEFAULTS: dict[str, tuple[str, str, int]] = {
    ROLE_PROFILE: ("claude-sonnet-5-5", "medium", 32_000),
    ROLE_STRUCTURE: ("claude-sonnet-5-5", "medium", 32_000),
    ROLE_CLASSIFY: ("claude-haiku-5-5", "low", 4_000),
    ROLE_TRANSLATE: ("claude-haiku-5-5", "medium", 96_000),
}
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class RoleConfig:
    """Which model answers one role, at what effort, with what output cap."""

    role: str
    model: str
    effort: str
    max_tokens: int


def role_config(role: str) -> RoleConfig:
    """The model, effort and max_tokens for ``role``, environment first.

    Read at call time rather than import time, so a test (or the Agent SDK
    launcher) can set ``CLAUDE_MODEL_<ROLE>`` after this module was imported.
    """
    if role not in _ROLE_DEFAULTS:
        raise ValueError(f"Unknown LLM role {role!r}; expected one of {', '.join(ROLES)}")
    model, effort, max_tokens = _ROLE_DEFAULTS[role]
    key = role.upper()
    effort = os.getenv(f"CLAUDE_EFFORT_{key}", effort).strip().lower()
    if effort not in EFFORTS:
        raise ValueError(f"CLAUDE_EFFORT_{key}={effort!r} is not one of {', '.join(EFFORTS)}")
    return RoleConfig(
        role=role,
        model=os.getenv(f"CLAUDE_MODEL_{key}", model).strip(),
        effort=effort,
        max_tokens=int(os.getenv(f"CLAUDE_MAX_TOKENS_{key}", str(max_tokens))),
    )


def llm_backend() -> str:
    """``anthropic`` (Python calls the Messages API) or ``exchange`` (Python
    writes request files and an agent answers them -- see `src.llm.exchange`).

    ``LLM_BACKEND`` wins when set. Otherwise the backend follows the key: no
    ``ANTHROPIC_API_KEY`` means a Claude Code subscription with no API access,
    which is exactly what the exchange is for. Read at call time for the same
    reason as `role_config`.
    """
    explicit = os.getenv("LLM_BACKEND", "").strip().lower()
    if explicit:
        if explicit not in ("anthropic", "exchange"):
            raise ValueError(f"LLM_BACKEND={explicit!r} must be 'anthropic' or 'exchange'")
        return explicit
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "exchange"


# Server-side refusal fallback on the Sonnet/Opus roles: if a safety
# classifier declines a request, the API re-runs it on a fallback model inside
# the same call. `default` lets the API pick by refusal category; `off`
# disables it. Haiku has no server-side fallback, and the Batch API rejects
# the parameter, so neither ever sends it.
CLAUDE_REFUSAL_FALLBACK: str = os.getenv("CLAUDE_REFUSAL_FALLBACK", "default").strip().lower()
# The SDK's own retry of 429/5xx/connection errors. Callers that retry on
# top (the translator, the figure classifier) multiply this, so keep it low.
ANTHROPIC_MAX_RETRIES: int = int(os.getenv("ANTHROPIC_MAX_RETRIES", "2"))

# Optional pause between vision calls, for an account with a low rate limit.
# 429s are retried regardless; this only spaces the calls out up front.
VISION_MIN_INTERVAL_SECONDS: float = float(os.getenv("VISION_MIN_INTERVAL_SECONDS", "0"))
# An uncaptioned image is dropped only when the classifier calls it a
# decoration with at least this confidence (and the numbering-gap guard
# passes). Dropping a real figure is the expensive error, so this is high.
FIGURE_DROP_CONFIDENCE: float = float(os.getenv("FIGURE_DROP_CONFIDENCE", "0.9"))

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
