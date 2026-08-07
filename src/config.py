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

# ── Secrets / validation ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Create a `.env` file in the project root "
        "(copy from `.env.example`) and set GEMINI_API_KEY=<your-key>."
    )

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# ── Settings ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
SOURCE_LANG: str = os.getenv("SOURCE_LANG", "Chinese")
TARGET_LANG: str = os.getenv("TARGET_LANG", "English")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "3000"))
API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "5"))
MAX_HEAL_ATTEMPTS: int = int(os.getenv("MAX_HEAL_ATTEMPTS", "3"))
