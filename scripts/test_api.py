"""
Standalone smoke test against the REAL LLM API.

Everything under ``tests/`` uses fakes and never touches the network, which
is right for a test suite and useless for answering "is my key working?".
This script is the other half: it makes real calls, cheapest first, and stops
at the first failure so the output points at one specific broken layer
instead of a wall of downstream errors.

    python scripts/test_api.py            # all five stages
    python scripts/test_api.py --quick    # stages 1-2 only (credentials + one call)

Note on stage 3: a fallback to the generic profile counts as a FAILURE here,
which is deliberately the opposite of production behaviour. In production the
fallback protects a multi-hour run from a profiling hiccup; in a smoke test,
that silent degradation is exactly the thing you are trying to detect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/test_api.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Glossary terms and sample text are CJK; a Windows console defaulting to
# cp1252 would otherwise abort this script with a UnicodeEncodeError while
# printing a *successful* result.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
    pass

SAMPLE_MARKDOWN = """\
# 第一章 极限与连续

本章介绍数列极限的定义与基本性质。设函数 $f(x)$ 在点 $x_0$ 的某个邻域内有定义。

若对任意给定的正数 $\\varepsilon$，总存在正数 $\\delta$，使得当 $0 < |x - x_0| < \\delta$
时，恒有 $|f(x) - A| < \\varepsilon$，则称 $A$ 为函数当 $x \\to x_0$ 时的极限。

$$\\lim_{x \\to x_0} f(x) = A$$

常用极限如下表所示：

| 表达式 | 极限值 |
| --- | --- |
| $\\lim_{x \\to 0} \\frac{\\sin x}{x}$ | $1$ |
| $\\lim_{n \\to \\infty} (1 + 1/n)^n$ | $e$ |

上述结论在后续章节中反复使用，读者应熟练掌握。
"""


class StageFailure(Exception):
    """A stage failed. Carries a remediation hint for the user."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def _has_cjk(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


# ── Stages ──────────────────────────────────────────────────────────────────


def stage_1_config() -> object:
    """Config loads, credentials are present, a provider can be constructed."""
    try:
        from src import config
        from src.llm.factory import get_llm
    except ValueError as exc:  # config.py raises this for a missing key
        raise StageFailure(
            f"Configuration failed to load: {exc}",
            "Copy .env.example to .env in the project root and set GEMINI_API_KEY=<your-key>.",
        ) from exc
    except ImportError as exc:
        raise StageFailure(
            f"Could not import the project: {exc}",
            "Install dependencies with `uv sync` and run from the project root.",
        ) from exc

    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        raise StageFailure(
            f"Could not construct the LLM provider: {exc}",
            f"Check LLM_PROVIDER (currently {config.LLM_PROVIDER!r}) in your .env.",
        ) from exc

    print(f"    provider={config.LLM_PROVIDER} model={config.GEMINI_MODEL}")
    print(f"    source_lang={config.SOURCE_LANG} target_lang={config.TARGET_LANG}")
    return llm


def stage_2_raw_call(llm: object) -> None:
    """One raw generate() call -- the cheapest possible proof the key works."""
    try:
        reply = llm.generate(  # type: ignore[attr-defined]
            "You are a translation assistant. Reply with the requested text and nothing else.",
            "Translate to English, output only the translation: 你好，世界。",
            0.0,
        )
    except RuntimeError as exc:
        raise StageFailure(
            f"The API call failed: {exc}",
            "Check that GEMINI_API_KEY is valid and not expired, that GEMINI_MODEL names a "
            "model your key can access, and that the network (or HTTPS_PROXY) can reach the API.",
        ) from exc

    if not reply.strip():
        raise StageFailure(
            "The API returned an empty response.",
            "Try a different GEMINI_MODEL; the current one may be refusing or rate-limiting.",
        )
    print(f"    response: {reply.strip()[:80]}")


def stage_3_profile(llm: object, work_dir: Path) -> object:
    """profile_book on a small sample; a generic fallback counts as failure."""
    from src.profiler import BookProfile, profile_book

    sample_path = work_dir / "merged.md"
    sample_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

    profile = profile_book(sample_path, work_dir, llm=llm, force=True)  # type: ignore[arg-type]

    if profile == BookProfile.generic():
        raise StageFailure(
            "profile_book fell back to the generic profile.",
            "This means the profiling call failed or returned unparseable JSON. Re-run with "
            "logging at WARNING level to see the reason; in production this failure is "
            "silent and only costs terminology consistency.",
        )
    if not profile.glossary:
        raise StageFailure(
            "The profile parsed but contains no glossary entries.",
            "The glossary is what keeps independently-translated chunks consistent. The "
            "model may have ignored the JSON schema -- check the profiling prompt.",
        )

    print(f"    subject={profile.subject!r} level={profile.education_level!r}")
    print(f"    glossary: {len(profile.glossary)} term(s), e.g. {profile.glossary[0]}")
    return profile


def stage_4_markdown(llm: object, profile: object) -> None:
    """translate_markdown in Markdown mode, plus structural spot-checks."""
    from src.translator import TranslationError, translate_markdown

    try:
        result = translate_markdown(
            SAMPLE_MARKDOWN,
            "English",
            llm=llm,  # type: ignore[arg-type]
            profile=profile,  # type: ignore[arg-type]
            output_format="markdown",
        )
    except TranslationError as exc:
        raise StageFailure(
            f"Markdown translation failed: {exc}",
            "The API accepted stage 2 but failed here; this usually means rate limiting on "
            "longer inputs. Lower MAX_CHUNK_TOKENS or retry in a few minutes.",
        ) from exc

    problems = []
    if "#" not in result:
        problems.append("no Markdown heading survived")
    if "\\lim" not in result:
        problems.append("the display formula (\\lim) was not preserved")
    if "|" not in result:
        problems.append("the Markdown table was not preserved")
    if _has_cjk(result):
        problems.append("untranslated Chinese text remains in the output")

    if problems:
        raise StageFailure(
            "Markdown output failed structural checks: " + "; ".join(problems),
            "The model is not following the Markdown rules in "
            "src/translator.py::_MARKDOWN_RULES. Try a stronger GEMINI_MODEL.",
        )
    print(f"    {len(result)} chars, heading + math + table preserved, no CJK left")


def stage_5_latex(llm: object, profile: object) -> None:
    """translate_markdown in LaTeX mode, plus validation and leakage checks."""
    from src.latex import validate_fragment
    from src.translator import TranslationError, translate_markdown

    try:
        result = translate_markdown(
            SAMPLE_MARKDOWN,
            "English",
            llm=llm,  # type: ignore[arg-type]
            profile=profile,  # type: ignore[arg-type]
            output_format="latex",
        )
    except TranslationError as exc:
        raise StageFailure(
            f"LaTeX translation failed: {exc}",
            "See the stage 4 hint; the LaTeX prompt is longer, so rate limits bite sooner.",
        ) from exc

    issues = validate_fragment(result)
    if issues:
        raise StageFailure(
            "The LaTeX fragment is structurally invalid: " + "; ".join(issues),
            "The heal loop in translate_chunk should have caught this -- if it did not, "
            "raise MAX_HEAL_ATTEMPTS or check src/translator.py::_LATEX_RULES.",
        )

    leaked = [
        command
        for command in (r"\documentclass", r"\usepackage", r"\begin{document}", "```")
        if command in result
    ]
    if leaked:
        raise StageFailure(
            f"Preamble/fence leakage in the fragment: {', '.join(leaked)}",
            "src/latex.py owns the preamble; the model must emit body content only. "
            "Check the hard constraints in src/translator.py::_LATEX_RULES.",
        )

    print(f"    {len(result)} chars, validate_fragment clean, no preamble leakage")


# ── Runner ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/test_api.py",
        description="Smoke-test the real LLM API, one layer at a time.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only stages 1-2 (config + one raw API call). No book-sized calls.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "work" / "_smoke_test",
        help="Scratch directory for the profiling stage.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Re-raise unexpected errors instead of printing a one-line summary.",
    )
    args = parser.parse_args()

    total = 2 if args.quick else 5
    print(f"Running {total} stage(s) against the real API.\n")

    try:
        print(f"[1/{total}] Config and credentials...")
        llm = stage_1_config()
        print("  OK\n")

        print(f"[2/{total}] Raw LLM call...")
        stage_2_raw_call(llm)
        print("  OK\n")

        if args.quick:
            print("Quick mode: stages 3-5 skipped. Credentials and the provider work.")
            return 0

        args.work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[3/{total}] Book profiling...")
        profile = stage_3_profile(llm, args.work_dir)
        print("  OK\n")

        print(f"[4/{total}] Markdown translation...")
        stage_4_markdown(llm, profile)
        print("  OK\n")

        print(f"[5/{total}] LaTeX translation...")
        stage_5_latex(llm, profile)
        print("  OK\n")

    except StageFailure as failure:
        print(f"  FAILED: {failure}")
        print(f"  Hint:   {failure.hint}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - a hint beats a traceback here
        if args.traceback:
            raise
        print(f"  FAILED (unexpected): {type(exc).__name__}: {exc}")
        print("  Hint:   This looks like a bug rather than a configuration problem.")
        print("          Re-run with --traceback for the full stack trace.")
        return 1

    print("All stages passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
