"""
LaTeX assets, fragment scanning and document assembly.

Division of labour with the translating model
---------------------------------------------
The model writes **body fragments only** -- chapter/section content, math,
tables, figures. ``assets/preamble.tex`` owns everything that must be
identical book-wide: documentclass, packages, caption styling, the
``example``/``exercise`` environments and the ``\\bookfig`` macros. That split
is deliberate. A book is translated in dozens of independent conversations;
anything global that each one re-invents (a package list, a float setup) comes
back subtly different every time and the document stops compiling. So the
model never emits a preamble, and preamble commands inside a fragment are
treated as a defect rather than something to merge.

The two assets are hand-maintained and are the specification
------------------------------------------------------------
``assets/preamble.tex`` and ``assets/system_prompt_latex.txt`` are edited
directly by the person running the pipeline -- the prompt in particular is
tuned against whatever the model got wrong last time. Neither is generated
from code, and neither should be: this module loads them, substitutes the
book's glossary into the prompt, and otherwise leaves them alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import ASSETS_DIR
from src.profiler import BookProfile, profile_to_prompt_block

logger = logging.getLogger(__name__)

PREAMBLE_PATH: Path = ASSETS_DIR / "preamble.tex"
SYSTEM_PROMPT_PATH: Path = ASSETS_DIR / "system_prompt_latex.txt"

# Where `profile_to_prompt_block` output goes in the prompt template.
GLOSSARY_PLACEHOLDER = "{{BOOK_CONTEXT_AND_GLOSSARY}}"

# Every environment the preamble actually defines, plus the LaTeX built-ins it
# loads packages for. Anything else is "! LaTeX Error: Environment X undefined"
# at compile time -- during testing the model invented `solution` and
# `theorem`. Starred variants (equation*, align*) are accepted: the check
# below strips the star before comparing.
DEFINED_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "example",
        "exercise",
        "figure",
        "table",
        "tabular",
        "itemize",
        "enumerate",
        "equation",
        "align",
        "center",
        "minipage",
        "document",
        "longtable",
    }
)

PREAMBLE_ONLY_COMMANDS: tuple[str, ...] = (
    r"\documentclass",
    r"\usepackage",
    r"\begin{document}",
    r"\end{document}",
)

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_ENV_ARG_RE = re.compile(r"\s*\{([^{}]*)\}")
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")


# ── Assets ──────────────────────────────────────────────────────────────────


def load_preamble(path: Path | None = None) -> str:
    """Read ``assets/preamble.tex``.

    Raises:
        FileNotFoundError: If the asset is missing. There is no generated
            fallback on purpose -- a book built against an improvised preamble
            would silently lose the figure macros and the chapter offset.
    """
    path = path or PREAMBLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"LaTeX preamble not found: {path}. It is a tracked asset; restore it "
            "rather than generating one."
        )
    return path.read_text(encoding="utf-8")


def build_system_prompt(profile: BookProfile | None = None, *, path: Path | None = None) -> str:
    """Load the AI Studio system prompt and fold the book's profile into it.

    The template lives in ``assets/system_prompt_latex.txt`` rather than in a
    string literal here because it is edited by hand between books.
    ``{{BOOK_CONTEXT_AND_GLOSSARY}}`` is replaced with the profiler's block,
    which is what keeps terminology consistent across independently
    translated chunks.
    """
    path = path or SYSTEM_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"LaTeX system prompt not found: {path}. It is a tracked asset; restore "
            "it rather than generating one."
        )

    template = path.read_text(encoding="utf-8")
    block = profile_to_prompt_block(profile) if profile is not None else ""

    if GLOSSARY_PLACEHOLDER not in template:
        logger.warning(
            "%s has no %s placeholder; the book context and glossary were not "
            "inserted.",
            path,
            GLOSSARY_PLACEHOLDER,
        )
        return template

    return template.replace(GLOSSARY_PLACEHOLDER, block)


# ── Fragment scanning ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnvIssue:
    """One environment defect, with the 1-based line it was found on."""

    kind: str  # unclosed | crossed | stray-end | undefined | missing-name
    env: str
    line: int
    message: str


def scan_environments(text: str) -> list[EnvIssue]:
    """Every ``\\begin``/``\\end`` defect in ``text``.

    One pass, tracking line numbers as it goes. Escaped literals (``\\{``,
    ``\\$``) and ``%`` comments are skipped so they cannot fake an
    environment. Crossed pairs (``\\begin{itemize}...\\end{enumerate}``) are
    reported as such rather than as two separate imbalances, because that is
    how they read in the source.
    """
    issues: list[EnvIssue] = []
    stack: list[tuple[str, int]] = []

    i = 0
    line = 1
    length = len(text)

    while i < length:
        char = text[i]

        if char == "\n":
            line += 1
            i += 1
            continue

        if char == "%":
            newline = text.find("\n", i)
            if newline == -1:
                break
            i = newline
            continue

        if char == "\\":
            match = _COMMAND_RE.match(text, i)
            if not match:
                # An escaped literal: \{ \} \$ \& \% \# \_ \\ ...
                if i + 1 < length and text[i + 1] == "\n":
                    line += 1
                i += 2
                continue

            name = match.group(1)
            if name not in ("begin", "end"):
                i = match.end()
                continue

            arg = _ENV_ARG_RE.match(text, match.end())
            if arg is None:
                issues.append(
                    EnvIssue(
                        "missing-name", "", line, f"\\{name} is missing its environment name"
                    )
                )
                i = match.end()
                continue

            env = arg.group(1).strip()
            at_line = line
            line += text.count("\n", i, arg.end())
            i = arg.end()

            if name == "begin":
                stack.append((env, at_line))
                if env.rstrip("*") not in DEFINED_ENVIRONMENTS:
                    issues.append(
                        EnvIssue(
                            "undefined",
                            env,
                            at_line,
                            f"\\begin{{{env}}} is not one of the defined environments "
                            "-- '! LaTeX Error: Environment "
                            f"{env} undefined.'",
                        )
                    )
            elif not stack:
                issues.append(
                    EnvIssue(
                        "stray-end", env, at_line, f"\\end{{{env}}} has no matching \\begin"
                    )
                )
            elif stack[-1][0] != env:
                open_env, open_line = stack.pop()
                issues.append(
                    EnvIssue(
                        "crossed",
                        env,
                        at_line,
                        f"\\begin{{{open_env}}} (line {open_line}) is closed by "
                        f"\\end{{{env}}}",
                    )
                )
            else:
                stack.pop()
            continue

        i += 1

    for env, at_line in reversed(stack):
        issues.append(
            EnvIssue("unclosed", env, at_line, f"\\begin{{{env}}} is never closed")
        )

    return sorted(issues, key=lambda issue: issue.line)


def brace_imbalance(text: str) -> int:
    """Net brace depth, ignoring escaped ``\\{`` and ``\\}``.

    Returns the closing surplus as a negative number when the text dips below
    depth 0 at any point, so ``}{`` is reported even though it nets to zero.
    """
    cleaned = re.sub(r"\\[{}]", "", text)
    depth = 0
    low = 0
    for char in cleaned:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            low = min(low, depth)
    return depth if depth else low


def find_preamble_leakage(text: str) -> list[tuple[int, str]]:
    """``(line, command)`` for every preamble command found in a body fragment.

    The preamble is added once, by ``assemble_document``. A second
    ``\\documentclass`` or a stray ``\\end{document}`` halfway through the book
    ends the document there and silently drops everything after it.
    """
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        for command in PREAMBLE_ONLY_COMMANDS:
            if command in line:
                hits.append((lineno, command))
    return hits


def validate_fragment(tex: str) -> list[str]:
    """Check one model-produced body fragment. ``[]`` means usable.

    The fast structural subset of ``src.lint``: this runs inside the
    translator's heal loop, where the only question is whether to re-ask for
    this chunk. ``src.lint`` is what produces the reviewable report.
    """
    issues = [
        f"{command} must not appear in a body fragment "
        "(the preamble is added once, at assembly)"
        for command in dict.fromkeys(command for _, command in find_preamble_leakage(tex))
    ]
    issues += [issue.message for issue in scan_environments(tex)]

    imbalance = brace_imbalance(tex)
    if imbalance > 0:
        issues.append(f"unbalanced braces: {imbalance} unclosed '{{'")
    elif imbalance < 0:
        issues.append("unbalanced braces: a closing '}' has no matching '{'")

    if len(_UNESCAPED_DOLLAR_RE.findall(tex.replace("$$", ""))) % 2:
        issues.append("unbalanced inline math: odd number of unescaped '$'")
    if tex.count("$$") % 2:
        issues.append("unbalanced display math: odd number of '$$'")

    return issues


# ── Assembly ────────────────────────────────────────────────────────────────


def _strip_preamble_leakage(body: str) -> str:
    """Drop preamble lines a model emitted despite being told not to.

    Cheaper than failing assembly: the fragment's real content is fine, and
    the asset preamble is the one that must win. ``src.lint`` reports the same
    lines, so nothing is hidden by doing this.
    """
    lines = [
        line
        for line in body.splitlines()
        if not any(line.lstrip().startswith(cmd) for cmd in PREAMBLE_ONLY_COMMANDS)
    ]
    return "\n".join(lines).strip("\n")


def assemble_document(
    body_parts: list[str],
    output_path: Path,
    *,
    preamble_path: Path | None = None,
) -> Path:
    """Wrap ``body_parts`` in the asset preamble and write a compilable .tex.

    ``assets/preamble.tex`` already ends with ``\\begin{document}``,
    ``\\frontmatter``, ``\\tableofcontents``, ``\\mainmatter`` and the chapter
    counter, so only ``\\end{document}`` is added after the body.

    The file must be written next to ``images/``: the preamble's
    ``\\graphicspath`` looks for figures there, relative to the .tex.
    """
    bodies = [
        cleaned for cleaned in (_strip_preamble_leakage(part) for part in body_parts) if cleaned
    ]

    document = "\n\n".join(
        [load_preamble(preamble_path).rstrip("\n"), "\n\n".join(bodies), "\\end{document}", ""]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    logger.info("Wrote LaTeX document (%d body part(s)): %s", len(bodies), output_path)
    return output_path


__all__ = [
    "DEFINED_ENVIRONMENTS",
    "GLOSSARY_PLACEHOLDER",
    "PREAMBLE_ONLY_COMMANDS",
    "PREAMBLE_PATH",
    "SYSTEM_PROMPT_PATH",
    "EnvIssue",
    "assemble_document",
    "brace_imbalance",
    "build_system_prompt",
    "find_preamble_leakage",
    "load_preamble",
    "scan_environments",
    "validate_fragment",
]
