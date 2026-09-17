"""
Every defect in one pass, on both sides of the translation.

Why this module exists
----------------------
LaTeX reports one error per compile. Finding four defects in an
840,000-character book therefore cost four full build cycles, each one
starting from a line number in a file the size of a phone book. This runs
every check at once, with line numbers and context, so the whole list is on
screen before the first build.

Two surfaces, two entry points
------------------------------
``lint_markdown`` checks the **source**: MinerU's output, which the model is
about to read. ``lint_latex`` checks the **translation**: the LaTeX the model
sent back, which has to compile. ``lint_kit`` runs the chunk-pair checks and
then hands the assembled translation to ``lint_latex``.

Every LaTeX check here exists because it was a real failure during testing,
and none of them is visible without an explicit check: the output looks
correct and only fails at compile time.

Historical source checks are kept on purpose
--------------------------------------------
Several markdown checks (delimiter whitespace, welded ``$$$$``, escaped
``\\$``, over-tabbed arrays) find **zero** defects in the current cloud VLM
output. They stay because they are nearly free and a future parser change
could reintroduce any of them silently. They are diagnostics; there are
deliberately no normalizer passes behind them.

Report, never guess
-------------------
The unknown-command check flagged 41 tokens on the test book. Forty were real
LaTeX simply missing from ``KNOWN_COMMANDS`` below; exactly one was the
genuine defect. A checker with that hit rate must report, never auto-fix.
Only two repairs are unambiguous enough to apply: a known command literally
doubled (``\\mathrmmathrm`` -> ``\\mathrm``), and a source number the model
repeated into a heading or caption that LaTeX numbers itself.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.config import SOURCE_RESIDUE_THRESHOLD
from src.latex import (
    DEFINED_ENVIRONMENTS,
    brace_imbalance,
    find_preamble_leakage,
    scan_environments,
)
from src.normalize import (
    HEADING_RE,
    MathSpan,
    display_parity,
    dollar_run_histogram,
    iter_math_spans,
)

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """How a finding should be acted on."""

    AUTO_FIX = "auto-fix"  # provably safe; `apply_auto_fixes` handles it
    REVIEW = "review"  # a human must look; gates the build
    INFO = "info"  # context, not a problem


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    message: str
    line: int = 0  # 1-based markdown line, 0 when not line-specific
    context: str = ""

    def render(self) -> str:
        where = f"L{self.line}" if self.line else "-"
        tail = f"\n        {self.context}" if self.context else ""
        return f"  [{self.severity.value:>8}] {where:>8}  {self.check}: {self.message}{tail}"


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)

    def of(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def needs_review(self) -> bool:
        return any(f.severity is Severity.REVIEW for f in self.findings)

    @property
    def exit_code(self) -> int:
        """Non-zero when anything needs review, so this can gate the build."""
        return 1 if self.needs_review else 0

    def render(self) -> str:
        lines: list[str] = []
        for severity in (Severity.REVIEW, Severity.AUTO_FIX, Severity.INFO):
            group = self.of(severity)
            if not group:
                continue
            lines.append(f"\n{severity.value.upper()} ({len(group)}):")
            lines.extend(f.render() for f in group)

        review = len(self.of(Severity.REVIEW))
        auto = len(self.of(Severity.AUTO_FIX))
        lines.append(
            f"\n{review} finding(s) need review, {auto} auto-fixable, "
            f"{len(self.of(Severity.INFO))} informational."
        )
        if not review:
            lines.append("Nothing blocking; the build can go ahead.")
        return "\n".join(lines)


# ── Command vocabulary ──────────────────────────────────────────────────────
# Plain LaTeX plus amsmath/amssymb, siunitx-adjacent bits, and the physics
# notation this corpus actually uses. Deliberately NOT exhaustive: the check
# it backs only ever reports, so a missing entry costs one line of noise,
# never a wrong edit.
KNOWN_COMMANDS: frozenset[str] = frozenset(
    """
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi
psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac dfrac tfrac cfrac sqrt sum int iint iiint oint prod coprod lim liminf
limsup inf sup max min log ln lg exp sin cos tan cot sec csc arcsin arccos
arctan sinh cosh tanh coth det dim ker deg gcd hom Pr arg bmod pmod
mathrm mathbf mathit mathcal mathbb mathfrak mathsf mathtt mathscr boldsymbol
pmb text textbf textit textrm textsf texttt mbox hbox operatorname
times div pm mp cdot cdots ldots vdots ddots dots dotsb dotsc ast
leq geq neq ne le ge approx approxeq equiv sim simeq cong propto ll gg
leqslant geqslant lesssim gtrsim doteq
rightarrow leftarrow Rightarrow Leftarrow leftrightarrow Leftrightarrow to
gets mapsto uparrow downarrow updownarrow longrightarrow longleftarrow
longleftrightarrow rightleftharpoons leftrightharpoons rightharpoonup
leftharpoonup nearrow searrow swarrow nwarrow implies iff
partial nabla infty forall exists nexists in ni notin subset supset subseteq
supseteq subsetneq supsetneq cup cap bigcup bigcap setminus emptyset
varnothing angle measuredangle perp parallel triangle square
hat bar vec dot ddot dddot tilde check breve acute grave widehat widetilde
overline underline overrightarrow overleftarrow underbrace overbrace stackrel
overset underset
left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr
langle rangle lfloor rfloor lceil rceil vert Vert lvert rvert lVert rVert
backslash
quad qquad hspace vspace phantom hphantom vphantom smash raisebox
displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits mathop
mathrel mathbin mathopen mathclose mathpunct
begin end label ref eqref tag notag nonumber caption footnote
prime circ star bullet oplus ominus otimes odot wedge vee neg lnot land lor
cdotp ldotp colon
binom dbinom tbinom choose over atop substack
color textcolor mathstrut strut
Re Im aleph hbar hslash ell wp imath jmath
degree celsius micro ohm angstrom
mathring not sb sp
hline cline vline multicolumn multirow toprule midrule bottomrule
mid nmid parallel shortmid AA SS aa ss textcircled textdegree textmu
longmapsto hookrightarrow hookleftarrow rightsquigarrow between
""".split()
)

# Commands that take a braced argument. One of these sitting at the very end
# of a math span has lost its argument -- "! Missing } inserted."
_ARGUMENT_COMMANDS: frozenset[str] = frozenset(
    """
frac dfrac tfrac cfrac sqrt text textbf textit textrm mathrm mathbf mathit
mathcal mathbb mathfrak mathsf mathtt mathscr boldsymbol pmb operatorname
hat bar vec dot ddot tilde widehat widetilde overline underline
overrightarrow overleftarrow underbrace overbrace binom stackrel overset
underset color textcolor mbox hbox
""".split()
)

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_DOUBLED_COMMAND_RE = re.compile(r"\\([A-Za-z]+?)\1(?![A-Za-z])")
_ARRAY_RE = re.compile(r"\\begin\{(array|tabular)\}\s*\{([^}]*)\}(.*?)\\end\{\1\}", re.DOTALL)
_TAG_RE = re.compile(r"\\tag\s*\*?\s*\{([^}]*)\}")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
_HTML_IMAGE_RE = re.compile(r'<img\b[^>]*\bsrc="([^"]+)"', re.IGNORECASE)
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

IMG_TOKEN_RE = re.compile(r"IMG_(\d{4})")
_MATH_STRIP_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)

# Source chunks are Markdown (that is what MinerU produces); the replies
# saved against them are LaTeX, and are named for the chunk they answer:
# chunks/007.md -> translated/007.tex.
TRANSLATED_SUFFIX = ".tex"

# ── LaTeX-side patterns ─────────────────────────────────────────────────────

# Math delimiters the dollar-based scanner in src.normalize does not know
# about. The translated book uses \[...\] for unnumbered display math and
# equation/align for numbered, so the math checks would otherwise see only a
# fraction of the mathematics in it.
_BRACKET_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_MATH_ENV_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray|displaymath)\*?\}"
    r"(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)

# Sectioning and caption commands whose argument LaTeX numbers by itself.
_NUMBERED_ARG_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|caption)\*?(?:\[[^\]]*\])?\{"
)

# A source number (or a "Figure"/"Table" label) the model copied into an
# argument LaTeX numbers itself, producing "12.1 12.1 Electric Charge". The
# trailing `\s+(?=\S)` is what keeps this unambiguous: it never fires on a
# title that merely starts with a digit ("3D Printing"), on a bare
# "\caption{Figure 12.3}" with nothing after the label, or on prose beginning
# with an unnumbered "Figure".
_LEADING_NUMBER_RE = re.compile(
    r"\A(?:(?:Figure|Fig\.|Table|Tab\.)\s*)?\d+(?:\.\d+)*\.?\s+(?=\S)",
    re.IGNORECASE,
)

# A LaTeX comment: an unescaped `%` to the end of the line.
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")

# Figure references in translated LaTeX: the \bookfig / \bookfigtwo macros
# from assets/preamble.tex, plus a raw \includegraphics if the model used one.
_TEX_IMAGE_RES: tuple[tuple[re.Pattern[str], tuple[int, ...]], ...] = (
    (re.compile(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^{}]*)\}"), (1,)),
    (re.compile(r"\\bookfig(?:\[[^\]]*\])?\s*\{([^{}]*)\}"), (1,)),
    (re.compile(r"\\bookfigtwo\s*\{([^{}]*)\}\s*\{[^{}]*\}\s*\{([^{}]*)\}"), (1, 2)),
)

# Truncation floor for a single chunk judged alone: a translation under 40%
# of its source's length. Deliberately lenient, because the source is CJK and
# the target is English -- the same passage expands roughly threefold, so a
# reply at 40% of the *source* length is already drastically short. A cut-off
# reply reads perfectly in isolation; only length catches it.
_TRUNCATION_RATIO = 0.4

# Across a whole kit there is a better signal available: the book's own
# median expansion ratio. A chunk that expanded half as much as its
# neighbours is suspect regardless of what the languages are, so this needs
# no per-language constant. Needs a few chunks before a median means
# anything.
_TRUNCATION_OUTLIER_RATIO = 0.5
_MIN_CHUNKS_FOR_MEDIAN = 4

_MAX_EXAMPLES = 12  # per check, so one noisy defect can't bury the rest

# Flat-outline detection. See `_check_headings`.
_FLAT_OUTLINE_RATIO = 0.9
_FLAT_OUTLINE_MIN_HEADINGS = 20


def _excerpt(text: str, limit: int = 120) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace, non-math characters in CJK ranges.

    Math is stripped first: a chunk that is 40% formulas would otherwise
    dilute its own residue score below the threshold.
    """
    stripped = _MATH_STRIP_RE.sub("", text)
    chars = [c for c in stripped if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "\u4e00" <= c <= "\u9fff") / len(chars)


# ── Markdown checks ─────────────────────────────────────────────────────────


def _check_dollar_runs(text: str) -> list[Finding]:
    histogram = dollar_run_histogram(text)
    findings = [
        Finding(
            "dollar-runs",
            Severity.INFO,
            f"`$` run-length histogram: {histogram}",
        )
    ]
    welded = {length: count for length, count in histogram.items() if length > 2}
    if welded:
        findings.append(
            Finding(
                "welded-delimiters",
                Severity.REVIEW,
                f"welded `$` runs longer than `$$`: {welded}. The delimiters pair "
                "up wrongly from there on, swallowing prose into mathematics.",
            )
        )
    return findings


def _check_display_parity(text: str) -> list[Finding]:
    count, even = display_parity(text)
    if even:
        return [Finding("display-parity", Severity.INFO, f"{count} `$$` delimiters, balanced")]
    return [
        Finding(
            "display-parity",
            Severity.REVIEW,
            f"{count} `$$` delimiters -- odd, so a display block is never closed "
            "and everything after it reads as mathematics.",
        )
    ]


def _check_inline_parity(text: str) -> list[Finding]:
    """Per-paragraph inline `$` parity, ignoring escaped `\\$`.

    Done per paragraph rather than per line: a display block spans lines, and
    per-document counting would net two unrelated errors back to zero.
    """
    findings: list[Finding] = []
    offset = 1
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        if paragraph.count("$$") % 2 == 0:
            without_display = paragraph.replace("$$", "")
            if len(_UNESCAPED_DOLLAR_RE.findall(without_display)) % 2:
                findings.append(
                    Finding(
                        "inline-parity",
                        Severity.REVIEW,
                        "paragraph has an odd number of unescaped `$`",
                        offset,
                        _excerpt(paragraph),
                    )
                )
        offset += paragraph.count("\n") + 2
    return findings[:_MAX_EXAMPLES]


def _check_delimiter_whitespace(text: str) -> list[Finding]:
    hits = [
        span
        for span in iter_math_spans(text)
        if span.kind == "inline" and span.body != span.body.strip()
    ]
    if not hits:
        return [
            Finding("delimiter-whitespace", Severity.INFO, "no inline span is padded (clean)")
        ]
    return [
        Finding(
            "delimiter-whitespace",
            Severity.REVIEW,
            "whitespace inside inline `$` delimiters; this does not read as "
            "math, so the LaTeX inside it leaks into text mode",
            span.line,
            _excerpt(span.body),
        )
        for span in hits[:_MAX_EXAMPLES]
    ]


def _check_escaped_dollars(text: str) -> list[Finding]:
    findings = [
        Finding(
            "escaped-dollar",
            Severity.REVIEW,
            "escaped `\\$`; in a book with no currency this is almost always a "
            "math delimiter that got escaped, breaking its pair",
            lineno,
            _excerpt(line),
        )
        for lineno, line in enumerate(text.split("\n"), start=1)
        if "\\$" in line
    ]
    return findings[:_MAX_EXAMPLES] or [
        Finding("escaped-dollar", Severity.INFO, "no escaped `\\$` (clean)")
    ]


def _check_braces(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for span in iter_math_spans(text):
        imbalance = brace_imbalance(span.body)
        if imbalance:
            sign = f"+{imbalance}" if imbalance > 0 else str(imbalance)
            findings.append(
                Finding(
                    "brace-balance",
                    Severity.REVIEW,
                    f"{span.kind} math span has unbalanced braces ({sign})",
                    span.line,
                    _excerpt(span.body),
                )
            )
    return findings[:_MAX_EXAMPLES]


def _check_multiple_tags(text: str) -> list[Finding]:
    """Math spans carrying more than one ``\\tag``.

    amsmath hard-errors on this -- "! Package amsmath Error: Multiple \\tag."
    -- so the document cannot compile at all. Seen once on a real book, where
    the parser read a stray number in the scan margin as a second tag and
    emitted ``\\tag {30}\\tag{30.12}``.

    Reported, never auto-fixed: the equation is guaranteed broken, but
    deciding *which* tag is the real equation number means looking at the
    book. Both are listed so that takes seconds.
    """
    findings: list[Finding] = []
    for span in iter_math_spans(text):
        tags = _TAG_RE.findall(span.body)
        if len(tags) > 1:
            findings.append(
                Finding(
                    "multiple-tags",
                    Severity.REVIEW,
                    f"{span.kind} math span carries {len(tags)} \\tag commands "
                    f"({', '.join(repr(t) for t in tags)}); amsmath allows one, so "
                    "the build fails outright. Delete whichever is not the real "
                    "equation number.",
                    span.line,
                    _excerpt(span.body),
                )
            )
    return findings[:_MAX_EXAMPLES]


def _check_arrays(text: str) -> list[Finding]:
    """Rows with more ``&`` than the column spec allows.

    ``&`` is counted at brace depth 0 only -- a nested ``\\begin{array}`` or a
    ``{a & b}`` group inside a cell is not this row's business. Produces
    "! Extra alignment tab has been changed to \\cr" at compile time.
    """
    findings: list[Finding] = []
    for match in _ARRAY_RE.finditer(text):
        env, spec, body = match.group(1), match.group(2), match.group(3)
        columns = len(re.findall(r"[lcr]", spec))
        if columns == 0:
            continue
        line = text.count("\n", 0, match.start()) + 1
        for index, row in enumerate(re.split(r"\\\\", body), start=1):
            depth = 0
            tabs = 0
            for char in row:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                elif char == "&" and depth == 0:
                    tabs += 1
            if tabs > columns - 1:
                findings.append(
                    Finding(
                        "array-columns",
                        Severity.REVIEW,
                        f"{env}{{{spec.strip()}}} declares {columns} column(s) but "
                        f"row {index} has {tabs} alignment tab(s)",
                        line,
                        _excerpt(row),
                    )
                )
    return findings[:_MAX_EXAMPLES]


def _check_dangling_commands(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for span in iter_math_spans(text):
        trailing = re.search(r"\\([A-Za-z]+)\s*$", span.body)
        if trailing and trailing.group(1) in _ARGUMENT_COMMANDS:
            findings.append(
                Finding(
                    "dangling-command",
                    Severity.REVIEW,
                    f"\\{trailing.group(1)} ends the {span.kind} math span with no "
                    "argument -- '! Missing } inserted.'",
                    span.line,
                    _excerpt(span.body),
                )
            )
    return findings[:_MAX_EXAMPLES]


def _check_unknown_commands(text: str) -> list[Finding]:
    """Report command tokens inside math that plain LaTeX will not know.

    Report only. On the test book this flagged 41 tokens, 40 of which were
    real LaTeX missing from ``KNOWN_COMMANDS``; auto-"fixing" any of them
    would have damaged 40 correct formulas to repair one.
    """
    seen: dict[str, tuple[int, str]] = {}
    counts: dict[str, int] = {}
    for span in iter_math_spans(text):
        for match in _COMMAND_RE.finditer(span.body):
            name = match.group(1)
            if name in KNOWN_COMMANDS:
                continue
            counts[name] = counts.get(name, 0) + 1
            seen.setdefault(name, (span.line, _excerpt(span.body)))

    if not counts:
        return [Finding("unknown-command", Severity.INFO, "no unrecognized math commands")]

    # Rarest first: a token used once is far likelier to be damage than one
    # used two hundred times.
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    return [
        Finding(
            "unknown-command",
            Severity.REVIEW,
            f"\\{name} appears {count}x in math and is not in KNOWN_COMMANDS -- "
            "check it is real LaTeX, then add it to the list",
            seen[name][0],
            seen[name][1],
        )
        for name, count in ordered[:_MAX_EXAMPLES]
    ]


def _check_doubled_commands(text: str) -> list[Finding]:
    """A known command written twice with one backslash (``\\mathrmmathrm``).

    The only auto-fixable defect here: it cannot be anything but damage, and
    the repair is unambiguous.
    """
    findings: list[Finding] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        for match in _DOUBLED_COMMAND_RE.finditer(line):
            if match.group(1) in KNOWN_COMMANDS:
                findings.append(
                    Finding(
                        "doubled-command",
                        Severity.AUTO_FIX,
                        f"\\{match.group(1)} is doubled ({match.group(0)}) -- "
                        "'! Undefined control sequence.'",
                        lineno,
                        _excerpt(line),
                    )
                )
    return findings


def _check_headings(text: str) -> list[Finding]:
    levels: dict[int, int] = {}
    for line in text.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            levels[level] = levels.get(level, 0) + 1

    if not levels:
        return [Finding("heading-levels", Severity.REVIEW, "the document has no headings at all")]

    findings = [
        Finding(
            "heading-levels",
            Severity.INFO,
            f"heading level counts: {dict(sorted(levels.items()))}",
        )
    ]

    total = sum(levels.values())
    level, count = max(levels.items(), key=lambda kv: kv[1])
    # "Everything at one level" in practice means one level swallowing nearly
    # everything, not literally all of it: the raw parse of the test book had
    # 20 headings at `#` and 422 at `##`, which is two levels and still a flat
    # outline. Below `_FLAT_OUTLINE_MIN_HEADINGS` the ratio means nothing --
    # a six-heading document legitimately has one level.
    if total >= _FLAT_OUTLINE_MIN_HEADINGS and count / total >= _FLAT_OUTLINE_RATIO:
        findings.append(
            Finding(
                "heading-levels",
                Severity.REVIEW,
                f"{count} of {total} headings sit at level {level}; the outline is "
                "flat, so the model has nothing to tell a chapter, a section and a "
                "worked example apart. Run the profiler's heading classification.",
            )
        )
    return findings


def _check_image_paths(text: str, base_dir: Path | None) -> list[Finding]:
    refs = _MD_IMAGE_RE.findall(text) + _HTML_IMAGE_RE.findall(text)
    if not refs:
        return [Finding("image-paths", Severity.INFO, "no image references")]
    if base_dir is None:
        return [
            Finding(
                "image-paths",
                Severity.INFO,
                f"{len(refs)} image reference(s); no base directory given, so none "
                "were resolved on disk",
            )
        ]

    missing: list[str] = []
    for ref in dict.fromkeys(refs):
        if ref.startswith(("http://", "https://", "data:")):
            continue
        if (base_dir / ref).exists() or (base_dir / "images" / Path(ref).name).exists():
            continue
        missing.append(ref)

    if not missing:
        return [
            Finding("image-paths", Severity.INFO, f"all {len(set(refs))} image path(s) resolve")
        ]
    return [
        Finding(
            "image-paths",
            Severity.REVIEW,
            f"{len(missing)} image path(s) do not resolve under {base_dir}",
            context=", ".join(missing[:5]),
        )
    ]


# ── LaTeX checks ────────────────────────────────────────────────────────────
#
# Each of these fired on real translated output during testing. They exist
# because none of the defects is visible by reading the file: it looks like
# perfectly ordinary LaTeX and only fails at compile time, one error per
# compile.


def iter_latex_math_spans(text: str):
    """Every math span in translated LaTeX, dollars and environments alike.

    ``src.normalize.iter_math_spans`` only knows ``$...$`` and ``$$...$$``,
    which is all the *source* markdown contains. The translation also uses
    ``\\[...\\]`` and equation/align environments, and the math checks would
    quietly skip those.
    """
    yield from iter_math_spans(text)
    for pattern, body_group in ((_BRACKET_MATH_RE, 1), (_MATH_ENV_RE, 2)):
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield MathSpan(line, "display", match.group(body_group))


def _check_environments(text: str) -> list[Finding]:
    """Unbalanced, crossed and undefined environments.

    Undefined ones are the expensive kind: during testing the model invented
    ``solution`` and ``theorem``, and an environment the preamble never
    defines stops the build dead.
    """
    findings = [
        Finding(
            "undefined-environment" if issue.kind == "undefined" else "environment-balance",
            Severity.REVIEW,
            issue.message,
            issue.line,
        )
        for issue in scan_environments(text)
    ]
    if findings:
        return findings[:_MAX_EXAMPLES]
    return [
        Finding(
            "environment-balance",
            Severity.INFO,
            f"every \\begin has a matching \\end, from the {len(DEFINED_ENVIRONMENTS)} "
            "defined environments",
        )
    ]


def _check_preamble_leakage(text: str) -> list[Finding]:
    """Preamble commands inside a body fragment.

    The preamble is added once, at assembly. A stray ``\\end{document}``
    halfway through ends the book there and silently drops the rest.
    """
    return [
        Finding(
            "preamble-leakage",
            Severity.REVIEW,
            f"{command} must not appear in translated body text; the preamble is "
            "added once, at assembly",
            line,
        )
        for line, command in find_preamble_leakage(text)[:_MAX_EXAMPLES]
    ]


def _check_latex_braces(text: str) -> list[Finding]:
    """Brace balance, per paragraph so the report carries a line number."""
    findings: list[Finding] = []
    offset = 1
    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        imbalance = brace_imbalance(paragraph)
        if imbalance:
            detail = (
                f"{imbalance} unclosed '{{'"
                if imbalance > 0
                else f"{-imbalance} closing '}}' with no matching '{{'"
            )
            findings.append(
                Finding(
                    "brace-balance",
                    Severity.REVIEW,
                    f"paragraph has unbalanced braces ({detail})",
                    offset,
                    _excerpt(paragraph),
                )
            )
        offset += paragraph.count("\n") + 2
    return findings[:_MAX_EXAMPLES]


def _braced_argument(text: str, open_index: int) -> tuple[str, int] | None:
    """The content of the group starting at ``text[open_index] == '{'``.

    Brace-matched rather than regex-matched, because these arguments
    routinely contain nested groups: ``\\caption{Field of $E_{x}$}``.
    """
    depth = 0
    i = open_index
    length = len(text)
    while i < length:
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
        i += 1
    return None


def _iter_numbered_arguments(text: str):
    """``(command, argument, start, end, line)`` for each sectioning/caption
    argument LaTeX numbers by itself."""
    for match in _NUMBERED_ARG_RE.finditer(text):
        found = _braced_argument(text, match.end() - 1)
        if found is None:
            continue
        argument, end = found
        line = text.count("\n", 0, match.start()) + 1
        yield match.group(1), argument, match.end() - 1, end, line


def _check_duplicated_numbers(text: str) -> list[Finding]:
    """Source numbers copied into arguments LaTeX numbers itself.

    The system prompt tells the model to strip them, and it mostly does.
    What survives comes out as "12.1 12.1 Electric Charge" or
    "Figure 12.3. Figure 12.3 Field distribution" in the PDF -- readable, so
    nothing fails, which is exactly why it has to be checked.

    Auto-fixable: the pattern is unambiguous, and the repair is to delete the
    number.
    """
    findings: list[Finding] = []
    for command, argument, _start, _end, line in _iter_numbered_arguments(text):
        match = _LEADING_NUMBER_RE.match(argument)
        if not match:
            continue
        findings.append(
            Finding(
                "duplicated-number",
                Severity.AUTO_FIX,
                f"\\{command} repeats the source number {match.group().strip()!r}; "
                "LaTeX numbers it, so the PDF shows the number twice",
                line,
                _excerpt(f"\\{command}{{{argument}}}"),
            )
        )
    return findings


def _check_latex_dangling_commands(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for span in iter_latex_math_spans(text):
        trailing = re.search(r"\\([A-Za-z]+)\s*$", span.body)
        if trailing and trailing.group(1) in _ARGUMENT_COMMANDS:
            findings.append(
                Finding(
                    "dangling-command",
                    Severity.REVIEW,
                    f"\\{trailing.group(1)} ends the {span.kind} math span with no "
                    "argument -- '! Missing } inserted.'",
                    span.line,
                    _excerpt(span.body),
                )
            )
    return findings[:_MAX_EXAMPLES]


def _check_latex_unknown_commands(text: str) -> list[Finding]:
    """Report-only, for the reason in the module docstring: 40 of the 41
    tokens this flagged on the test book were legitimate."""
    seen: dict[str, tuple[int, str]] = {}
    counts: dict[str, int] = {}
    for span in iter_latex_math_spans(text):
        for match in _COMMAND_RE.finditer(span.body):
            name = match.group(1)
            if name in KNOWN_COMMANDS:
                continue
            counts[name] = counts.get(name, 0) + 1
            seen.setdefault(name, (span.line, _excerpt(span.body)))

    if not counts:
        return [Finding("unknown-command", Severity.INFO, "no unrecognized math commands")]

    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    return [
        Finding(
            "unknown-command",
            Severity.REVIEW,
            f"\\{name} appears {count}x in math and is not in KNOWN_COMMANDS -- "
            "check it is real LaTeX, then add it to the list",
            seen[name][0],
            seen[name][1],
        )
        for name, count in ordered[:_MAX_EXAMPLES]
    ]


def latex_image_refs(text: str) -> list[str]:
    """Every figure path referenced by the translated LaTeX, in order.

    Two things in an assembled document are not figure references, and both
    are in ``assets/preamble.tex``: the ``\\bookfig`` definition itself, which
    is ``\\includegraphics{#2}``, and the worked examples in the comments
    above it, which reference ``IMG_0042``.
    """
    without_comments = _COMMENT_RE.sub("", text)
    refs: list[tuple[int, str]] = []
    for pattern, groups in _TEX_IMAGE_RES:
        for match in pattern.finditer(without_comments):
            refs.extend((match.start(), match.group(group).strip()) for group in groups)
    return [ref for _, ref in sorted(refs) if ref and "#" not in ref]


def _check_latex_image_paths(text: str, base_dir: Path | None) -> list[Finding]:
    refs = latex_image_refs(text)
    if not refs:
        return [Finding("image-paths", Severity.INFO, "no figure references")]
    if base_dir is None:
        return [
            Finding(
                "image-paths",
                Severity.INFO,
                f"{len(refs)} figure reference(s); no base directory given, so none "
                "were resolved on disk",
            )
        ]

    missing = [ref for ref in dict.fromkeys(refs) if not _resolves(ref, base_dir)]
    if not missing:
        return [
            Finding("image-paths", Severity.INFO, f"all {len(set(refs))} figure path(s) resolve")
        ]
    return [
        Finding(
            "image-paths",
            Severity.REVIEW,
            f"{len(missing)} figure path(s) do not resolve under {base_dir}",
            context=", ".join(missing[:5]),
        )
    ]


def _resolves(ref: str, base_dir: Path) -> bool:
    if ref.startswith(("http://", "https://", "data:")):
        return True
    return (base_dir / ref).exists() or (base_dir / "images" / Path(ref).name).exists()


def lint_latex(text: str, *, base_dir: Path | None = None) -> LintReport:
    """Run every check that applies to translated LaTeX body text.

    Args:
        text: The concatenated translation, with real image paths restored.
            Body content only -- the preamble is added at assembly, and
            passing a wrapped document here would report its own
            ``\\documentclass`` as leakage.
        base_dir: Directory figure paths resolve against. Omit to skip the
            on-disk check.
    """
    findings: list[Finding] = []
    findings += _check_environments(text)
    findings += _check_preamble_leakage(text)
    findings += _check_latex_braces(text)
    findings += _check_inline_parity(text)
    findings += _check_duplicated_numbers(text)
    findings += _check_latex_dangling_commands(text)
    findings += _check_doubled_commands(text)
    findings += _check_latex_unknown_commands(text)
    findings += _check_latex_image_paths(text, base_dir)
    return LintReport(findings)


def lint_markdown(text: str, *, base_dir: Path | None = None) -> LintReport:
    """Run every source-side check over ``text``.

    This is the parser's output, checked before it is chunked and handed to
    the translating model: a welded ``$$$$`` or a flat heading outline in the
    source is something the model then faithfully reproduces.

    Args:
        text: The markdown to check.
        base_dir: Directory image references resolve against. Omit to skip
            the on-disk image check.
    """
    findings: list[Finding] = []
    findings += _check_dollar_runs(text)
    findings += _check_display_parity(text)
    findings += _check_inline_parity(text)
    findings += _check_delimiter_whitespace(text)
    findings += _check_escaped_dollars(text)
    findings += _check_braces(text)
    findings += _check_multiple_tags(text)
    findings += _check_arrays(text)
    findings += _check_dangling_commands(text)
    findings += _check_doubled_commands(text)
    findings += _check_unknown_commands(text)
    findings += _check_headings(text)
    findings += _check_image_paths(text, base_dir)
    return LintReport(findings)


def apply_auto_fixes(text: str) -> tuple[str, int]:
    """Apply only the provably safe repairs. Returns ``(text, count)``.

    Two things qualify: a known command literally doubled, and a source
    number repeated into a heading or caption LaTeX numbers itself. Both are
    unambiguous -- they cannot be anything but damage, and the repair has one
    possible form. Every other finding is reported for a human.
    """
    count = 0

    def repair(match: re.Match[str]) -> str:
        nonlocal count
        name = match.group(1)
        if name not in KNOWN_COMMANDS:
            return match.group(0)
        count += 1
        return f"\\{name}"

    text, stripped = strip_duplicated_numbers(text)
    return _DOUBLED_COMMAND_RE.sub(repair, text), count + stripped


def strip_duplicated_numbers(text: str) -> tuple[str, int]:
    """Delete source numbers from sectioning and caption arguments.

    ``\\section{12.1 Electric Charge}`` -> ``\\section{Electric Charge}``;
    ``\\section{Electric Charge}`` is left exactly as it is.
    """
    pieces: list[str] = []
    cursor = 0
    count = 0

    for _command, argument, start, end, _line in _iter_numbered_arguments(text):
        match = _LEADING_NUMBER_RE.match(argument)
        if not match:
            continue
        pieces.append(text[cursor:start])
        pieces.append("{" + argument[match.end() :] + "}")
        cursor = end
        count += 1

    if not count:
        return text, 0

    pieces.append(text[cursor:])
    return "".join(pieces), count


# ── Translation-kit checks ──────────────────────────────────────────────────


def lint_chunk_pair(name: str, source: str, translation: str) -> list[Finding]:
    """Compare one source chunk against its translation.

    Three independent failures, none of which the others catch:

    * **Untranslated.** A passed-through chunk is *full length* and perfectly
      formatted, so only the source-script density gives it away.
    * **Truncated.** A reply cut off mid-document reads fine in isolation;
      only the length ratio finds it.
    * **Image tokens.** A dropped or invented ``IMG_nnnn`` becomes a missing
      figure at build time with no earlier warning.
    """
    findings: list[Finding] = []

    ratio = cjk_ratio(translation)
    if ratio > SOURCE_RESIDUE_THRESHOLD:
        findings.append(
            Finding(
                "cjk-residue",
                Severity.REVIEW,
                f"chunk {name} is {ratio * 100:.1f}% CJK (threshold "
                f"{SOURCE_RESIDUE_THRESHOLD * 100:.0f}%) -- it was probably returned "
                "untranslated",
            )
        )

    if source and len(translation) < len(source) * _TRUNCATION_RATIO:
        findings.append(
            Finding(
                "truncated-chunk",
                Severity.REVIEW,
                f"chunk {name} is {len(translation):,} chars against a "
                f"{len(source):,}-char source -- probably truncated",
            )
        )

    source_tokens = set(IMG_TOKEN_RE.findall(source))
    target_tokens = set(IMG_TOKEN_RE.findall(translation))
    lost = sorted(source_tokens - target_tokens)
    invented = sorted(target_tokens - source_tokens)
    if lost:
        findings.append(
            Finding(
                "image-tokens",
                Severity.REVIEW,
                f"chunk {name} lost {len(lost)} image token(s)",
                context=", ".join(f"IMG_{t}" for t in lost[:8]),
            )
        )
    if invented:
        findings.append(
            Finding(
                "image-tokens",
                Severity.REVIEW,
                f"chunk {name} invented {len(invented)} image token(s)",
                context=", ".join(f"IMG_{t}" for t in invented[:8]),
            )
        )

    return findings


def lint_kit(kit_dir: Path) -> LintReport:
    """Check a translation kit: every chunk pair, then the assembled book.

    Expects ``kit_dir/chunks/NNN.md`` (the source chunk, still Markdown --
    that is what MinerU produces) and ``kit_dir/translated/NNN.tex`` (the
    saved reply, which is LaTeX). Missing translations are reported, not
    raised on.
    """
    chunks_dir = kit_dir / "chunks"
    translated_dir = kit_dir / "translated"

    if not chunks_dir.is_dir():
        return LintReport(
            [Finding("kit-layout", Severity.REVIEW, f"{chunks_dir} not found")]
        )

    sources = sorted(chunks_dir.glob("*.md"))
    if not sources:
        return LintReport(
            [Finding("kit-layout", Severity.REVIEW, f"no source chunks in {chunks_dir}")]
        )

    findings: list[Finding] = []
    translations: list[str] = []
    ratios: list[tuple[str, float]] = []

    for source_path in sources:
        target_path = translated_dir / f"{source_path.stem}{TRANSLATED_SUFFIX}"
        if not target_path.exists():
            findings.append(
                Finding(
                    "missing-translation",
                    Severity.REVIEW,
                    f"chunk {source_path.stem} has no translation at {target_path}",
                )
            )
            continue
        source = source_path.read_text(encoding="utf-8")
        translation = strip_wrapping_fence(target_path.read_text(encoding="utf-8"))
        findings += lint_chunk_pair(source_path.stem, source, translation)
        translations.append(translation)
        if source.strip():
            ratios.append((source_path.stem, len(translation) / len(source)))

    findings += _check_expansion_outliers(ratios)

    map_path = kit_dir / "image_map.json"

    if translations:
        combined = "\n\n".join(translations)
        overall = cjk_ratio(combined)
        findings.append(
            Finding(
                "cjk-residue",
                Severity.REVIEW if overall > SOURCE_RESIDUE_THRESHOLD else Severity.INFO,
                f"whole book is {overall * 100:.2f}% CJK across {len(translations)} chunk(s)",
            )
        )
        # Restore the real paths before the LaTeX checks run. The saved
        # replies still carry IMG_nnnn tokens, so linting them as-is would
        # report every single token as a figure that does not resolve.
        findings += lint_latex(
            _restore_for_lint(combined, map_path), base_dir=kit_dir
        ).findings

    if map_path.exists():
        findings += _check_image_map(kit_dir, map_path, translations)

    return LintReport(findings)


def _restore_for_lint(text: str, map_path: Path) -> str:
    """Substitute real image paths back in, for checking purposes only.

    ``src.kit`` imports this module, so the import has to be function-level
    to keep the dependency one-way. Restoring is best-effort here: an
    unreadable map just means the image-path check has nothing to resolve,
    which ``_check_image_map`` reports separately.
    """
    if not map_path.exists():
        return text
    try:
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return text

    from src.kit import restore_images

    return restore_images(text, mapping)[0]


def _check_expansion_outliers(ratios: list[tuple[str, float]]) -> list[Finding]:
    """Flag chunks that expanded far less than the book's own median.

    Self-calibrating, so it works whatever the language pair is: if every
    other chunk grew 3x and this one grew 1.2x, this one lost content. This
    catches the truncations the absolute floor in ``lint_chunk_pair`` is too
    lenient to see -- a reply cut off halfway is still well above 40% of its
    source's length when the source is CJK.
    """
    if len(ratios) < _MIN_CHUNKS_FOR_MEDIAN:
        return []

    ordered = sorted(r for _, r in ratios)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    if median <= 0:
        return []

    threshold = median * _TRUNCATION_OUTLIER_RATIO
    return [
        Finding(
            "truncated-chunk",
            Severity.REVIEW,
            f"chunk {name} expanded {ratio:.2f}x against a book median of "
            f"{median:.2f}x -- far short of its neighbours, so it was probably cut off",
        )
        for name, ratio in ratios
        if ratio < threshold
    ]


def _check_image_map(kit_dir: Path, map_path: Path, translations: list[str]) -> list[Finding]:
    try:
        token_to_path: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Finding("image-map", Severity.REVIEW, f"{map_path} is unreadable: {exc}")]

    used = set(IMG_TOKEN_RE.findall("\n".join(translations)))
    expected = {token.removeprefix("IMG_") for token in token_to_path}
    never_used = sorted(expected - used)
    if never_used:
        return [
            Finding(
                "image-map",
                Severity.REVIEW,
                f"{len(never_used)} image(s) from the source never appear in the "
                "translation",
                context=", ".join(f"IMG_{t}" for t in never_used[:8]),
            )
        ]
    return [
        Finding("image-map", Severity.INFO, f"all {len(token_to_path)} image token(s) survived")
    ]


_FENCE_OPEN_RE = re.compile(r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```\s*\Z")


def strip_wrapping_fence(text: str) -> str:
    """Drop a ``` fence wrapped around a whole reply.

    AI Studio adds ````latex` around the answer regardless of the prompt
    telling it not to, and the fence markers are not LaTeX: left in, they are
    typeset into the book as three literal backticks.
    """
    return _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text))


__all__ = [
    "IMG_TOKEN_RE",
    "KNOWN_COMMANDS",
    "TRANSLATED_SUFFIX",
    "Finding",
    "LintReport",
    "Severity",
    "apply_auto_fixes",
    "brace_imbalance",
    "cjk_ratio",
    "latex_image_refs",
    "lint_chunk_pair",
    "lint_kit",
    "lint_latex",
    "lint_markdown",
    "strip_duplicated_numbers",
    "strip_wrapping_fence",
]
