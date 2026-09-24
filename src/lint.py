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
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.config import SOURCE_RESIDUE_THRESHOLD
from src.latex import (
    DEFINED_ENVIRONMENTS,
    brace_imbalance,
    find_preamble_leakage,
    load_preamble,
    preamble_first_chapter,
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

# Structural commands, which `KNOWN_COMMANDS` deliberately does not cover:
# that list is a *math* vocabulary, checked only inside math spans.
#
# The doubled-command check needs these as well. `\sectionsection` is exactly
# the same defect as `\mathrmmathrm`, and it reached a real build and stopped
# it -- five times across four chunks -- because `section` was not in any list
# the check consulted, so nothing reported it and nothing repaired it.
_STRUCTURE_COMMANDS: frozenset[str] = frozenset(
    """
chapter section subsection subsubsection paragraph subparagraph part
caption label ref eqref item textbf textit texttt textrm emph underline
centering raggedright raggedleft footnote textsuperscript textsubscript
bookfig bookfigtwo includegraphics begin end
""".split()
)

# What a doubled command may be repaired to: anything this module recognises
# in either vocabulary.
_REPAIRABLE_COMMANDS: frozenset[str] = KNOWN_COMMANDS | _STRUCTURE_COMMANDS

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

# Commands that are never meaningfully written twice in a row, so
# `\section\section{...}` or `\mathrm\mathrm{mm}` can only be damage. The
# second spelling of the doubled-command defect: the first (`\sectionsection`)
# drops the inner backslash, this one keeps it, and it stopped a real build
# the same way. Deliberately excludes accents and operators -- `\bar\bar{x}`,
# `\prime\prime`, `\Psi\Psi^*` and `\quad\quad` are all legitimate.
_NEVER_REPEATED_COMMANDS: frozenset[str] = frozenset(
    """
    mathrm mathbf mathit mathcal mathbb mathfrak mathsf mathtt mathscr
    boldsymbol pmb operatorname text textbf textit textrm textsf texttt emph
    mbox chapter section subsection subsubsection paragraph subparagraph part
    caption label textsuperscript textsubscript centering bookfig bookfigtwo
    includegraphics
    """.split()
)

# Commands that need a package, and the package(s) that provide them. A
# model reproducing a source table reaches for `\multirow` whether or not the
# preamble loads it, and the result is "! Undefined control sequence." with
# nothing in the text looking wrong. Not exhaustive -- only commands a
# textbook translation plausibly emits.
_PACKAGE_FOR_COMMAND: dict[str, tuple[str, ...]] = {
    "multirow": ("multirow",),
    "makecell": ("makecell",),
    "thead": ("makecell",),
    "cancel": ("cancel",),
    "bcancel": ("cancel",),
    "xcancel": ("cancel",),
    "SI": ("siunitx",),
    "si": ("siunitx",),
    "num": ("siunitx",),
    "qty": ("siunitx",),
    "unit": ("siunitx",),
    "ce": ("mhchem",),
    "mathscr": ("mathrsfs",),
    "ding": ("pifont",),
    "bm": ("bm",),
    "uline": ("ulem",),
    "uwave": ("ulem",),
    "sout": ("ulem",),
    "hl": ("soul",),
    "cellcolor": ("colortbl", "xcolor"),
    "rowcolor": ("colortbl", "xcolor"),
    "color": ("color", "xcolor"),
    "textcolor": ("color", "xcolor"),
    "degree": ("gensymb",),
    "celsius": ("gensymb",),
    "ohm": ("gensymb", "siunitx"),
    "micro": ("gensymb", "siunitx"),
    "upmu": ("upgreek",),
    "coloneqq": ("mathtools",),
    "mathclap": ("mathtools",),
    "toprule": ("booktabs",),
    "midrule": ("booktabs",),
    "bottomrule": ("booktabs",),
    "url": ("url", "hyperref"),
    "href": ("hyperref",),
}

# Text-mode commands that do not work in math: LaTeX warns "Command
# \textcircled invalid in math mode" and typesets something else. Four came
# through a real build, from source footnote markers like `^{\textcircled{1}}`.
_TEXT_ONLY_COMMANDS: frozenset[str] = frozenset(
    "textcircled textsuperscript textsubscript textdegree textmu".split()
)

# Groups inside math that switch back to text mode, where the commands above
# are fine: `$\text{\textcircled{1}}$` is the correct spelling.
_TEXT_GROUP_RE = re.compile(r"\\(?:text|textrm|textbf|textit|mbox|hbox)\s*\{")

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_DOUBLED_COMMAND_RE = re.compile(r"\\([A-Za-z]+?)\1(?![A-Za-z])")
_REPEATED_COMMAND_RE = re.compile(r"\\([A-Za-z]+)(?:\s*\\\1(?![A-Za-z]))+")
_USEPACKAGE_RE = re.compile(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_DEFINED_COMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\s*\{?\\([A-Za-z]+)"
    r"|\\def\s*\\([A-Za-z]+)"
)
_CHAPTER_RE = re.compile(r"\\chapter(\*?)\s*(?:\[[^\]]*\])?\s*\{")
_CHAPTER_TAG_RE = re.compile(r"\\tag\s*\{\s*(\d+)\.\d+")
# A Markdown heading left in the LaTeX. At the start of a line `#` is TeX's
# macro-parameter character: "! You can't use `macro parameter character #'".
_MARKDOWN_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+\S.*$", re.MULTILINE)
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
    r"\\(chapter|section|subsection|subsubsection|caption)(\*?)(?:\[[^\]]*\])?\{"
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


def _is_doubled_command(name: str) -> bool:
    """``"mathrmmathrm"`` -> True. A known command written twice over one
    backslash, which ``_check_doubled_commands`` reports and repairs."""
    half, odd = divmod(len(name), 2)
    return not odd and name[:half] == name[half:] and name[:half] in _REPAIRABLE_COMMANDS


def _check_doubled_commands(text: str) -> list[Finding]:
    """A known command written twice: over one backslash (``\\mathrmmathrm``),
    or with both backslashes (``\\section\\section``).

    Auto-fixable: it cannot be anything but damage, and the repair is
    unambiguous. The second form is limited to `_NEVER_REPEATED_COMMANDS`,
    because ``\\prime\\prime`` and ``\\bar\\bar{x}`` are real LaTeX.
    """
    findings: list[Finding] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        for match in _DOUBLED_COMMAND_RE.finditer(line):
            if match.group(1) in _REPAIRABLE_COMMANDS:
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
        for match in _REPEATED_COMMAND_RE.finditer(line):
            if match.group(1) in _NEVER_REPEATED_COMMANDS:
                findings.append(
                    Finding(
                        "doubled-command",
                        Severity.AUTO_FIX,
                        f"\\{match.group(1)} is written twice ({match.group(0)}) -- the "
                        "first one takes the second as its argument, which stops the build",
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


def _iter_numbered_arguments(text: str, *, starred: bool = False):
    """``(command, argument, start, end, line)`` for each sectioning/caption
    argument LaTeX numbers by itself -- or, with ``starred=True``, for each
    starred one, which LaTeX does not number."""
    for match in _NUMBERED_ARG_RE.finditer(text):
        if bool(match.group(2)) != starred:
            continue
        found = _braced_argument(text, match.end() - 1)
        if found is None:
            continue
        argument, end = found
        line = text.count("\n", 0, match.start()) + 1
        yield match.group(1), argument, match.end() - 1, end, line


def _check_numbered_starred_headings(text: str) -> list[Finding]:
    """A starred heading whose title starts with a number.

    LaTeX does not number ``\\subsection*``, so nothing is duplicated -- but
    on a real book every one of these was a numbered list item in a chapter
    summary ("4. Magnetic Field Intensity Vector") that MinerU had parsed as
    a heading. Its siblings stayed plain "1. ...", "2. ..." paragraphs, so
    the PDF showed item 4 as a bold heading in the middle of a list. Which
    form is right depends on the book, so this only reports.
    """
    findings: list[Finding] = []
    for command, argument, _start, _end, line in _iter_numbered_arguments(text, starred=True):
        match = _LEADING_NUMBER_RE.match(argument)
        if not match or command == "caption":
            continue
        findings.append(
            Finding(
                "numbered-starred-heading",
                Severity.REVIEW,
                f"\\{command}* starts with the number {match.group().strip()!r}. This is "
                "usually a numbered list item the parser turned into a heading; if "
                "its neighbours are plain numbered paragraphs, make it one too",
                line,
                _excerpt(f"\\{command}*{{{argument}}}"),
            )
        )
    return findings[:_MAX_EXAMPLES]


def _check_duplicated_numbers(text: str) -> list[Finding]:
    """Source numbers copied into arguments LaTeX numbers itself.

    The system prompt tells the model to strip them, and it mostly does.
    What survives comes out as "12.1 12.1 Electric Charge" or
    "Figure 12.3. Figure 12.3 Field distribution" in the PDF -- readable, so
    nothing fails, which is exactly why it has to be checked.

    Auto-fixable: the pattern is unambiguous, and the repair is to delete the
    number. Starred headings are skipped: LaTeX does not number them, so
    there is nothing to duplicate (see `_check_numbered_starred_headings`).
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


# Characters above Latin-1 that `inputenc`'s utf8 support does define, so
# pdflatex typesets them rather than stopping: the General Punctuation that
# maps onto a real font glyph. Everything above U+00FF that is not here is
# "! LaTeX Error: Unicode character X not set up for use with LaTeX."
_TYPESETTABLE_PUNCTUATION: frozenset[str] = frozenset(
    "–—‘’‚“”„"
    "†‡•…‰‹›"
)

# Invisible characters that carry no meaning at all. They survive a copy out
# of a chat window and are impossible to see in an editor, and each one is a
# fatal pdflatex error -- so deleting them is the only possible repair.
_ZERO_WIDTH: frozenset[str] = frozenset("​‌‍⁠﻿")

_LATIN1_CEILING = 0xFF


def _untypesettable_characters(text: str) -> dict[str, list[int]]:
    """``{character: [line, ...]}`` for everything pdflatex cannot typeset."""
    found: dict[str, list[int]] = {}
    for lineno, line in enumerate(text.split("\n"), start=1):
        for char in line:
            if ord(char) <= _LATIN1_CEILING or char in _TYPESETTABLE_PUNCTUATION:
                continue
            found.setdefault(char, []).append(lineno)
    return found


def _check_unicode_characters(text: str) -> list[Finding]:
    """Literal Unicode the model emitted instead of a LaTeX command.

    A real book came back with 482 of these across 60 distinct characters:
    ``ε`` for ``$\\varepsilon$``, ``−`` (U+2212) for a math minus, ``θ``,
    ``①``, six stray Chinese characters, and 94 zero-width spaces. Under
    pdflatex + inputenc every single one is fatal, **one error per compile** --
    which is 482 build cycles to find them by compiling.

    The CJK ones overlap with ``cjk-residue``, but that check measures a
    *ratio*: six characters in a 700,000-character book is 0.01%, far below
    any sane residue threshold, and still six dead builds.
    """
    found = _untypesettable_characters(text)
    if not found:
        return []

    findings: list[Finding] = []
    for char, lines in sorted(found.items(), key=lambda kv: -len(kv[1])):
        name = unicodedata.name(char, "unnamed character")
        zero_width = char in _ZERO_WIDTH
        findings.append(
            Finding(
                "zero-width-character" if zero_width else "unicode-character",
                Severity.AUTO_FIX if zero_width else Severity.REVIEW,
                (
                    f"invisible U+{ord(char):04X} ({name}) appears {len(lines)}x and "
                    "cannot be typeset; it carries no meaning, so it is deleted"
                    if zero_width
                    else f"literal '{char}' (U+{ord(char):04X}, {name}) appears "
                    f"{len(lines)}x -- pdflatex cannot typeset it. Replace it with "
                    "the LaTeX it stands for, or build with XeLaTeX (see the "
                    "commented swap in assets/preamble.tex)"
                ),
                lines[0],
                f"also on line(s): {', '.join(str(n) for n in lines[1:6])}"
                if len(lines) > 1
                else "",
            )
        )
    return findings[:_MAX_EXAMPLES]


def strip_zero_width(text: str) -> tuple[str, int]:
    """Delete zero-width characters. Returns ``(text, count)``."""
    count = sum(text.count(char) for char in _ZERO_WIDTH)
    if not count:
        return text, 0
    for char in _ZERO_WIDTH:
        text = text.replace(char, "")
    return text, count


# Unnumbered display math, in both spellings the translation uses.
_DISPLAY_BLOCK_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("$$ ... $$", re.compile(r"\$\$(.*?)\$\$", re.DOTALL)),
    ("\\[ ... \\]", re.compile(r"\\\[(.*?)\\\]", re.DOTALL)),
)


def _check_misplaced_tags(text: str) -> list[Finding]:
    """``\\tag`` inside display math that cannot carry one.

    amsmath allows ``\\tag`` only in an equation-like environment; in
    ``$$...$$`` or ``\\[...\\]`` it is "! Package amsmath Error: \\tag not
    allowed here." The reference book tags 570 of its equations and the
    prompt says to keep every tag, so this arrives whenever the model picks
    the unnumbered form for one of them.

    Auto-fixable: a tag *is* the book's equation number, so the block is a
    numbered equation written in the wrong wrapper. Promoting it to
    ``equation`` is the repair the prompt already asks for.
    """
    findings: list[Finding] = []
    for label, pattern in _DISPLAY_BLOCK_RES:
        for match in pattern.finditer(text):
            tags = _TAG_RE.findall(match.group(1))
            if not tags:
                continue
            line = text.count("\n", 0, match.start()) + 1
            if len(tags) > 1:
                # Two tags is a different defect, and not this one's to fix:
                # amsmath rejects the second one wherever the block ends up,
                # and choosing between them means looking at the book.
                findings.append(
                    Finding(
                        "multiple-tags",
                        Severity.REVIEW,
                        f"{label} carries {len(tags)} \\tag commands "
                        f"({', '.join(repr(t) for t in tags)}); amsmath allows one. "
                        "Delete whichever is not the real equation number.",
                        line,
                        _excerpt(match.group(1)),
                    )
                )
                continue
            findings.append(
                Finding(
                    "misplaced-tag",
                    Severity.AUTO_FIX,
                    f"\\tag{{{tags[0]}}} sits in {label}, which cannot carry one -- "
                    "'! Package amsmath Error: \\tag not allowed here.' The block is "
                    "a numbered equation, so it becomes \\begin{equation}",
                    line,
                    _excerpt(match.group(1)),
                )
            )
    return findings[:_MAX_EXAMPLES]


def promote_tagged_display(text: str) -> tuple[str, int]:
    """Wrap tagged display blocks in ``equation``. Returns ``(text, count)``."""
    count = 0

    def repair(match: re.Match[str]) -> str:
        nonlocal count
        body = match.group(1)
        # Exactly one tag. A block carrying two is broken in a way that
        # promoting it does not fix -- amsmath rejects the second tag inside
        # `equation` as well -- so it is left for the human it needs.
        if len(_TAG_RE.findall(body)) != 1:
            return match.group(0)
        count += 1
        return f"\\begin{{equation}}{body}\\end{{equation}}"

    for _label, pattern in _DISPLAY_BLOCK_RES:
        text = pattern.sub(repair, text)
    return text, count


def _bare_display_delimiters(text: str) -> list[tuple[int, str]]:
    """``(line, "[" or "]")`` for every line that is nothing but a bracket.

    A display block opened with ``[`` instead of ``\\[``. Measured on a real
    book: 971 blocks correct, **94 with the backslash dropped** -- so LaTeX
    typesets a literal bracket and runs the mathematics after it in text
    mode, which fails on the first ``\\frac``. A lone bracket on its own line
    is never anything else; a bracket in prose sits inside a sentence.
    """
    return [
        (lineno, line.strip())
        for lineno, line in enumerate(text.split("\n"), start=1)
        if line.strip() in ("[", "]")
    ]


def _delimiters_are_repairable(hits: list[tuple[int, str]]) -> bool:
    """True when the bare brackets strictly alternate ``[``, ``]``, ``[`` ...

    Anything else -- an unpaired opener, two in a row -- means guessing which
    bracket belongs to which block, so those are reported instead.
    """
    return bool(hits) and all(
        bracket == ("[" if index % 2 == 0 else "]")
        for index, (_line, bracket) in enumerate(hits)
    )


def _check_display_delimiters(text: str) -> list[Finding]:
    hits = _bare_display_delimiters(text)
    if not hits:
        return []

    repairable = _delimiters_are_repairable(hits)
    severity = Severity.AUTO_FIX if repairable else Severity.REVIEW
    detail = (
        "lost its backslash"
        if repairable
        else "lost its backslash, and the brackets do not pair up, so the repair "
        "needs a human"
    )
    findings = [
        Finding(
            "display-delimiter",
            severity,
            f"a line containing only `{bracket}` -- a display-math `\\{bracket}` "
            f"that {detail}; the mathematics after it runs in text mode",
            lineno,
        )
        for lineno, bracket in hits[:_MAX_EXAMPLES]
    ]
    if len(hits) > _MAX_EXAMPLES:
        # Without this the summary line reads "12 auto-fixable" and `--fix`
        # then reports repairing 188 -- on a real book this check fires in the
        # hundreds, and the cap is per-check, not per-book.
        findings.append(
            Finding(
                "display-delimiter",
                severity,
                f"... and {len(hits) - _MAX_EXAMPLES} more bare `[` / `]` line(s), "
                f"{len(hits)} in total",
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
            if name in KNOWN_COMMANDS or _is_doubled_command(name):
                # A doubled command (\mathrmmathrm) is unknown by definition,
                # but `_check_doubled_commands` already owns it and can repair
                # it. Reporting it here too would gate the build on a defect
                # `--fix` clears, under the advice "add it to KNOWN_COMMANDS",
                # which is the one thing that must not happen to it.
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


def _check_markdown_headings(text: str) -> list[Finding]:
    """A Markdown heading the model passed through instead of converting.

    Three reached a real build, and each stopped it: at the start of a line
    ``#`` is TeX's macro-parameter character. All three were numbered summary
    items ("### 2. Electron Spin ...") that MinerU had parsed as headings, so
    the right repair depends on context and this only reports.
    """
    findings = [
        Finding(
            "markdown-heading",
            Severity.REVIEW,
            f"a Markdown `{match.group(1)}` heading in the LaTeX -- '! You can't use "
            "`macro parameter character #' in vertical mode.' Convert it to \\section/\\subsection, or "
            "to a plain numbered paragraph if it is a list item",
            text.count("\n", 0, match.start()) + 1,
            _excerpt(match.group(0)),
        )
        for match in _MARKDOWN_HEADING_RE.finditer(_COMMENT_RE.sub("", text))
    ]
    return findings[:_MAX_EXAMPLES]


def _preamble_provides(preamble: str) -> tuple[set[str], set[str]]:
    """``(packages loaded, commands defined)`` by ``preamble``, ignoring
    commented-out lines -- the XeLaTeX swap sits there as comments."""
    live = _COMMENT_RE.sub("", preamble)
    packages = {
        name.strip()
        for group in _USEPACKAGE_RE.findall(live)
        for name in group.split(",")
        if name.strip()
    }
    defined = {a or b for a, b in _DEFINED_COMMAND_RE.findall(live)}
    return packages, defined


def _check_required_packages(text: str, preamble: str | None) -> list[Finding]:
    """A command whose package the preamble does not load.

    ``\\multirow`` in a translated table stopped a real build: the text is
    perfectly ordinary LaTeX, and only the preamble decides whether it
    compiles. Skipped when no preamble is given (a lone fragment).
    """
    if preamble is None:
        return []
    packages, defined = _preamble_provides(preamble)

    first_line: dict[str, int] = {}
    counts: dict[str, int] = {}
    for lineno, line in enumerate(_COMMENT_RE.sub("", text).split("\n"), start=1):
        for name in _COMMAND_RE.findall(line):
            providers = _PACKAGE_FOR_COMMAND.get(name)
            if providers is None or name in defined or packages.intersection(providers):
                continue
            first_line.setdefault(name, lineno)
            counts[name] = counts.get(name, 0) + 1

    return [
        Finding(
            "missing-package",
            Severity.REVIEW,
            f"\\{name} appears {counts[name]}x but the preamble does not load "
            f"{' or '.join(_PACKAGE_FOR_COMMAND[name])} -- '! Undefined control "
            f"sequence.' Add \\usepackage{{{_PACKAGE_FOR_COMMAND[name][0]}}} to "
            "assets/preamble.tex",
            line,
        )
        for name, line in sorted(first_line.items(), key=lambda kv: kv[1])
    ][:_MAX_EXAMPLES]


def _without_text_groups(body: str) -> str:
    """``body`` with every ``\\text{...}``-style group removed, since text
    mode inside math is exactly where text commands belong."""
    pieces: list[str] = []
    cursor = 0
    for match in _TEXT_GROUP_RE.finditer(body):
        if match.start() < cursor:
            continue
        found = _braced_argument(body, match.end() - 1)
        if found is None:
            continue
        pieces.append(body[cursor : match.start()])
        cursor = found[1]
    pieces.append(body[cursor:])
    return "".join(pieces)


def _check_text_commands_in_math(text: str) -> list[Finding]:
    """A text-mode command used directly in math (``^{\\textcircled{1}}``).

    Not fatal -- LaTeX warns "Command \\textcircled invalid in math mode" and
    carries on -- but what it typesets is not the symbol. Wrapping it in
    ``\\text{...}`` is the fix.
    """
    findings: list[Finding] = []
    for span in iter_latex_math_spans(text):
        for name in _COMMAND_RE.findall(_without_text_groups(span.body)):
            if name in _TEXT_ONLY_COMMANDS:
                findings.append(
                    Finding(
                        "text-command-in-math",
                        Severity.REVIEW,
                        f"\\{name} is a text-mode command inside {span.kind} math -- "
                        f"LaTeX typesets it wrongly. Wrap it: \\text{{\\{name}{{...}}}}, "
                        "or delete it if it is a stray footnote marker",
                        span.line,
                        _excerpt(span.body),
                    )
                )
    return findings[:_MAX_EXAMPLES]


def _check_chapter_numbering(text: str, preamble: str | None) -> list[Finding]:
    """Chapter numbers that drift away from the book's own equation tags.

    Every ``\\tag{15.3}`` says which chapter the book thinks it is in, and
    LaTeX's count of numbered ``\\chapter`` commands says which one the PDF
    will print. On a real book they drifted by up to three: five
    supplementary readings and one part heading ("Optics") had come through
    as numbered chapters. Each one pushed every later chapter up by one, and
    nothing failed -- the PDF just said "Chapter 24" over chapter 22.

    Reports where each drift starts, naming the untagged chapters since the
    last one that agreed -- the likely culprits. Needs the preamble for the
    starting number, so a lone fragment is skipped.
    """
    if preamble is None:
        return []

    number = preamble_first_chapter(preamble)
    matches = list(_CHAPTER_RE.finditer(text))
    findings: list[Finding] = []
    previous_drift = 0
    suspects: list[str] = []
    last_tagged = ""

    for index, match in enumerate(matches):
        if match.group(1):  # \chapter* takes no number
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found = _braced_argument(text, match.end() - 1)
        title = _excerpt(found[0], 60) if found else "?"

        prefixes = [int(p) for p in _CHAPTER_TAG_RE.findall(text, match.end(), end)]
        if not prefixes:
            suspects.append(title)
            number += 1
            continue

        tagged = max(set(prefixes), key=prefixes.count)
        drift = number - tagged
        if drift not in (previous_drift, 0):
            cause = (
                "an earlier \\chapter is probably a supplementary reading that should "
                "be \\chapter*, or a \\part written as \\chapter"
                if drift > 0
                else "a chapter heading is probably missing, or was written as \\chapter*"
            )
            where = f"since '{last_tagged}'" if last_tagged else "before it"
            candidates = f" Untagged chapters {where}: {', '.join(suspects)}." if suspects else ""
            findings.append(
                Finding(
                    "chapter-numbering",
                    Severity.REVIEW,
                    f"\\chapter{{{title}}} will be printed as chapter {number}, but its "
                    f"equations are tagged {tagged}.x -- {cause}.{candidates}",
                    text.count("\n", 0, match.start()) + 1,
                )
            )
        previous_drift = drift
        last_tagged = title
        suspects = []
        number += 1

    return findings[:_MAX_EXAMPLES]


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


def lint_latex(
    text: str, *, base_dir: Path | None = None, preamble: str | None = None
) -> LintReport:
    """Run every check that applies to translated LaTeX body text.

    Args:
        text: The concatenated translation, with real image paths restored.
            Body content only -- the preamble is added at assembly, and
            passing a wrapped document here would report its own
            ``\\documentclass`` as leakage.
        base_dir: Directory figure paths resolve against. Omit to skip the
            on-disk check.
        preamble: The preamble the body will be compiled with. Omit to skip
            the checks that depend on it (loaded packages, the starting
            chapter number) -- right for a lone fragment, which is neither a
            whole book nor necessarily headed for this preamble.
    """
    findings: list[Finding] = []
    findings += _check_environments(text)
    findings += _check_preamble_leakage(text)
    findings += _check_unicode_characters(text)
    findings += _check_display_delimiters(text)
    findings += _check_misplaced_tags(text)
    findings += _check_latex_braces(text)
    findings += _check_inline_parity(text)
    findings += _check_duplicated_numbers(text)
    findings += _check_numbered_starred_headings(text)
    findings += _check_markdown_headings(text)
    findings += _check_latex_dangling_commands(text)
    findings += _check_doubled_commands(text)
    findings += _check_latex_unknown_commands(text)
    findings += _check_text_commands_in_math(text)
    findings += _check_required_packages(text, preamble)
    findings += _check_chapter_numbering(text, preamble)
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

    Three things qualify: a known command literally doubled, a source number
    repeated into a heading or caption LaTeX numbers itself, and a display
    delimiter that lost its backslash. Each is unambiguous -- it cannot be
    anything but damage, and the repair has exactly one possible form. Every
    other finding is reported for a human.
    """
    count = 0

    def repair(match: re.Match[str]) -> str:
        nonlocal count
        name = match.group(1)
        if name not in _REPAIRABLE_COMMANDS:
            return match.group(0)
        count += 1
        return f"\\{name}"

    def collapse(match: re.Match[str]) -> str:
        nonlocal count
        name = match.group(1)
        if name not in _NEVER_REPEATED_COMMANDS:
            return match.group(0)
        count += 1
        return f"\\{name}"

    text, stripped = strip_duplicated_numbers(text)
    # Delimiters first: a block whose `\[` lost its backslash has to be put
    # back together before it can be recognised as a tagged equation.
    text, restored = restore_display_delimiters(text)
    text, promoted = promote_tagged_display(text)
    text, invisible = strip_zero_width(text)
    text = _REPEATED_COMMAND_RE.sub(collapse, text)
    return (
        _DOUBLED_COMMAND_RE.sub(repair, text),
        count + stripped + restored + promoted + invisible,
    )


def restore_display_delimiters(text: str) -> tuple[str, int]:
    """Put the backslash back on bare ``[`` / ``]`` display delimiters.

    Only when they strictly alternate across the whole document, so every
    opener has its own closer. If they do not, nothing is touched and
    ``_check_display_delimiters`` reports them for review instead.
    """
    hits = _bare_display_delimiters(text)
    if not _delimiters_are_repairable(hits):
        return text, 0

    targets = dict(hits)
    lines = text.split("\n")
    for lineno, bracket in targets.items():
        line = lines[lineno - 1]
        lines[lineno - 1] = line.replace(bracket, f"\\{bracket}", 1)
    return "\n".join(lines), len(targets)


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
        source_prose = _prose_length(source)
        if source_prose:
            ratios.append((source_path.stem, _prose_length(translation) / source_prose))

    findings += _check_expansion_outliers(ratios)

    map_path = kit_dir / "image_map.json"
    try:
        preamble: str | None = load_preamble()
    except FileNotFoundError:
        preamble = None

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
            _restore_for_lint(combined, map_path), base_dir=kit_dir, preamble=preamble
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


# Everything that is carried over rather than translated: math on both sides,
# plus HTML tables in the source and tabular/table environments in the
# translation.
_NON_PROSE_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|<table\b.*?</table>"
    r"|\\begin\{(equation|align|gather|multline|tabular|table|longtable)\*?\}"
    r".*?\\end\{\1\*?\}",
    re.DOTALL,
)


def _prose_length(text: str) -> int:
    """Length of ``text`` without its math and tables.

    Math and tables pass through translation at about their own length,
    while prose grows roughly threefold (CJK to English). Measured on raw
    length, a chunk that is mostly formulas and tables therefore looks
    "short": a real book's chunk 018 -- 12% CJK, the rest math and tables --
    sat at 0.35x the median raw expansion and was reported as truncated,
    though it matched its source paragraph for paragraph. On prose alone it
    sat at 0.99x, and every chunk of that book fell between 0.93x and 1.1x.
    """
    return len(_NON_PROSE_RE.sub("", text))


def _check_expansion_outliers(ratios: list[tuple[str, float]]) -> list[Finding]:
    """Flag chunks whose prose expanded far less than the book's own median.

    Self-calibrating, so it works whatever the language pair is: if every
    other chunk grew 3x and this one grew 1.2x, this one lost content. This
    catches the truncations the absolute floor in ``lint_chunk_pair`` is too
    lenient to see -- a reply cut off halfway is still well above 40% of its
    source's length when the source is CJK. The ratios are of prose length
    (see `_prose_length`), so a math-heavy chunk is not mistaken for a cut-off
    one.
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
            f"chunk {name}'s prose expanded {ratio:.2f}x against a book median of "
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
    "promote_tagged_display",
    "restore_display_delimiters",
    "strip_duplicated_numbers",
    "strip_zero_width",
    "strip_wrapping_fence",
]
