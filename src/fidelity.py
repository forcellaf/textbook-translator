"""
Semantic fidelity: does the translation still say the same mathematics?

Why this module exists
----------------------
``src.lint`` proves the translation *compiles*. Nothing proved it says the
same thing: a model that turns ``10^{-20}`` into ``10^{-2}``, flips a sign,
drops a display or a ``\\tag``, or swaps two subscripts produces LaTeX that
compiles, reads fluently, and is wrong. This is a deterministic, LLM-free
comparison of each source chunk (MinerU's Markdown) with its translation
(LaTeX), run by ``lint_kit`` for every chunk pair.

What it cannot catch -- read this
---------------------------------
It catches **translation drift only**. A hallucination in the *parse* -- the
VLM reading an exponent wrong off the page -- is in the source chunk too, so
both sides agree and every check here passes. Parse fidelity needs the page
image; this module never sees it. Every run log lists this as a known
limitation.

Normalisation
-------------
The two sides write the same mathematics differently on purpose: the source
uses ``$$``, the translation ``\\[`` or ``equation``; the ``pipeline`` parser
spaces every character (``1 0 ^ { - 2 0 }``); the model adds ``\\mathrm`` and
``\\,``; prose inside ``\\text{}`` is *supposed* to change language. So both
sides are reduced to a canonical token stream first:

* delimiters unified (``$``, ``$$``, ``\\[ \\]``, ``\\( \\)``, ``equation`` /
  ``align`` / ``gather`` ... and their inner ``aligned`` / ``split``);
* the parser's character spacing removed between digits;
* Unicode minus, ``×``, ``·`` and friends mapped to their LaTeX forms;
* presentation-only commands dropped, their arguments kept (``\\mathrm``,
  ``\\rm``, ``\\,``, ``\\quad``, ``\\displaystyle``, ``\\left`` / ``\\right``,
  the ``\\big`` family, alignment ``&`` and ``\\\\``);
* ``\\text{}``, ``\\mbox{}``, ``\\textrm{}`` contents dropped, and any CJK
  left in the math: those are prose, translated on purpose;
* aliases unified (``\\le`` / ``\\leq``, ``\\dfrac`` / ``\\frac``, ``^\\prime``
  / ``'`` ...) and braces around a single token removed.

Checks
------
=========================  ==========================================  ========
``fidelity-tags``          equation tags (``\\tag{12.3}``, or ``(12.3)``  review
                           closing a display) as a multiset
``fidelity-display-count`` number of display blocks                    review
``fidelity-numbers``       numeric literals in math, signed exponents  review
                           included, as a multiset
``fidelity-blocks``        display blocks aligned in order: token       review
                           similarity, and the sign/relation operators
``fidelity-prose-numbers`` Arabic numerals in source prose missing      info
                           from the translation's prose
=========================  ==========================================  ========

The prose check is information only: Chinese numerals, rewritten units and
section numbers stripped on purpose (LaTeX numbers sections itself) make it
noisy. False positives elsewhere are what ``kit/lint_waivers.json`` is for.

Thresholds are module constants, calibrated on the 89 translated chunk pairs
of five real books so that formatting-only differences stay quiet.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.findings import Finding, Severity

# ── Thresholds (see the module docstring) ───────────────────────────────────

# A matched display pair whose normalised token streams are less similar
# than this is reported. Calibrated so that every formatting-only pair on the
# reference books scores above it.
BLOCK_SIMILARITY = 0.85
# Below this, two blocks are not considered "the same block" at all when
# aligning source and translation displays.
BLOCK_PAIRING = 0.5
# Findings per check per chunk, so one broken chunk does not bury the rest.
MAX_FINDINGS = 8
_EXCERPT = 100

# ── Math spans ──────────────────────────────────────────────────────────────

_DISPLAY_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "displaymath", "flalign")
_ENV_OPEN_RE = re.compile(r"\\begin\{(" + "|".join(_DISPLAY_ENVS) + r")(\*?)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")


@dataclass(frozen=True)
class MathSpan:
    display: bool
    body: str
    start: int  # offset of the span in the text
    end: int
    tag_after: str = ""  # "(12.3)" written right after the closing delimiter


def _blank_comments(text: str) -> str:
    """Comments replaced by spaces, so offsets (and line numbers) survive."""
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


def math_spans(text: str, *, latex: bool) -> list[MathSpan]:
    """Every math span in ``text``, in order.

    ``latex`` blanks ``%`` comments first (the translation); the Markdown
    source has none -- a ``%`` there is a percent sign.
    """
    if latex:
        text = _blank_comments(text)
    spans: list[MathSpan] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            nxt = text[index + 1]
            if nxt in "[(":
                close = "\\]" if nxt == "[" else "\\)"
                end = text.find(close, index + 2)
                if end == -1:
                    index += 2
                    continue
                spans.append(_span(text, nxt == "[", index + 2, end, end + 2, index))
                index = end + 2
                continue
            env = _ENV_OPEN_RE.match(text, index)
            if env:
                closing = f"\\end{{{env.group(1)}{env.group(2)}}}"
                end = text.find(closing, env.end())
                if end == -1:
                    index = env.end()
                    continue
                spans.append(_span(text, True, env.end(), end, end + len(closing), index))
                index = end + len(closing)
                continue
            index += 2  # an escaped character, \$ included
            continue
        if char == "$":
            if text.startswith("$$", index):
                end = text.find("$$", index + 2)
                if end == -1:
                    index += 2
                    continue
                spans.append(_span(text, True, index + 2, end, end + 2, index))
                index = end + 2
                continue
            end = _inline_close(text, index + 1)
            if end == -1:
                index += 1
                continue
            spans.append(_span(text, False, index + 1, end, end + 1, index))
            index = end + 1
            continue
        index += 1
    return spans


def _inline_close(text: str, start: int) -> int:
    """The closing ``$`` of an inline span, on the same paragraph."""
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "$":
            return index
        if char == "\n" and text.startswith("\n", index + 1):
            return -1
        index += 1
    return -1


# "(12.3)" right after a display's closing delimiter -- on the same line, or
# alone on the next line (MinerU does both). Alone: a reference in prose,
# "由式(12.3)可知", is not a tag.
_TAG_AFTER_RE = re.compile(
    r"[ \t]*(?:\n[ \t]*){0,2}[(（]\s*(\d+(?:\s*[.\-．]\s*\d+)+\s*[a-z]?)\s*[)）][ \t]*(?=\n|$)"
)


# A tag alone on its line, far from any display: the parser lost the formula
# and kept its number ("(3.5.1)"). It is still the book's tag.
_ORPHAN_TAG_RE = re.compile(
    r"^[ \t]*[(（]\s*(\d+(?:\s*[.\-．]\s*\d+)+\s*[a-z]?)\s*[)）][ \t]*$", re.MULTILINE
)


def _all_tags(text: str, spans: list[MathSpan], *, latex: bool) -> Counter[str]:
    tags = Counter(
        t for s in spans for t in split_tags(s.body, s.tag_after, display=s.display)[1]
    )
    body = _blank_comments(text) if latex else text
    claimed = {
        _line(body, s.end + body[s.end :].find(s.tag_after)) for s in spans if s.tag_after
    }
    outside = list(body)
    for span in spans:
        # Blank the math but keep its newlines, so line numbers still agree.
        outside[span.start : span.end] = [
            c if c == "\n" else " " for c in body[span.start : span.end]
        ]
    prose = "".join(outside)
    for match in _ORPHAN_TAG_RE.finditer(prose):
        if _line(prose, match.start()) not in claimed:
            tags[normalise_tag(match.group(1))] += 1
    return tags


def _span_numbers(span: MathSpan) -> list[str]:
    return numbers(tokens(split_tags(span.body, span.tag_after, display=span.display)[0]))


def _span(text: str, display: bool, body_start: int, body_end: int, end: int, start: int) -> MathSpan:
    tag_after = ""
    if display:
        after = _TAG_AFTER_RE.match(text, end)
        if after:
            tag_after = after.group(1)
    return MathSpan(display, text[body_start:body_end], start, end, tag_after)


# ── Normalisation ───────────────────────────────────────────────────────────

_UNICODE = {
    "\u2212": "-", "\u2013": "-", "\uff0d": "-", "\u00d7": r"\times ", "\u00b7": r"\cdot ",
    "\u22c5": r"\cdot ", "\u2219": r"\cdot ", "\uff0b": "+", "\uff1d": "=", "\u2264": r"\leq ",
    "\u2265": r"\geq ", "\u2260": r"\neq ", "\u2248": r"\approx ", "\u00b1": r"\pm ",
    "\u221e": r"\infty ", "\u2192": r"\rightarrow ", "\u2032": "'", "\u2033": "''",
    "\uff08": "(", "\uff09": ")", "\uff0c": ",", "\uff0e": ".", "\u3001": ",",
}
_UNICODE_RE = re.compile("|".join(map(re.escape, _UNICODE)))

# Groups whose *content* is prose: translated on purpose, so dropped.
_TEXT_GROUPS = ("text", "mbox", "textrm", "textit", "textbf", "hbox", "textnormal", "textup")
# Commands that change only presentation; the argument (if any) is kept.
# Bold is here too: the parser often loses a vector's bold face and the model
# restores it, and neither changes a number, a sign or a structure.
_PRESENTATION = frozenset(
    """
    mathrm rm mathit it mathnormal mathup mathbf bf boldsymbol bm pmb
    displaystyle textstyle scriptstyle
    scriptscriptstyle left right middle big Big bigg Bigg bigl bigr Bigl Bigr
    biggl biggr bigm Bigm quad qquad enspace thinspace medspace thickspace
    negthinspace limits nolimits nonumber notag allowbreak mathstrut strut
    """.split()
)
# Dropped along with their braced argument.
_DROP_WITH_ARGUMENT = frozenset("label hspace vspace hspace* phantom hphantom vphantom".split())
# Alignment wrappers inside a display: structure, not mathematics. `array`
# is one more -- the model adds or removes it around the same lines -- and
# its column spec goes with it.
_WRAPPER_ENVS = frozenset(
    "aligned alignedat split gathered align align* equation equation* gather gather* "
    "multline multline* eqnarray eqnarray* array".split()
)
# Function names written out in a font command: \mathrm{cos} is \cos.
_FUNCTIONS = frozenset(
    "sin cos tan cot sec csc ln log lg exp max min lim det arg sinh cosh tanh".split()
)
_FONT_WORD_RE = re.compile(
    r"\\(?:mathrm|rm|mathit|mathbf|bf|operatorname|mathsf|mathnormal)\s*"
    r"\{\s*((?:[A-Za-z ~]|\\ )+?)\s*\}"
)
# Footnote markers in math (^{①}, \textcircled{1}): the translation is told
# to delete them, so neither side counts them.
_FOOTNOTE_RE = re.compile(
    r"\^\s*\{?\s*(?:[\u2460-\u2473]|\\textcircled\s*\{[^{}]*\})\s*\}?"
    r"|[\u2460-\u2473]|\\textcircled\s*\{[^{}]*\}"
)
# A word inside math is prose: the source writes P_{损}, the translation
# P_{\mathrm{loss}}. Three or more letters, lowercase after an optional
# capital, so "MHz", "Hz", "nm" and single symbols stay mathematics.
_WORD_RE = re.compile(r"^[A-Za-z][a-z]{2,}(?: [A-Za-z]?[a-z]+)*$")
_ALIASES = {
    r"\le": r"\leq", r"\leqslant": r"\leq", r"\ge": r"\geq", r"\geqslant": r"\geq",
    r"\ne": r"\neq", r"\to": r"\rightarrow", r"\gets": r"\leftarrow", r"\dfrac": r"\frac",
    r"\tfrac": r"\frac", r"\cfrac": r"\frac", r"\lbrace": r"\{", r"\rbrace": r"\}",
    r"\ldots": r"\dots", r"\cdots": r"\dots", r"\dotsc": r"\dots", r"\dotsb": r"\dots",
    r"\boldsymbol": r"\mathbf", r"\bm": r"\mathbf", r"\pmb": r"\mathbf", r"\bf": r"\mathbf",
    r"\vert": "|", r"\lvert": "|", r"\rvert": "|", r"\Vert": r"\|", r"\lVert": r"\|",
    r"\rVert": r"\|", r"\langle": "<", r"\rangle": ">", r"\lt": "<", r"\gt": ">",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\colon": ":",
    r"\varnothing": r"\emptyset", r"\dbinom": r"\binom", r"\tbinom": r"\binom",
    r"\bar": r"\overline", r"\longrightarrow": r"\rightarrow", r"\Longrightarrow": r"\Rightarrow",
    r"\leftrightarrow": r"\leftrightarrow", r"\widehat": r"\hat", r"\widetilde": r"\tilde",
}
_SPACING = frozenset({r"\,", r"\;", r"\:", r"\!", r"\ ", "~", "&", r"\\", r"\>"})

_TAG_CMD_RE = re.compile(r"\\tag\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_CIRCLED_RE = re.compile(r"\\textcircled\s*\{\s*(?:\\[a-z]+size\s*)?(\d+)\s*\}")
_TRAILING_TAG_RE = re.compile(
    r"(?:\\q?quad|\\[,;:!]|~|\s)*[(（]\s*(\d+(?:\s*[.\-．]\s*\d+)+\s*[a-z]?)\s*[)）]\s*(?:\\\\)?\s*$"
)
_DIGIT_SPACE_RE = re.compile(r"(?<=\d)\s+(?=\d)")
# Thin spaces and braced commas inside a number: 1\,000, 1~000, 1{,}000.
_DIGIT_GROUP_RE = re.compile(r"(?<=\d)(?:\\[,!: ]|~|\{,\})(?=\d)")
_SEPARATOR = "\x00"
_DECIMAL_SPACE_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_TOKEN_RE = re.compile(r"\\[A-Za-z]+\*?|\\.|\d+(?:\.\d+)?|[A-Za-z]|\S")
_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")

OPERATORS = frozenset(
    {"+", "-", "=", "<", ">", r"\pm", r"\mp", r"\leq", r"\geq", r"\neq", r"\approx",
     r"\equiv", r"\sim", r"\simeq", r"\propto", r"\ll", r"\gg"}
)


def normalise_tag(tag: str) -> str:
    """``12 . 3`` -> ``12.3``; ``\textcircled{1}`` -> ``①``."""
    tag = _CIRCLED_RE.sub(lambda m: chr(0x245F + int(m.group(1))) if 0 < int(m.group(1)) <= 20 else m.group(1), tag)
    return re.sub(r"\s+", "", tag).replace("．", ".")


def split_tags(
    body: str, tag_after: str = "", *, display: bool = True
) -> tuple[str, list[str]]:
    """``body`` without its equation tags, and the tags it carried.

    Only a display can end in a written-out tag: inline ``$(12.3)$`` is a
    reference to an equation, not a tag.
    """
    tags = [normalise_tag(t) for t in _TAG_CMD_RE.findall(body)]
    body = _TAG_CMD_RE.sub(" ", body)
    trailing = _TRAILING_TAG_RE.search(body) if display else None
    if trailing:
        tags.append(normalise_tag(trailing.group(1)))
        body = body[: trailing.start()]
    if tag_after:
        tags.append(normalise_tag(tag_after))
    return body, tags


def _font_word(match: re.Match[str]) -> str:
    word = re.sub(r"(?:\\ |~)+", " ", match.group(1)).strip()
    if word in _FUNCTIONS:
        return f"\\{word} "
    if _WORD_RE.match(word):
        return _SEPARATOR
    return match.group(0)


def _drop_groups(body: str, names: tuple[str, ...] | frozenset[str]) -> str:
    """Remove ``\\name{...}`` groups whole, brace-matched."""
    pattern = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z])\s*\{")
    out: list[str] = []
    cursor = 0
    for match in pattern.finditer(body):
        if match.start() < cursor:
            continue
        depth, index = 1, match.end()
        while index < len(body) and depth:
            if body[index] == "\\":
                index += 2
                continue
            depth += {"{": 1, "}": -1}.get(body[index], 0)
            index += 1
        out.append(body[cursor : match.start()])
        out.append(_SEPARATOR)
        cursor = index
    out.append(body[cursor:])
    return "".join(out)


def tokens(body: str) -> list[str]:
    """The canonical token stream of one math body (tags already removed)."""
    body = _UNICODE_RE.sub(lambda m: _UNICODE[m.group(0)], body)
    body = _FONT_WORD_RE.sub(_font_word, body)
    body = _FOOTNOTE_RE.sub(_SEPARATOR, body)
    body = _drop_groups(body, _TEXT_GROUPS)
    body = _drop_groups(body, _DROP_WITH_ARGUMENT)
    body = _CJK_RE.sub(_SEPARATOR, body)
    body = _DIGIT_GROUP_RE.sub("", body)
    body = _DECIMAL_SPACE_RE.sub(".", _DIGIT_SPACE_RE.sub("", body))

    raw = _TOKEN_RE.findall(body)
    out: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token in (r"\begin", r"\end") and index + 1 < len(raw) and raw[index + 1] == "{":
            close = raw.index("}", index) if "}" in raw[index:] else -1
            name = "".join(raw[index + 2 : close]) if close != -1 else ""
            if close != -1 and name in _WRAPPER_ENVS:
                index = close + 1
                if token == r"\begin" and name == "array" and index < len(raw) and raw[index] == "{":
                    spec_close = raw.index("}", index) if "}" in raw[index:] else index
                    index = spec_close + 1  # the column spec
                continue
        name = token[1:] if token.startswith("\\") else ""
        if token in _SPACING or name in _PRESENTATION or token == _SEPARATOR:
            index += 1
            continue
        out.append(_ALIASES.get(token, token))
        index += 1
    # Sentence punctuation closing a display is prose, not mathematics.
    while out and out[-1] in (".", ",", ";", ":"):
        out.pop()
    return _collapse(out)


def _drop_empty_parens(seq: list[str]) -> list[str]:
    """``( )`` and ``( , )`` left behind once the CJK inside was dropped."""
    out: list[str] = []
    for token in seq:
        out.append(token)
        if token == ")":
            back = len(out) - 2
            while back >= 0 and out[back] in (",", ";"):
                back -= 1
            if back >= 0 and out[back] == "(":
                del out[back:]
    return out


def _collapse(seq: list[str]) -> list[str]:
    """Remove braces around single tokens and empty groups, and unify primes,
    until nothing changes: ``{x}`` -> ``x``, ``^{\\prime}`` -> ``'``."""
    seq = _drop_empty_parens(["'" if t == r"\prime" else t for t in seq])
    changed = True
    while changed:
        changed = False
        out: list[str] = []
        index = 0
        while index < len(seq):
            if seq[index] == "{" and index + 1 < len(seq) and seq[index + 1] == "}":
                index += 2
                changed = True
                continue
            if (
                seq[index] == "{"
                and index + 2 < len(seq)
                and seq[index + 2] == "}"
                and seq[index + 1] not in "{}"
            ):
                out.append(seq[index + 1])
                index += 3
                changed = True
                continue
            if seq[index] == "{" and index + 1 < len(seq) and seq[index + 1] == "{":
                inner = _matching(seq, index + 1)
                if inner is not None and inner + 1 < len(seq) and seq[inner + 1] == "}":
                    out.extend(seq[index + 1 : inner + 1])  # {{x y}} -> {x y}
                    index = inner + 2
                    changed = True
                    continue
            if seq[index] == "^" and index + 1 < len(seq) and seq[index + 1] == "'":
                index += 1  # ^' -> '
                changed = True
                continue
            if seq[index] == "^" and index + 1 < len(seq) and seq[index + 1] == "{":
                close = index + 2
                while close < len(seq) and seq[close] == "'":
                    close += 1
                if close > index + 2 and close < len(seq) and seq[close] == "}":
                    out.extend(seq[index + 2 : close])  # ^{''} -> ''
                    index = close + 1
                    changed = True
                    continue
            out.append(seq[index])
            index += 1
        seq = out
    return seq


def _matching(seq: list[str], open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(seq)):
        if seq[index] == "{":
            depth += 1
        elif seq[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def numbers(seq: list[str]) -> list[str]:
    """Numeric literals, with an attached sign and a position marker:
    ``10^{-20}`` gives ``10`` and ``^-20``; ``u_{02}`` gives ``_02``.

    ASCII digits only -- ``str.isdigit`` would also accept ``①``, the
    footnote markers the translation is told to delete.
    """
    found: list[str] = []
    for index, token in enumerate(seq):
        if token[0] not in "0123456789":
            continue
        before = index - 1
        sign = ""
        if before >= 0 and seq[before] in ("-", "+"):
            sign = seq[before]
            before -= 1
        while before >= 0 and seq[before] == "{":
            before -= 1
        marker = seq[before] if before >= 0 and seq[before] in ("^", "_") else ""
        found.append(f"{marker}{sign}{token}")
    return found


# ── Blocks ──────────────────────────────────────────────────────────────────


@dataclass
class Block:
    span: MathSpan
    tokens: list[str]
    tags: list[str]

    @property
    def key(self) -> str:
        return " ".join(self.tokens)


def _blocks(text: str, *, latex: bool) -> tuple[list[Block], list[MathSpan]]:
    spans = math_spans(text, latex=latex)
    blocks: list[Block] = []
    for span in spans:
        body, tags = split_tags(span.body, span.tag_after, display=span.display)
        if span.display:
            blocks.append(Block(span, tokens(body), tags))
    return blocks, spans


def _similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _split_moved(
    blocks: list[Block], other_spans: list[MathSpan]
) -> tuple[list[Block], list[Block]]:
    """``(unexplained, moved)``: a block is moved when an inline span on the
    other side carries the same mathematics."""
    inline = [
        tokens(split_tags(s.body, display=False)[0]) for s in other_spans if not s.display
    ]
    unexplained: list[Block] = []
    moved: list[Block] = []
    for block in blocks:
        if any(_similarity(block.tokens, other) >= BLOCK_SIMILARITY for other in inline):
            moved.append(block)
        else:
            unexplained.append(block)
    return unexplained, moved


def align_blocks(
    source: list[Block], target: list[Block]
) -> tuple[list[tuple[Block, Block]], list[Block], list[Block]]:
    """Pair source and translation displays in order.

    Identical blocks anchor the alignment; between anchors, blocks are paired
    in order when similar enough to be "the same block, possibly changed".
    Returns ``(pairs, unmatched_source, unmatched_target)``.
    """
    matcher = SequenceMatcher(None, [b.key for b in source], [b.key for b in target], autojunk=False)
    pairs: list[tuple[Block, Block]] = []
    lost: list[Block] = []
    extra: list[Block] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            pairs.extend(zip(source[i1:i2], target[j1:j2]))
            continue
        left, right = source[i1:i2], target[j1:j2]
        if len(left) == len(right):
            pairs.extend(zip(left, right))
            continue
        cursor = 0
        for block in left:
            best, best_score = None, BLOCK_PAIRING
            for k in range(cursor, len(right)):
                score = _similarity(block.tokens, right[k].tokens)
                if score > best_score:
                    best, best_score = k, score
            if best is None:
                lost.append(block)
                continue
            extra.extend(right[cursor:best])
            pairs.append((block, right[best]))
            cursor = best + 1
        extra.extend(right[cursor:])
    return pairs, lost, extra


# ── The checks ──────────────────────────────────────────────────────────────


def _excerpt(text: str, limit: int = _EXCERPT) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


_HEADING_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s|\\(?:part|chapter|section|subsection|subsubsection)\*?\s*\{)"
)
_CAPTION_LINE_RE = re.compile(r"^\s*(?:图|表|例|习题|思考题)\s*[A-Z]?\s*\d")
_PROSE_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w])")
# Looser, for "did this number move out of the math": any digit run with its
# sign, wherever it sits ("R2", "-60mA", "LiNbO3").
_LOOSE_NUMBER_RE = re.compile(r"[-+]?(?<![\d.])\d+(?:\.\d+)?(?![\d])")
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Superscripts in prose are exponents: "10⁻³" carries the exponent "^-3".
_SUPERSCRIPT_RE = re.compile(r"([⁺⁻]?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_PROSE_EXPONENT_RE = re.compile(
    r"(?:\^\s*\{?|<sup>)\s*([-+\u2212]?)\s*(\d+)", re.IGNORECASE
)
# "10 000", "10,000", "10\u2009000": one number in prose, as in math.
_PROSE_THOUSANDS_RE = re.compile(r"(?<=\d)[ ,\u2009\u202f](?=\d{3}(?!\d))")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|IMG_\d{4}|<[^>]+>")
_TEX_SKIP_RE = re.compile(r"\\(?:bookfig|bookfigtwo|includegraphics|label|ref|eqref|tag)\b[^\n]*?\}")


def _prose_numbers(
    text: str, spans: list[MathSpan], *, latex: bool, loose: bool = False
) -> Counter[str]:
    pieces: list[str] = []
    cursor = 0
    body = _blank_comments(text) if latex else text
    for span in spans:
        pieces.append(body[cursor : span.start])
        cursor = span.end
    pieces.append(body[cursor:])
    prose = " ".join(pieces)
    found: Counter[str] = Counter()
    if loose:
        prose = _UNICODE_RE.sub(lambda m: _UNICODE[m.group(0)], prose).translate(_SUBSCRIPT_DIGITS)
        prose = _PROSE_THOUSANDS_RE.sub("", prose)
        for sign, digits in _SUPERSCRIPT_RE.findall(prose):
            found["^" + ("-" if sign == "⁻" else "") + digits.translate(_SUPERSCRIPT_DIGITS)] += 1
        for sign, digits in _PROSE_EXPONENT_RE.findall(prose):
            found["^" + ("-" if sign in "-\u2212" and sign else "") + digits] += 1
        prose = _IMAGE_RE.sub(" ", prose)
        found.update(_LOOSE_NUMBER_RE.findall(_SUPERSCRIPT_RE.sub(" ", prose)))
        return found
    for line in prose.split("\n"):
        if _HEADING_LINE_RE.match(line) or _CAPTION_LINE_RE.match(line):
            continue
        line = _IMAGE_RE.sub(" ", line)
        if latex:
            line = _TEX_SKIP_RE.sub(" ", line)
        found.update(_PROSE_NUMBER_RE.findall(line))
    return found


def check_pair(source: str, translation: str, *, file: str = "") -> list[Finding]:
    """Every fidelity finding for one source chunk and its translation.

    ``file`` labels the findings (``translated/012.tex``); their line numbers
    are lines of the translation.
    """
    findings: list[Finding] = []
    src_blocks, src_spans = _blocks(source, latex=False)
    tgt_blocks, tgt_spans = _blocks(translation, latex=True)

    def where(block: Block | None) -> int:
        return _line(translation, block.span.start) if block else 0

    # Tags: every tag, display or inline, as a multiset.
    src_tags = _all_tags(source, src_spans, latex=False)
    tgt_tags = _all_tags(translation, tgt_spans, latex=True)
    missing_tags = sorted((src_tags - tgt_tags).elements())
    extra_tags = sorted((tgt_tags - src_tags).elements())
    if missing_tags or extra_tags:
        parts = []
        if missing_tags:
            parts.append(f"missing {', '.join(missing_tags[:MAX_FINDINGS])}")
        if extra_tags:
            parts.append(f"not in the source {', '.join(extra_tags[:MAX_FINDINGS])}")
        findings.append(
            Finding(
                "fidelity-tags",
                Severity.REVIEW,
                f"equation tags differ from the source: {'; '.join(parts)}",
                file=file,
            )
        )

    pairs, lost, extra = align_blocks(src_blocks, tgt_blocks)
    # A display that is inline math on the other side was moved, not lost or
    # invented: the model promotes "$F = qE$" to a display, or the reverse,
    # and the mathematics is the same. On five real books this explained
    # most display-count differences.
    lost, moved_out = _split_moved(lost, tgt_spans)
    extra, moved_in = _split_moved(extra, src_spans)
    if lost or extra:
        detail = [f"source has {len(src_blocks)} display block(s), translation {len(tgt_blocks)}"]
        detail += [f"lost: {_excerpt(b.span.body, 60)}" for b in lost[:3]]
        detail += [f"added: {_excerpt(b.span.body, 60)}" for b in extra[:3]]
        findings.append(
            Finding(
                "fidelity-display-count",
                Severity.REVIEW,
                f"{detail[0]}; {len(lost)} lost and {len(extra)} added with no inline "
                "counterpart",
                where(extra[0]) if extra else 0,
                "; ".join(detail[1:]),
                file=file,
            )
        )
    elif moved_in or moved_out:
        findings.append(
            Finding(
                "fidelity-display-count",
                Severity.INFO,
                f"{len(moved_in)} inline formula(s) became displays and {len(moved_out)} "
                "display(s) became inline; the mathematics matches",
                where(moved_in[0]) if moved_in else 0,
                file=file,
            )
        )

    # Numbers across all the math of the chunk. A number that moved between
    # prose and math is not a change: the source often writes "I=-60 mA" or
    # "3.4×10⁻³" as plain text and the translation, correctly, as math. Only
    # the *exact* value -- sign and exponent marker included -- gets that
    # pass, so a changed exponent or a flipped sign is still reported.
    src_numbers = Counter(n for s in src_spans for n in _span_numbers(s))
    tgt_numbers = Counter(n for s in tgt_spans for n in _span_numbers(s))
    src_prose = _prose_numbers(source, src_spans, latex=False, loose=True)
    tgt_prose = _prose_numbers(translation, tgt_spans, latex=True, loose=True)

    def moved(value: str, prose: Counter[str]) -> bool:
        return value in prose or value.lstrip("+") in prose

    missing = sorted(v for v in set(src_numbers - tgt_numbers) if not moved(v, tgt_prose))
    invented = sorted(v for v in set(tgt_numbers - src_numbers) if not moved(v, src_prose))
    # A value gone from one side entirely is the drift signature (10^{-20}
    # became 10^{-2}): review. A value present on both sides a different
    # number of times is usually a formula rewritten or split, not a changed
    # value -- measured on five real books, nearly all of these were "0",
    # "2" or "^2" one count apart -- so those are reported, but as info.
    counted = [
        v for v in missing if tgt_numbers[v]
    ] + [v for v in invented if src_numbers[v]]
    missing = [v for v in missing if not tgt_numbers[v]]
    invented = [v for v in invented if not src_numbers[v]]
    if counted:
        findings.append(
            Finding(
                "fidelity-numbers",
                Severity.INFO,
                f"{len(counted)} number(s) occur a different number of times in the source "
                "math and the translation's (present on both sides)",
                context=", ".join(
                    f"{v} {src_numbers[v]}->{tgt_numbers[v]}" for v in sorted(counted)[:20]
                ),
                file=file,
            )
        )
    for value, sign, other in (
        *((v, "missing", "source") for v in missing),
        *((v, "invented", "translation") for v in invented),
    ):
        if sum(1 for f in findings if f.check == "fidelity-numbers" and f.severity is Severity.REVIEW) >= MAX_FINDINGS:
            break
        home = src_spans if other == "source" else tgt_spans
        span = next((s for s in home if value in _span_numbers(s)), None)
        findings.append(
            Finding(
                "fidelity-numbers",
                Severity.REVIEW,
                f"{value} is {sign}: {src_numbers[value]}x in the source math, "
                f"{tgt_numbers[value]}x in the translation's",
                _line(translation, span.start) if span and other == "translation" else 0,
                f"{other}: {_excerpt(span.body)}" if span else "",
                file=file,
            )
        )

    # Matched display blocks.
    reported = 0
    for src, tgt in pairs:
        if reported >= MAX_FINDINGS or src.tokens == tgt.tokens:
            continue
        score = _similarity(src.tokens, tgt.tokens)
        src_ops = Counter(t for t in src.tokens if t in OPERATORS)
        tgt_ops = Counter(t for t in tgt.tokens if t in OPERATORS)
        problems = []
        if score < BLOCK_SIMILARITY:
            problems.append(f"similarity {score:.2f} < {BLOCK_SIMILARITY}")
        if src_ops != tgt_ops:
            diff = (src_ops - tgt_ops) + (tgt_ops - src_ops)
            problems.append(f"operators differ ({' '.join(sorted(diff.elements()))})")
        if not problems:
            continue
        reported += 1
        findings.append(
            Finding(
                "fidelity-blocks",
                Severity.REVIEW,
                f"a display block changed: {'; '.join(problems)}",
                where(tgt),
                f"source: {_excerpt(src.span.body)} | translation: {_excerpt(tgt.span.body)}",
                file=file,
            )
        )

    missing_prose = sorted(
        set(_prose_numbers(source, src_spans, latex=False))
        - set(_prose_numbers(translation, tgt_spans, latex=True)),
        key=lambda n: (len(n), n),
    )
    if missing_prose:
        findings.append(
            Finding(
                "fidelity-prose-numbers",
                Severity.INFO,
                f"{len(missing_prose)} number(s) in the source prose are not in the "
                "translation's prose (often legitimate: Chinese numerals, rewritten units)",
                context=", ".join(missing_prose[:20]),
                file=file,
            )
        )
    return findings


__all__ = [
    "BLOCK_PAIRING",
    "BLOCK_SIMILARITY",
    "MathSpan",
    "align_blocks",
    "check_pair",
    "math_spans",
    "numbers",
    "split_tags",
    "tokens",
]
