"""
Book structure: which top-level headings are parts, chapters and readings.

Why this exists
---------------
The translating model decides every heading's LaTeX command on its own, one
chunk at a time, and on a real book it decided wrong in two ways nothing
flagged until the PDF: a part heading ("第篇 光学" -- the parser dropped the
number) became ``\\chapter{Optics}``, and five supplementary readings became
numbered chapters. Each extra chapter pushed every later one up by one, so
the PDF printed "Chapter 24" over the book's chapter 22.

The profiler's heading classification could not have caught it: it sees
heading *text* one string at a time, and the parser also dropped "第N章" from
most chapter titles, which leaves "静电场" (chapter 12) and "大气电学" (a
reading) looking exactly alike. What tells them apart is under the heading --
a chapter's sections and figures carry its number (12.1, 图12.3), a
reading's are lettered (G.1, 图G.3) -- and how that compares with its
siblings.

Who decides: DeepSeek, checked by code
--------------------------------------
Every book has its own conventions, so no fixed rule may decide. The steps:

1. **Evidence** (code, book-neutral): for each top-level heading, the number
   prefixes of its sub-headings, of the caption line after each image
   (whatever the caption word is), and of its equation tags.
2. **One DeepSeek call labels the whole outline** -- every top-level heading
   in one table with its evidence -- so it can compare siblings: 25 rows with
   sections numbered by chapter and 5 with lettered ones are two kinds.
3. **Code cross-checks the labels** against what holds for any book: chapter
   numbers run on consecutively, a chapter's number matches the numbering
   under it, a part has chapters after it. A second opinion comes from the
   numbering rules this corpus follows (第…篇 is a part, numbered evidence is
   a chapter, lettered evidence a reading); they never decide, but a
   disagreement is a conflict. Each conflicting row gets one follow-up call,
   with the whole outline as context and the conflict spelled out. Whatever
   still conflicts is flagged in the printed outline.
4. **Rewrite** (code): the headings become final LaTeX -- ``\\part``,
   ``\\chapter``, ``\\chapter*`` -- so the translating model only translates
   the title. Sections inside an unnumbered chapter become starred (a
   numbered ``\\section`` there would be numbered from the chapter before
   it), and numbered items under a chapter summary, which the parser turned
   into headings, go back to being paragraphs.

Decisions are cached in ``<work-dir>/structure.json``, which is meant to be
opened and corrected: set an entry's ``"by"`` to ``"manual"`` and its
``"kind"`` (and ``"number"``) win on the next run.

With no LLM, or when the outline call fails, the numbering rules are the
fallback and the outline says so. Nothing here raises.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.config import SOURCE_LANG
from src.llm.base import BaseLLM
from src.profiler import extract_json_object

logger = logging.getLogger(__name__)

STRUCTURE_FILENAME = "structure.json"

PART, CHAPTER, READING, OTHER = "part", "chapter", "reading", "other"
KINDS: tuple[str, ...] = (PART, CHAPTER, READING, OTHER)
UNNUMBERED: frozenset[str] = frozenset({READING, OTHER})

# Same shape as `src.profiler._HEADING_LINE_RE`; both read the same lines.
_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)[ \t]*$")
_IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")

# A number the book gives a section, figure or equation: "12.1", "G.3",
# "G. 3" (the parser inserts spaces), "§3.1".
_PREFIX = r"(\d+|[A-Z])\s*[.．]\s*\d+"
_SECTION_NUMBER_RE = re.compile(rf"^(?:\\?\*\s*|§\s*)?{_PREFIX}")
# A caption line: up to eight non-digit characters of caption word (图, 表,
# Figure, Fig., Table ...) and then the number.
_CAPTION_NUMBER_RE = re.compile(rf"^\s*[^\d\s]{{0,8}}\s*{_PREFIX}")
_TAG_RE = re.compile(rf"\\tag\s*\{{\s*{_PREFIX}")

# The second opinion's title conventions, from this corpus. Hints only.
_PART_TITLE_RE = re.compile(r"^第\s*[0-9一二三四五六七八九十]*\s*(?:篇|编|部分)\s*")
_CHAPTER_TITLE_RE = re.compile(r"^第\s*(\d+)\s*章\s*")
# Labels to strip from a title when the model's clean title is unusable.
_TITLE_LABEL_RE = re.compile(
    r"^(?:第\s*[0-9一二三四五六七八九十百零〇两]*\s*(?:章|篇|编|部分)"
    r"|(?:chapter|part)\s+[0-9IVXLC]+[.:]?)\s*",
    re.IGNORECASE,
)

# The leading number of a heading inside an unnumbered chapter, or of a
# summary item: "G.1 ", "M. 1 ", "5 ", "1. ", "\* 5. ". A lone capital must
# be followed by a number, so "BCS 理论" keeps its "BCS".
_LEADING_NUMBER_RE = re.compile(
    r"^(?:\\?\*\s*|§\s*)?(?:[A-Z]\s*[.．]\s*\d+|\d+(?:\s*[.．]\s*\d+)*)\s*[.．、]?\s*"
)
# "4. 电容器的电容", "2.夫琅禾费衍射", "\* 5. ..." -- but not "12.1 ...", which is
# a section number.
_SUMMARY_ITEM_RE = re.compile(r"^(?:\\?\*\s*)?\d+\s*[.．、](?!\s*\d)")

# End-of-chapter summary headings. Everything numbered underneath one is a
# list item, whatever heading level the parser gave it.
SUMMARY_TITLES: frozenset[str] = frozenset(
    {"提要", "小结", "本章小结", "内容提要", "本章提要", "总结", "Summary", "Chapter Summary"}
)

# Evidence counts as decisive when one prefix accounts for this share of it.
_DOMINANCE = 0.6
_EXCERPT_CHARS = 160
_SUBHEADINGS_SHOWN = 6
# Follow-up calls are one per conflicting row; this bounds a bad run.
_MAX_FOLLOW_UPS = 10


@dataclass
class Division:
    """One top-level heading and what is known about it."""

    line: int  # 1-based line of the heading in the analysed text
    title: str
    evidence: Counter[str] = field(default_factory=Counter)
    subheadings: list[str] = field(default_factory=list)
    excerpt: str = ""
    kind: str | None = None  # part | chapter | reading | other; None = undecided
    number: int | None = None  # chapter number, when known
    clean_title: str = ""  # the title without its 第12章 / Part II label
    decided_by: str = ""  # llm | follow-up | manual | rules
    reason: str = ""
    rule_kind: str | None = None  # the numbering rules' opinion
    rule_number: int | None = None
    rule_reason: str = ""
    rule_from_title: bool = False  # printed in the heading, not read from OCR'd text
    flag: str = ""  # a conflict that survived the follow-up

    @property
    def dominant_prefix(self) -> str | None:
        total = sum(self.evidence.values())
        if not total:
            return None
        prefix, count = self.evidence.most_common(1)[0]
        return prefix if count / total >= _DOMINANCE else None

    @property
    def latex_title(self) -> str:
        if self.clean_title:
            return self.clean_title
        stripped = _TITLE_LABEL_RE.sub("", self.title, count=1).strip()
        return stripped or self.title


@dataclass
class StructureReport:
    divisions: list[Division]
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0

    def counts(self) -> Counter[str]:
        return Counter(d.kind or "undecided" for d in self.divisions)


# ── 1. Evidence ─────────────────────────────────────────────────────────────


def find_divisions(text: str) -> list[Division]:
    """Every top-level (``#``) heading, with the evidence under it."""
    lines = text.split("\n")
    starts = [
        (index, match.group(2))
        for index, line in enumerate(lines)
        if (match := _HEADING_LINE_RE.match(line)) and len(match.group(1)) == 1
    ]

    divisions: list[Division] = []
    for position, (index, title) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        span = lines[index + 1 : end]
        subheadings = [m.group(2) for line in span if (m := _HEADING_LINE_RE.match(line))]
        prose = " ".join(
            line.strip()
            for line in span
            if line.strip() and not line.startswith("#") and not _IMAGE_LINE_RE.match(line)
        )
        divisions.append(
            Division(
                line=index + 1,
                title=title,
                evidence=_evidence(span),
                subheadings=subheadings,
                excerpt=prose[:_EXCERPT_CHARS],
            )
        )
    return divisions


def _evidence(span: list[str]) -> Counter[str]:
    evidence: Counter[str] = Counter()
    after_image = False
    for line in span:
        heading = _HEADING_LINE_RE.match(line)
        if heading:
            number = _SECTION_NUMBER_RE.match(heading.group(2))
            if number:
                evidence[number.group(1)] += 1
        if _IMAGE_LINE_RE.match(line):
            after_image = True
            continue
        if after_image and line.strip():
            caption = _CAPTION_NUMBER_RE.match(line)
            if caption:
                evidence[caption.group(1)] += 1
            after_image = False
    evidence.update(match.group(1) for match in _TAG_RE.finditer("\n".join(span)))
    return evidence


def _describe(evidence: Counter[str]) -> str:
    if not evidence:
        return "none"
    return ", ".join(f"{prefix}.x x{count}" for prefix, count in evidence.most_common(4))


# ── The second opinion ──────────────────────────────────────────────────────


def rule_opinion(division: Division) -> None:
    """What this corpus's numbering conventions say. Never decides alone."""
    title, dominant = division.title, division.dominant_prefix

    if _PART_TITLE_RE.match(title):
        division.rule_kind, division.rule_reason = PART, "第…篇 in the title"
        return
    titled = _CHAPTER_TITLE_RE.match(title)
    if titled:
        division.rule_kind, division.rule_number = CHAPTER, int(titled.group(1))
        division.rule_reason = f"第{titled.group(1)}章 in the title"
        division.rule_from_title = True
        return
    if dominant is None:
        return
    if dominant.isdigit():
        division.rule_kind, division.rule_number = CHAPTER, int(dominant)
        division.rule_reason = f"numbered {dominant}.x"
    else:
        division.rule_kind, division.rule_reason = READING, f"lettered {dominant}.x"


# ── 2. One call labels the whole outline ────────────────────────────────────


_KIND_DEFINITIONS = (
    "  part    -- a division that groups several chapters. Usually a short "
    "introduction with no numbered sections of its own.\n"
    "  chapter -- one of the book's main, numbered chapters.\n"
    "  reading -- supplementary material between chapters that is not numbered "
    "as a chapter (a reading, feature or special topic).\n"
    "  other   -- front or back matter: preface, contents, appendix, answers, "
    "index, references.\n"
)


def _outline_prompt(source_lang: str) -> str:
    return (
        f"You are checking the top-level structure of a {source_lang} textbook "
        "that was converted from PDF. Below is every top-level heading in "
        "document order, one row each, with evidence collected from the text "
        "under it.\n\n"
        f"Label every row with one kind:\n{_KIND_DEFINITIONS}\n"
        "Every book has its own conventions, and the converter often drops the "
        "chapter or part label from a title, so do not judge a row by its title "
        "alone. Compare the rows with each other: the main chapters share one "
        "pattern of evidence (for example sections numbered 12.1, 12.2 and "
        "captions numbered 12.x), and rows that break that pattern -- lettered "
        "numbering, no numbered sections, a different kind of content -- are "
        "usually parts, readings or front/back matter.\n\n"
        "For each chapter give its number as the book numbers it, from its "
        "evidence or its title. For every row give \"title\": the heading with "
        "any label such as 第12章, 第三篇, Chapter 12 or Part II removed, and "
        "otherwise copied exactly.\n\n"
        "Return ONLY a JSON object, no prose and no code fences:\n"
        '{"rows": [{"row": 1, "kind": "part" | "chapter" | "reading" | "other", '
        '"number": 12 or null, "title": "...", "reason": "<a few words>"}]}\n'
        "Include every row exactly once."
    )


def _row(index: int, division: Division, *, labelled: bool = False) -> str:
    shown = "; ".join(division.subheadings[:_SUBHEADINGS_SHOWN])
    more = len(division.subheadings) - _SUBHEADINGS_SHOWN
    label = ""
    if labelled:
        label = f"  [labelled: {division.kind or 'undecided'}"
        label += f" {division.number}]" if division.kind == CHAPTER and division.number else "]"
    return (
        f"row {index}: {division.title}{label}\n"
        f"  sub-headings ({len(division.subheadings)}): {shown or '(none)'}"
        f"{f' ... +{more}' if more > 0 else ''}\n"
        f"  numbers under it: {_describe(division.evidence)}\n"
        f"  excerpt: {division.excerpt or '(none)'}"
    )


def _usable_title(answer: object, division: Division) -> str:
    """The model's clean title, if it only removed a label from the original."""
    title = str(answer or "").strip()
    return title if title and title in division.title else ""


def _apply_answer(division: Division, answer: dict, decided_by: str) -> bool:
    kind = str(answer.get("kind", "")).strip().lower()
    if kind not in KINDS:
        return False
    number = answer.get("number")
    division.kind = kind
    division.number = number if kind == CHAPTER and isinstance(number, int) else None
    division.clean_title = _usable_title(answer.get("title"), division) or division.clean_title
    division.reason = str(answer.get("reason", "")).strip()
    division.decided_by = decided_by
    return True


def _label_outline(
    divisions: list[Division], llm: BaseLLM, source_lang: str
) -> bool:
    """Label every row in one call. Returns False if the call is unusable."""
    table = "\n".join(_row(i, d) for i, d in enumerate(divisions, start=1))
    try:
        raw = llm.generate(_outline_prompt(source_lang), table, temperature=0.0)
        rows = extract_json_object(raw).get("rows")
    except Exception as exc:  # noqa: BLE001 - structure checks must never be fatal
        logger.warning("Outline labelling failed (%s)", exc)
        return False
    if not isinstance(rows, list):
        return False

    labelled = 0
    for answer in rows:
        if not isinstance(answer, dict):
            continue
        index = answer.get("row")
        if isinstance(index, int) and 1 <= index <= len(divisions):
            division = divisions[index - 1]
            if division.decided_by != "manual" and _apply_answer(division, answer, "llm"):
                labelled += 1
    return labelled > 0


# ── 3. Cross-checks and follow-ups ──────────────────────────────────────────


def sequence_problems(divisions: list[Division]) -> list[str]:
    """Places where chapter numbers do not run on consecutively.

    A chapter with no number of its own takes the next one in sequence, so it
    can never be a problem by itself -- but it does move the expectation for
    the chapter after it.
    """
    return [problem for _, problem in _sequence_breaks(divisions)]


def _sequence_breaks(divisions: list[Division]) -> list[tuple[int, str]]:
    breaks: list[tuple[int, str]] = []
    expected: int | None = None
    previous: Division | None = None
    for index, division in enumerate(divisions):
        if division.kind != CHAPTER:
            continue
        if division.number is None:
            expected = expected + 1 if expected is not None else None
        else:
            if expected is not None and division.number != expected:
                before = f"after '{previous.title}' " if previous else ""
                breaks.append(
                    (
                        index,
                        f"'{division.title}' is chapter {division.number}, but {before}the "
                        f"next chapter should be {expected}",
                    )
                )
            expected = division.number + 1
        previous = division
    return breaks


def _expected_number(divisions: list[Division], index: int) -> int | None:
    """The chapter number row ``index`` would have to carry to fit between
    its labelled neighbours, or None if no neighbour is numbered."""
    for earlier in range(index - 1, -1, -1):
        previous = divisions[earlier]
        if previous.kind == CHAPTER and previous.number is not None:
            return previous.number + 1
    for later in range(index + 1, len(divisions)):
        following = divisions[later]
        if following.kind == CHAPTER and following.number is not None:
            return following.number - 1
    return None


def _credible_rule(divisions: list[Division], index: int) -> bool:
    """Whether the rules' "chapter N" could be true where the row sits.

    OCR reads a reading's lettered I.1, I.2 as 1.1, 1.2, and the rules then
    say "chapter 1" -- between chapters 20 and 21. On a real book that
    second opinion pushed a correct "reading" label into a wrong "chapter 13"
    on follow-up. An opinion the sequence rules out is noise, not a conflict.
    """
    division = divisions[index]
    if division.rule_kind != CHAPTER or division.rule_number is None or division.rule_from_title:
        return True
    expected = _expected_number(divisions, index)
    return expected is None or division.rule_number == expected


def conflicts(divisions: list[Division]) -> dict[int, str]:
    """``{row index: what is wrong}`` for every label the checks dispute."""
    found: dict[int, str] = {}

    def add(index: int, problem: str) -> None:
        found[index] = f"{found[index]}; {problem}" if index in found else problem

    for index, problem in _sequence_breaks(divisions):
        add(index, problem)
        # The break may be the previous chapter's fault, e.g. a reading
        # labelled as a chapter just before it.
        for earlier in range(index - 1, -1, -1):
            if divisions[earlier].kind == CHAPTER:
                add(earlier, f"the chapter after it breaks the sequence: {problem}")
                break

    for index, division in enumerate(divisions):
        if division.kind is None:
            continue
        credible = _credible_rule(divisions, index)
        if credible and division.rule_kind and division.rule_kind != division.kind:
            add(
                index,
                f"labelled {division.kind}, but the numbering says {division.rule_kind} "
                f"({division.rule_reason})",
            )
        elif (
            credible
            and division.kind == CHAPTER
            and division.rule_number is not None
            and division.number is not None
            and division.rule_number != division.number
        ):
            add(
                index,
                f"labelled chapter {division.number}, but {division.rule_reason}",
            )
        if division.kind == PART:
            following = divisions[index + 1] if index + 1 < len(divisions) else None
            if following is None or following.kind == PART:
                add(index, "a part with no chapter after it")

    return {i: p for i, p in found.items() if divisions[i].decided_by != "manual"}


def _follow_up_prompt(source_lang: str) -> str:
    return (
        f"You labelled the top-level headings of a {source_lang} textbook "
        f"converted from PDF, with these kinds:\n{_KIND_DEFINITIONS}\n"
        "One row conflicts with the consistency checks. Look at that row again, "
        "in the context of its siblings in the outline, and answer for that row "
        "only. If your label was right, keep it and say why the check is "
        "misleading.\n\n"
        "Return ONLY a JSON object, no prose and no code fences:\n"
        '{"kind": "part" | "chapter" | "reading" | "other", "number": 12 or null, '
        '"reason": "<one sentence>"}'
    )


_LABEL_FIELDS = ("kind", "number", "clean_title", "decided_by", "reason")


def _follow_up(
    divisions: list[Division], index: int, problem: str, llm: BaseLLM, source_lang: str
) -> None:
    """Re-ask about one row. The new answer is kept only if it helps.

    "Helps" means the row's own conflict is gone and no other row gained
    one. Otherwise the first label stands and the row is flagged: a follow-up
    that trades one conflict for another is a guess, and the first answer was
    made with the whole outline in view. Fewer conflicts overall is not
    enough -- relabelling 第13章 as "other" removes a chapter, and with it
    the sequence break, while contradicting its own title.
    """
    division = divisions[index]
    outline = "\n".join(_row(i, d, labelled=True) for i, d in enumerate(divisions, start=1))
    question = f"Outline:\n{outline}\n\nThe row to re-check: row {index + 1}.\nConflict: {problem}"
    try:
        answer = extract_json_object(
            llm.generate(_follow_up_prompt(source_lang), question, temperature=0.0)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up for %r failed (%s)", division.title, exc)
        return

    before = conflicts(divisions)
    snapshot = {name: getattr(division, name) for name in _LABEL_FIELDS}
    if not _apply_answer(division, answer, "follow-up"):
        return
    after = conflicts(divisions)
    if index in after or not set(after) <= set(before):
        for name, value in snapshot.items():
            setattr(division, name, value)


# ── Orchestration ───────────────────────────────────────────────────────────


def _load_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return {}
    headings = data.get("headings", {}) if isinstance(data, dict) else {}
    return {k: v for k, v in headings.items() if isinstance(v, dict)}


def _save_cache(path: Path, divisions: list[Division]) -> None:
    payload = {
        "headings": {
            d.title: {
                "kind": d.kind,
                "number": d.number,
                "title": d.latex_title,
                "by": d.decided_by or "undecided",
                "reason": d.reason,
                **({"flag": d.flag} if d.flag else {}),
            }
            for d in divisions
        }
    }
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError as exc:
        logger.warning("Could not save %s: %s", path, exc)


def _restore(division: Division, entry: dict) -> None:
    division.kind = entry["kind"]
    number = entry.get("number")
    division.number = number if division.kind == CHAPTER and isinstance(number, int) else None
    division.clean_title = _usable_title(entry.get("title"), division)
    division.decided_by = str(entry.get("by", ""))
    division.reason = str(entry.get("reason", ""))


def _fall_back_to_rules(divisions: list[Division]) -> None:
    """Fill what the model left open: an unlabelled row takes the rules'
    opinion, and a chapter with no number takes the one under it."""
    for division in divisions:
        if division.kind == CHAPTER and division.number is None:
            if division.rule_kind == CHAPTER:
                division.number = division.rule_number
            continue
        if division.decided_by == "manual" or division.kind is not None:
            continue
        if division.rule_kind:
            division.kind, division.number = division.rule_kind, division.rule_number
            division.decided_by, division.reason = "rules", division.rule_reason


def analyze_structure(
    text: str,
    *,
    llm: BaseLLM | None = None,
    cache_path: Path | None = None,
    force: bool = False,
    first_chapter: int | None = None,
    source_lang: str = SOURCE_LANG,
) -> StructureReport:
    """Decide every top-level heading's kind. Never raises.

    Args:
        text: The book's Markdown, with heading levels already corrected.
        llm: Labels the outline. ``None`` falls back to the numbering rules,
            unchecked, and says so.
        cache_path: ``structure.json``. Manual entries always win; earlier
            DeepSeek labels are reused, without a call, unless ``force`` or a
            heading is new.
        first_chapter: The chapter number the preamble starts at, to warn
            when the book starts somewhere else.
    """
    divisions = find_divisions(text)
    report = StructureReport(divisions)
    cache = _load_cache(cache_path)

    for division in divisions:
        rule_opinion(division)
        entry = cache.get(division.title, {})
        if entry.get("by") == "manual" and entry.get("kind") in KINDS:
            _restore(division, entry)

    open_rows = [d for d in divisions if d.decided_by != "manual"]
    reusable = not force and open_rows and all(
        cache.get(d.title, {}).get("by") in ("llm", "follow-up")
        and cache[d.title].get("kind") in KINDS
        for d in open_rows
    )

    if reusable:
        for division in open_rows:
            _restore(division, cache[division.title])
    elif llm is not None and open_rows:
        report.llm_calls += 1
        if not _label_outline(divisions, llm, source_lang):
            report.warnings.append(
                "DeepSeek could not label the outline; fell back to the numbering "
                "rules, unchecked -- read the outline carefully"
            )
        else:
            for index, problem in list(conflicts(divisions).items())[:_MAX_FOLLOW_UPS]:
                report.llm_calls += 1
                _follow_up(divisions, index, problem, llm, source_lang)
    elif open_rows:
        report.warnings.append(
            "no LLM: parts, chapters and readings were decided by the numbering "
            "rules alone -- read the outline carefully"
        )

    _fall_back_to_rules(divisions)

    for index, problem in conflicts(divisions).items():
        divisions[index].flag = problem
        report.warnings.append(f"check '{divisions[index].title}': {problem}")
    for division in divisions:
        if division.kind is None:
            report.warnings.append(
                f"'{division.title}' (line {division.line}) is undecided; it stays a "
                "Markdown heading. Set its kind in structure.json to decide it."
            )

    numbered = [d.number for d in divisions if d.kind == CHAPTER and d.number is not None]
    if first_chapter is not None and numbered and numbered[0] != first_chapter:
        report.warnings.append(
            f"the book's first chapter is {numbered[0]} but the preamble starts at "
            f"{first_chapter}: set \\setcounter{{chapter}}{{{numbered[0] - 1}}} in "
            "assets/preamble.tex"
        )

    if cache_path is not None:
        _save_cache(cache_path, divisions)
    return report


# ── 4. Rewrite ──────────────────────────────────────────────────────────────


def _strip_number(title: str) -> str:
    stripped = _LEADING_NUMBER_RE.sub("", title, count=1).strip()
    return stripped or title


def apply_structure(text: str, report: StructureReport) -> tuple[str, Counter[str]]:
    """Rewrite the decided headings as LaTeX. Pure; returns ``(text, counts)``."""
    by_line = {d.line: d for d in report.divisions}
    lines = text.split("\n")
    counts: Counter[str] = Counter()
    unnumbered = False
    summary_level: int | None = None

    for index, line in enumerate(lines):
        match = _HEADING_LINE_RE.match(line)
        if not match:
            continue
        level, title = len(match.group(1)), match.group(2)

        if level == 1:
            summary_level = None
            division = by_line.get(index + 1)
            if division is None or division.title != title or division.kind is None:
                unnumbered = False
                continue
            unnumbered = division.kind in UNNUMBERED
            command = {PART: "part", CHAPTER: "chapter"}.get(division.kind, "chapter*")
            lines[index] = f"\\{command}{{{division.latex_title}}}"
            counts[division.kind] += 1
            continue

        if summary_level is not None:
            # The parser puts summary items anywhere from the summary's own
            # level downwards, so a numbered heading at that level is still
            # an item; anything else at or above it ends the summary.
            if level >= summary_level and _SUMMARY_ITEM_RE.match(title):
                # "### \* 5. 电容器的充放电" -> "*5. 电容器的充放电"
                lines[index] = re.sub(r"^\\\*\s*", "*", title)
                counts["summary item"] += 1
                continue
            if level <= summary_level:
                summary_level = None
        if title.replace("\\*", "").strip() in SUMMARY_TITLES:
            summary_level = level
            continue

        if unnumbered and level in (2, 3):
            command = "section" if level == 2 else "subsection"
            lines[index] = f"\\{command}*{{{_strip_number(title)}}}"
            counts["unnumbered section"] += 1

    return "\n".join(lines), counts


def render_outline(report: StructureReport) -> str:
    """The outline, one line per top-level heading, for a ten-second check."""
    rows: list[str] = []
    for division in report.divisions:
        if division.kind == PART:
            label, indent = "PART", ""
        elif division.kind == CHAPTER:
            label, indent = f"{division.number or '?':>4}", "  "
        elif division.kind == READING:
            label, indent = "   *", "  "
        elif division.kind == OTHER:
            label, indent = "   -", "  "
        else:
            label, indent = "   ?", "  "

        if division.rule_kind is None or division.decided_by in ("rules", "manual"):
            check = ""
        elif division.rule_kind == division.kind:
            check = " (rules agree)"
        else:
            check = f" (rules said {division.rule_kind})"
        by = f"[{division.decided_by}{check}] " if division.decided_by else ""
        flag = f"  <-- CHECK: {division.flag}" if division.flag else ""
        rows.append(f"  {indent}{label}  {division.title:<28} {by}{division.reason}{flag}")
    return "\n".join(rows)


__all__ = [
    "CHAPTER",
    "KINDS",
    "OTHER",
    "PART",
    "READING",
    "STRUCTURE_FILENAME",
    "SUMMARY_TITLES",
    "Division",
    "StructureReport",
    "analyze_structure",
    "apply_structure",
    "conflicts",
    "find_divisions",
    "render_outline",
    "rule_opinion",
    "sequence_problems",
]
