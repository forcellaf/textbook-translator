"""
Meaning verification: a second, stronger model reads every translated chunk.

Why this module exists
----------------------
``src.lint`` proves the translation compiles and ``src.fidelity`` proves its
*mathematics* still matches the source. Neither reads the prose: a dropped
sentence, a lost "not", a mistranslated term or a derivation step summarised
away all pass every deterministic check. The translator is the cheapest model
in the pipeline, so a stronger one reads its work.

Two agents, two files per chunk
-------------------------------
``chunk-verifier`` (one per chunk) compares ``chunks/NNN.md`` with
``translated/NNN.tex``, **fixes meaning problems in place**, and writes what
it found and changed to ``kit/verify/NNN.json``:

.. code-block:: json

    {"chunk": "007",
     "findings": [
       {"id": 1, "kind": "omission", "severity": "major", "action": "fixed",
        "source_excerpt": "exact text from chunks/007.md",
        "translation_excerpt": "the translation's text before the fix",
        "fixed_text": "the translation's text after the fix",
        "explanation": "one sentence"}]}

It never edits mathematics. A major problem with an equation -- the
translation and the source disagree, or the source itself looks garbled --
gets ``"action": "page-check"``: only the printed page can say which side is
right, because the source chunk is the parser's reading of it.

``page-verifier`` (one per chunk with page checks) looks at the rendered page
images, decides, fixes in place, and writes ``kit/verify/NNN.pages.json``:

.. code-block:: json

    {"chunk": "007",
     "checks": [
       {"finding": 3, "verdict": "parse-error", "action": "fixed",
        "page_reads": "what the page shows, in LaTeX",
        "source_fix": {"old": "...", "new": "..."},
        "translation_fix": {"old": "...", "new": "..."},
        "explanation": "one sentence"}]}

A parse error is fixed in the source chunk *and* the translation, so the
fidelity lint keeps comparing two texts that both match the book.

Log only
--------
Nothing here gates the build. `verification_report` collects both files per
chunk into the summary the orchestrator copies into the run log: every fix
under Decisions, every finding left unfixed under Uncertainties, every
confirmed parse error in the spot-check list.

Checking the checker
--------------------
An LLM reviewer can invent a problem or quote text that is not there. Every
excerpt is therefore looked up in the file it claims to come from (after
removing whitespace and ``$``, the way the parser and the model differ), and
a fix's new text must be in the file now. A miss is reported as a flag on
the finding, never silently dropped: the run log says which findings could
not be confirmed.

Finding the page
----------------
MinerU's ``content_list.json`` gives every parsed block a ``page_idx``. The
source excerpt (or the paragraph around it, when the excerpt is short enough
to occur on many pages) is matched against those blocks, preferring pages
inside the chunk's own span; if nothing matches, the page is interpolated
from the excerpt's position in the chunk. `locate_pages` reports which of
these it used, so a guessed page is visible as one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from src.figures import load_content_list
from src.kit import CHUNKS_DIRNAME, TRANSLATED_DIRNAME
from src.page_range import (
    MANIFEST_PATH,
    SOURCE_BODY_NAME,
    original_page,
    page_offset,
    was_trimmed,
)

VERIFY_DIRNAME = "verify"
PAGES_DIRNAME = "pages"  # kit/verify/pages/pNNNN.png
PAGE_CHECKS_NAME = "page_checks.json"

KINDS: tuple[str, ...] = (
    "omission",  # source content missing from the translation
    "addition",  # translation says something the source does not
    "mistranslation",  # wrong meaning
    "negation",  # a "not" lost or added
    "terminology",  # a technical term rendered wrongly or inconsistently
    "untranslated",  # Chinese left as prose, or a passage left in the source language
    "formula",  # an equation differs between source and translation
    "source-suspect",  # the source itself looks garbled (a parse error?)
)
# Mathematics is never edited by the chunk verifier: these kinds are either
# sent to a page check (major) or left in the log (minor).
FORMULA_KINDS: frozenset[str] = frozenset({"formula", "source-suspect"})
SEVERITIES: tuple[str, ...] = ("major", "minor")
ACTIONS: tuple[str, ...] = ("fixed", "page-check", "left")

PAGE_VERDICTS: tuple[str, ...] = (
    "translation-error",  # page == source, translation wrong: fix the translation
    "parse-error",  # page != source: fix the source chunk (and the translation)
    "no-error",  # page, source and translation agree: a false alarm
    "unresolved",  # unreadable, or the passage is not on the rendered pages
)
PAGE_ACTIONS: tuple[str, ...] = ("fixed", "left")

PAGE_DPI = 200  # sub- and superscripts must be legible
# Below this similarity a content-list block is not taken as the excerpt's.
LOCATE_MIN_SCORE = 0.6
# An excerpt shorter than this (compacted) is matched by its paragraph
# instead: "F=ma" occurs on many pages, its paragraph on one.
LOCATE_SHORT_EXCERPT = 40
# Lines of the chunk sampled to find the chunk's own page span.
_SPAN_SAMPLE_LINES = 40
_SPAN_MIN_LINE = 15


# ── Text matching ───────────────────────────────────────────────────────────

_SPACE_RE = re.compile(r"\s+")


def compact(text: str) -> str:
    """Whitespace and ``$`` removed: the parser spaces characters out and the
    two sides delimit math differently, but the characters are the same."""
    return _SPACE_RE.sub("", text).replace("$", "")


# A JSON writer that forgot to double a backslash turns `\frac` into a form
# feed + "rac" and `\nabla` into a newline + "abla". Both still parse.
_UNESCAPED = {"\x0c": "\\f", "\x08": "\\b", "\r": "\\r", "\t": "\\t"}
_LOST_NEWLINE_COMMAND_RE = re.compile(r"\n(?=[a-zA-Z])")


def _repair_escapes(text: str) -> str:
    for char, command in _UNESCAPED.items():
        text = text.replace(char, command)
    return _LOST_NEWLINE_COMMAND_RE.sub(r"\\n", text)


def _variants(text: str) -> list[str]:
    repaired = _repair_escapes(text)
    return [text] if repaired == text else [text, repaired]


def find_line(haystack: str, needle: str) -> int | None:
    """1-based line of ``haystack`` where ``needle`` starts, comparing both
    compacted; ``None`` when it is not there (or ``needle`` is empty)."""
    flat_chars: list[str] = []
    origin: list[int] = []  # flat index -> haystack index
    for index, char in enumerate(haystack):
        if char.isspace() or char == "$":
            continue
        flat_chars.append(char)
        origin.append(index)
    flat = "".join(flat_chars)
    for variant in _variants(needle):
        target = compact(variant)
        if not target:
            continue
        at = flat.find(target)
        if at >= 0:
            return haystack.count("\n", 0, origin[at]) + 1
    return None


def contains(haystack: str, needle: str) -> bool:
    return find_line(haystack, needle) is not None


# ── Reading the agents' files ───────────────────────────────────────────────


@dataclass
class VerifyFinding:
    """One problem the chunk verifier reported."""

    chunk: str
    id: int
    kind: str
    severity: str
    action: str
    source_excerpt: str
    translation_excerpt: str
    explanation: str
    fixed_text: str = ""
    line: int | None = None  # translated/NNN.tex, where the fix or excerpt is now
    flags: list[str] = field(default_factory=list)

    @property
    def location(self) -> str:
        suffix = f":{self.line}" if self.line else ""
        return f"{TRANSLATED_DIRNAME}/{self.chunk}.tex{suffix}"


@dataclass
class PageCheck:
    """The page verifier's decision on one page-check finding."""

    chunk: str
    finding: int
    verdict: str
    action: str
    page_reads: str
    explanation: str
    pages: list[int] = field(default_factory=list)  # original physical pages
    source_fix: dict | None = None
    translation_fix: dict | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class ChunkVerification:
    chunk: str
    findings: list[VerifyFinding] = field(default_factory=list)
    page_checks: list[PageCheck] = field(default_factory=list)
    # The file is unusable or a finding breaks the rules: re-ask the agent.
    problems: list[str] = field(default_factory=list)

    def pending_page_checks(self) -> list[VerifyFinding]:
        done = {check.finding for check in self.page_checks}
        return [f for f in self.findings if f.action == "page-check" and f.id not in done]


def verify_dir(kit_dir: Path) -> Path:
    return kit_dir / VERIFY_DIRNAME


def result_path(kit_dir: Path, chunk: str) -> Path:
    return verify_dir(kit_dir) / f"{chunk}.json"


def pages_result_path(kit_dir: Path, chunk: str) -> Path:
    return verify_dir(kit_dir) / f"{chunk}.pages.json"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_json(path: Path, problems: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        problems.append(f"{path.name}: unreadable ({exc})")
        return None
    except json.JSONDecodeError as exc:
        problems.append(
            f"{path.name}: not valid JSON ({exc.msg}, line {exc.lineno}); "
            "is every backslash doubled?"
        )
        return None
    if not isinstance(data, dict):
        problems.append(f"{path.name}: expected a JSON object")
        return None
    return data


def _text(entry: dict, key: str) -> str:
    value = entry.get(key, "")
    return value if isinstance(value, str) else ""


def _parse_finding(chunk: str, index: int, entry: object, problems: list[str]) -> VerifyFinding | None:
    if not isinstance(entry, dict):
        problems.append(f"finding #{index + 1}: not an object")
        return None
    raw_id = entry.get("id", index + 1)
    finding = VerifyFinding(
        chunk=chunk,
        id=raw_id if isinstance(raw_id, int) else index + 1,
        kind=_text(entry, "kind"),
        severity=_text(entry, "severity"),
        action=_text(entry, "action"),
        source_excerpt=_text(entry, "source_excerpt"),
        translation_excerpt=_text(entry, "translation_excerpt"),
        explanation=_text(entry, "explanation").strip(),
        fixed_text=_text(entry, "fixed_text"),
    )
    label = f"finding {finding.id}"
    for name, value, allowed in (
        ("kind", finding.kind, KINDS),
        ("severity", finding.severity, SEVERITIES),
        ("action", finding.action, ACTIONS),
    ):
        if value not in allowed:
            problems.append(f"{label}: {name} {value!r} is not one of {', '.join(allowed)}")
            return None
    if not finding.explanation:
        problems.append(f"{label}: no explanation")
    if finding.action == "fixed" and not finding.fixed_text.strip():
        problems.append(f"{label}: action 'fixed' without fixed_text")
    if finding.action == "fixed" and finding.kind in FORMULA_KINDS:
        problems.append(f"{label}: a {finding.kind} finding was fixed; mathematics goes to a page check")
    if finding.action == "page-check" and finding.severity != "major":
        problems.append(f"{label}: only a major finding gets a page check")
    return finding


def _check_excerpts(finding: VerifyFinding, source: str, translation: str) -> None:
    """Flag (never drop) what cannot be found where the finding says."""
    if finding.source_excerpt.strip() and not contains(source, finding.source_excerpt):
        finding.flags.append("source excerpt not found in the source chunk")
    if finding.action == "fixed":
        finding.line = find_line(translation, finding.fixed_text)
        if finding.line is None:
            finding.flags.append("fixed text not found in the translation (changed since?)")
    elif finding.translation_excerpt.strip():
        finding.line = find_line(translation, finding.translation_excerpt)
        if finding.line is None:
            finding.flags.append("translation excerpt not found in the translation")


def _parse_fix(value: object, label: str, problems: list[str]) -> dict | None:
    if value in (None, {}, ""):
        return None
    if not (isinstance(value, dict) and isinstance(value.get("new"), str)):
        problems.append(f"{label}: expected {{\"old\": ..., \"new\": ...}}")
        return None
    return {"old": str(value.get("old", "")), "new": value["new"]}


def _parse_page_check(
    chunk: str,
    entry: object,
    by_id: dict[int, VerifyFinding],
    source: str,
    translation: str,
    problems: list[str],
) -> PageCheck | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("finding"), int):
        problems.append("page check: expected an object with an integer 'finding'")
        return None
    label = f"page check of finding {entry['finding']}"
    check = PageCheck(
        chunk=chunk,
        finding=entry["finding"],
        verdict=_text(entry, "verdict"),
        action=_text(entry, "action") or "left",
        page_reads=_text(entry, "page_reads"),
        explanation=_text(entry, "explanation").strip(),
        pages=[p for p in entry.get("pages", []) if isinstance(p, int)]
        if isinstance(entry.get("pages"), list)
        else [],
        source_fix=_parse_fix(entry.get("source_fix"), label, problems),
        translation_fix=_parse_fix(entry.get("translation_fix"), label, problems),
    )
    if check.verdict not in PAGE_VERDICTS:
        problems.append(f"{label}: verdict {check.verdict!r} is not one of {', '.join(PAGE_VERDICTS)}")
        return None
    if check.action not in PAGE_ACTIONS:
        problems.append(f"{label}: action {check.action!r} is not one of {', '.join(PAGE_ACTIONS)}")
    target = by_id.get(check.finding)
    if target is None or target.action != "page-check":
        problems.append(f"{label}: no page-check finding with that id")
    if not check.explanation:
        problems.append(f"{label}: no explanation")
    if check.action == "fixed":
        if check.verdict == "translation-error" and check.translation_fix is None:
            problems.append(f"{label}: translation-error fixed without a translation_fix")
        if check.verdict == "parse-error" and check.source_fix is None:
            problems.append(f"{label}: parse-error fixed without a source_fix")
        if check.verdict in ("no-error", "unresolved"):
            problems.append(f"{label}: verdict {check.verdict!r} cannot be 'fixed'")
    if check.source_fix and not contains(source, check.source_fix["new"]):
        check.flags.append("source fix's new text not found in the source chunk")
    if check.translation_fix and not contains(translation, check.translation_fix["new"]):
        check.flags.append("translation fix's new text not found in the translation")
    return check


def load_verification(kit_dir: Path, chunk: str) -> ChunkVerification | None:
    """Both agent files for one chunk, validated; ``None`` if not verified."""
    path = result_path(kit_dir, chunk)
    if not path.exists():
        return None
    result = ChunkVerification(chunk=chunk)
    source = _read(kit_dir / CHUNKS_DIRNAME / f"{chunk}.md")
    translation = _read(kit_dir / TRANSLATED_DIRNAME / f"{chunk}.tex")

    data = _load_json(path, result.problems)
    if data is None:
        return result
    raw = data.get("findings", [])
    if not isinstance(raw, list):
        result.problems.append(f"{path.name}: 'findings' must be a list")
        raw = []
    for index, entry in enumerate(raw):
        finding = _parse_finding(chunk, index, entry, result.problems)
        if finding is not None:
            _check_excerpts(finding, source, translation)
            result.findings.append(finding)
    ids = [f.id for f in result.findings]
    if len(ids) != len(set(ids)):
        result.problems.append(f"{path.name}: finding ids are not unique")

    pages_path = pages_result_path(kit_dir, chunk)
    if pages_path.exists():
        pages = _load_json(pages_path, result.problems)
        checks = pages.get("checks", []) if pages else []
        by_id = {f.id: f for f in result.findings}
        for entry in checks if isinstance(checks, list) else []:
            check = _parse_page_check(chunk, entry, by_id, source, translation, result.problems)
            if check is not None:
                result.page_checks.append(check)
    return result


def translated_chunks(kit_dir: Path) -> list[str]:
    return sorted(p.stem for p in (kit_dir / TRANSLATED_DIRNAME).glob("*.tex") if p.stem.isdigit())


# ── Locating a source excerpt on the page ───────────────────────────────────


@dataclass
class Located:
    """Where an excerpt is in the parsed PDF: 1-based body pages."""

    pages: list[int]
    method: str  # "exact" | "fuzzy" | "interpolated" | "none"
    score: float = 0.0


def _entry_text(entry: dict) -> str:
    parts = [entry.get(key) for key in ("text", "table_body")]
    for key in ("img_caption", "table_caption", "image_caption"):
        value = entry.get(key)
        if isinstance(value, list):
            parts.extend(value)
    return " ".join(p for p in parts if isinstance(p, str))


def _paragraph_around(source: str, excerpt: str) -> str:
    """The blank-line paragraph of ``source`` holding ``excerpt``, else ``excerpt``."""
    line = find_line(source, excerpt)
    if line is None:
        return excerpt
    lines = source.split("\n")
    start = end = line - 1
    while start > 0 and lines[start - 1].strip():
        start -= 1
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    return "\n".join(lines[start : end + 1])


def _chunk_span(source: str, blocks: list[tuple[int, str]]) -> tuple[int, int] | None:
    """The body pages the chunk's own lines are found on (10th-90th
    percentile, so a line repeated elsewhere in the book cannot stretch it)."""
    lines = [compact(line) for line in source.split("\n")]
    lines = [line for line in lines if len(line) >= _SPAN_MIN_LINE and "IMG_" not in line]
    if not lines:
        return None
    step = max(1, len(lines) // _SPAN_SAMPLE_LINES)
    pages: list[int] = []
    for line in lines[::step]:
        hits = [page for page, text in blocks if line in text]
        if len(hits) == 1:
            pages.append(hits[0])
    if not pages:
        return None
    pages.sort()
    low = pages[len(pages) // 10]
    high = pages[min(len(pages) - 1, (len(pages) * 9) // 10)]
    return low, high


def _score(target: str, text: str) -> float:
    if not target or not text:
        return 0.0
    if target in text:
        return 1.0
    matcher = SequenceMatcher(None, target, text, autojunk=False)
    if matcher.real_quick_ratio() < LOCATE_MIN_SCORE * 0.5:
        return 0.0
    match = matcher.find_longest_match(0, len(target), 0, len(text))
    # How much of the excerpt one contiguous run covers, refined by the ratio.
    return max(match.size / len(target), matcher.ratio())


def locate_pages(work_dir: Path, source: str, excerpt: str) -> Located:
    """The body page(s) of the parsed PDF that show ``excerpt``."""
    entries = load_content_list(work_dir)
    blocks = [
        (entry["page_idx"] + 1, compact(_entry_text(entry)))
        for entry in entries
        if isinstance(entry.get("page_idx"), int)
    ]
    blocks = [(page, text) for page, text in blocks if text]
    if not blocks:
        return Located([], "none")
    span = _chunk_span(source, blocks)

    key = excerpt
    if len(compact(excerpt)) < LOCATE_SHORT_EXCERPT:
        key = _paragraph_around(source, excerpt)
    targets = [compact(v) for v in _variants(key)]

    def in_span(page: int) -> bool:
        return span is None or span[0] - 1 <= page <= span[1] + 1

    def scan(candidates: list[tuple[int, str]]) -> dict[int, float]:
        best: dict[int, float] = {}
        for page, text in candidates:
            score = max(_score(target, text) for target in targets)
            if score >= LOCATE_MIN_SCORE and score > best.get(page, 0.0):
                best[page] = score
        return best

    # The chunk's own pages first; the whole book only if they have no match
    # (fuzzy matching every block of a book is slow, and less likely right).
    best = scan([b for b in blocks if in_span(b[0])]) if span else {}
    if not best:
        best = scan(blocks)
    if best:
        ranked = sorted(best.items(), key=lambda item: (not in_span(item[0]), -item[1], item[0]))
        page, score = ranked[0]
        # A passage that crosses a page break matches the end of one page and
        # the start of the next about equally: show both.
        pages = sorted(p for p, s in ranked[:2] if abs(p - page) == 1 and s >= score - 0.1) or []
        pages = sorted({page, *pages})
        return Located(pages, "exact" if score >= 1.0 else "fuzzy", round(score, 2))

    if span is not None:
        line = find_line(source, excerpt)
        total = max(1, source.count("\n") + 1)
        fraction = (line - 1) / total if line else 0.5
        guess = span[0] + round(fraction * (span[1] - span[0]))
        pages = [p for p in (guess - 1, guess, guess + 1) if span[0] - 1 <= p <= span[1] + 1 and p >= 1]
        return Located(pages, "interpolated")
    return Located([], "none")


def parsed_pdf(work_dir: Path) -> Path | None:
    """The PDF MinerU parsed, whose pages ``page_idx`` counts."""
    body = work_dir / SOURCE_BODY_NAME
    if was_trimmed(work_dir) and body.exists():
        return body
    try:
        manifest = json.loads((work_dir / MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    original = manifest.get("original_pdf")
    if original and Path(original).exists():
        return Path(original)
    return body if body.exists() else None


def render_body_pages(pdf: Path, body_pages: list[int], offset: int, out_dir: Path) -> list[Path]:
    """Render body pages, named by their *original* physical page number."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with pymupdf.open(pdf) as doc:
        for page in body_pages:
            if not 1 <= page <= doc.page_count:
                continue
            path = out_dir / f"p{original_page(page, offset):04d}.png"
            if not path.exists():
                doc[page - 1].get_pixmap(dpi=PAGE_DPI).save(path)
            written.append(path)
    return written


def prepare_page_checks(kit_dir: Path, work_dir: Path) -> dict:
    """Locate and render the pages of every pending page check.

    Writes ``kit/verify/page_checks.json`` and returns it: one task per
    finding, grouped by the chunk the page verifier is spawned for.
    """
    offset = page_offset(work_dir)
    pdf = parsed_pdf(work_dir)
    out_dir = verify_dir(kit_dir) / PAGES_DIRNAME
    tasks: list[dict] = []
    for chunk in translated_chunks(kit_dir):
        result = load_verification(kit_dir, chunk)
        if result is None:
            continue
        source = _read(kit_dir / CHUNKS_DIRNAME / f"{chunk}.md")
        for finding in result.pending_page_checks():
            excerpt = finding.source_excerpt or finding.translation_excerpt
            located = locate_pages(work_dir, source, excerpt)
            images = render_body_pages(pdf, located.pages, offset, out_dir) if pdf else []
            tasks.append(
                {
                    "chunk": chunk,
                    "finding": finding.id,
                    "pages": [original_page(p, offset) for p in located.pages],
                    "images": [str(p) for p in images],
                    "locate": located.method,
                    "score": located.score,
                    "note": "" if images else "no page image: " + (
                        "the passage was not located" if pdf else "no parsed PDF found"
                    ),
                }
            )
    by_chunk: dict[str, list[dict]] = {}
    for task in tasks:
        by_chunk.setdefault(task["chunk"], []).append(task)
    payload = {"tasks": tasks, "chunks": sorted(by_chunk), "pdf": str(pdf) if pdf else None}
    path = verify_dir(kit_dir) / PAGE_CHECKS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


# ── The report for the run log ──────────────────────────────────────────────


def _clip(text: str, limit: int = 160) -> str:
    flat = _SPACE_RE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def verification_report(kit_dir: Path) -> tuple[dict, str]:
    """``(summary, text)``: the JSON summary and the full report.

    The summary lists what the run log needs: chunks not yet verified, page
    checks still pending, every fix, every finding left, every page verdict,
    and the files that must be rewritten (``problems``).
    """
    chunks = translated_chunks(kit_dir)
    results = {c: r for c in chunks if (r := load_verification(kit_dir, c)) is not None}
    findings = [f for r in results.values() for f in r.findings]
    checks = [c for r in results.values() for c in r.page_checks]
    by_key = {(f.chunk, f.id): f for f in findings}

    counts: dict[str, dict[str, int]] = {"kind": {}, "severity": {}, "action": {}, "verdict": {}}
    for f in findings:
        for name in ("kind", "severity", "action"):
            value = getattr(f, name)
            counts[name][value] = counts[name].get(value, 0) + 1
    for c in checks:
        counts["verdict"][c.verdict] = counts["verdict"].get(c.verdict, 0) + 1

    def brief(f: VerifyFinding) -> dict:
        item = {
            "at": f.location,
            "kind": f.kind,
            "severity": f.severity,
            "explanation": _clip(f.explanation),
        }
        if f.action == "fixed":
            item["before"] = _clip(f.translation_excerpt)
            item["after"] = _clip(f.fixed_text)
        if f.flags:
            item["flags"] = f.flags
        return item

    def page_brief(c: PageCheck) -> dict:
        finding = by_key.get((c.chunk, c.finding))
        item = {
            "at": finding.location if finding else f"{TRANSLATED_DIRNAME}/{c.chunk}.tex",
            "finding": c.finding,
            "pages": c.pages,
            "verdict": c.verdict,
            "action": c.action,
            "page_reads": _clip(c.page_reads),
            "explanation": _clip(c.explanation),
        }
        for name in ("source_fix", "translation_fix"):
            fix = getattr(c, name)
            if fix:
                item[name] = {"old": _clip(fix["old"]), "new": _clip(fix["new"])}
        if c.flags:
            item["flags"] = c.flags
        return item

    summary = {
        "chunks": len(chunks),
        "verified": len(results),
        "unverified": [c for c in chunks if c not in results],
        "findings": len(findings),
        "counts": counts,
        "fixed": [brief(f) for f in findings if f.action == "fixed"],
        "left": [brief(f) for f in findings if f.action == "left"],
        "page_checks_pending": [
            {"chunk": r.chunk, "finding": f.id} for r in results.values() for f in r.pending_page_checks()
        ],
        "page_checks": [page_brief(c) for c in checks],
        "flagged": sum(1 for f in findings if f.flags) + sum(1 for c in checks if c.flags),
        "problems": {r.chunk: r.problems for r in results.values() if r.problems},
    }

    lines = [
        (
            f"Meaning verification: {len(results)} of {len(chunks)} chunk(s) verified, "
            f"{len(findings)} finding(s), {len(checks)} page check(s) decided."
        ),
    ]
    if summary["unverified"]:
        lines.append("Not verified: " + ", ".join(summary["unverified"]))
    for chunk, result in sorted(results.items()):
        if not (result.findings or result.problems):
            continue
        lines.append("")
        lines.append(f"== chunk {chunk} ==")
        for problem in result.problems:
            lines.append(f"  PROBLEM  {problem}")
        decided = {c.finding: c for c in result.page_checks}
        for f in result.findings:
            lines.append(f"  [{f.severity} {f.kind} -> {f.action}] {f.location}: {f.explanation}")
            if f.source_excerpt:
                lines.append(f"      source: {_clip(f.source_excerpt, 300)}")
            if f.translation_excerpt:
                lines.append(f"      before: {_clip(f.translation_excerpt, 300)}")
            if f.fixed_text:
                lines.append(f"      after:  {_clip(f.fixed_text, 300)}")
            for flag in f.flags:
                lines.append(f"      FLAG: {flag}")
            check = decided.get(f.id)
            if check is not None:
                pages = ", ".join(str(p) for p in check.pages) or "?"
                lines.append(f"      page check (p. {pages}): {check.verdict}, {check.action} -- {check.explanation}")
                if check.page_reads:
                    lines.append(f"      page reads: {_clip(check.page_reads, 300)}")
                for name in ("source_fix", "translation_fix"):
                    fix = getattr(check, name)
                    if fix:
                        lines.append(f"      {name}: {_clip(fix['old'], 200)}  =>  {_clip(fix['new'], 200)}")
                for flag in check.flags:
                    lines.append(f"      FLAG: {flag}")
            elif f.action == "page-check":
                lines.append("      page check: PENDING")
    return summary, "\n".join(lines) + "\n"


__all__ = [
    "ACTIONS",
    "FORMULA_KINDS",
    "KINDS",
    "PAGE_ACTIONS",
    "PAGE_CHECKS_NAME",
    "PAGE_VERDICTS",
    "SEVERITIES",
    "VERIFY_DIRNAME",
    "ChunkVerification",
    "Located",
    "PageCheck",
    "VerifyFinding",
    "compact",
    "contains",
    "find_line",
    "load_verification",
    "locate_pages",
    "parsed_pdf",
    "prepare_page_checks",
    "render_body_pages",
    "translated_chunks",
    "verification_report",
]
