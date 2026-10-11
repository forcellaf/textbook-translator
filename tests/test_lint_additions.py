"""Tests for the per-book lint additions an unattended run relies on.

Waivers (``kit/lint_waivers.json``), the per-book command allowlist
(``kit/known_commands.txt``), the ``% END-OF-CHUNK`` sentinel, findings
located in the fragment they came from, and ``"by": "agent"`` provenance in
structure.json, figures.json and book_profile.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.figures import FigureRecord, build_figures
from src.findings import Severity
from src.kit import assemble
from src.lint import SENTINEL, lint_kit
from src.profiler import profile_book
from src.structure import analyze_structure, render_outline

SOURCE = "## 12.1 电荷\n\n电荷守恒。这是一个很长的句子，用于说明电荷的性质。\n"
GOOD = "\\section{Electric Charge}\n\nCharge is conserved. This long sentence explains charge.\n"


def _kit(tmp_path: Path, translation: str, *, sentinel_prompt: bool = True) -> Path:
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir()
    (kit / "chunks" / "001.md").write_text(SOURCE, encoding="utf-8")
    (kit / "translated" / "001.tex").write_text(translation, encoding="utf-8")
    prompt = f"Translate.\nEnd your reply with:\n{SENTINEL}\n" if sentinel_prompt else "Translate."
    (kit / "system_prompt.txt").write_text(prompt, encoding="utf-8")
    return kit


def _review(kit: Path) -> list:
    return lint_kit(kit).of(Severity.REVIEW)


# ── Sentinel ────────────────────────────────────────────────────────────────


def test_a_missing_sentinel_is_a_review_finding(tmp_path: Path) -> None:
    review = _review(_kit(tmp_path, GOOD))
    assert [f.check for f in review] == ["truncated-sentinel"]
    assert review[0].file == "translated/001.tex"


def test_a_reply_ending_with_the_sentinel_passes(tmp_path: Path) -> None:
    assert _review(_kit(tmp_path, GOOD + SENTINEL + "\n")) == []


def test_a_fenced_reply_with_the_sentinel_after_the_fence_passes(tmp_path: Path) -> None:
    assert _review(_kit(tmp_path, f"```latex\n{GOOD}```\n{SENTINEL}\n")) == []


def test_kits_built_before_the_sentinel_are_not_held_to_it(tmp_path: Path) -> None:
    assert _review(_kit(tmp_path, GOOD, sentinel_prompt=False)) == []


def test_assembly_strips_the_sentinel(tmp_path: Path) -> None:
    kit = _kit(tmp_path, GOOD + SENTINEL + "\n")
    assembled, _problems = assemble(kit)
    assert SENTINEL not in assembled.read_text(encoding="utf-8")


# ── Locations ───────────────────────────────────────────────────────────────


def test_latex_findings_point_at_the_fragment_and_its_line(tmp_path: Path) -> None:
    broken = GOOD + "\n\\begin{solution}\nx\n\\end{solution}\n" + SENTINEL + "\n"
    kit = _kit(tmp_path, broken)
    (kit / "chunks" / "002.md").write_text(SOURCE, encoding="utf-8")
    (kit / "translated" / "002.tex").write_text(broken, encoding="utf-8")

    hits = [f for f in _review(kit) if f.check == "undefined-environment" or "solution" in f.message]
    assert {f.location for f in hits} == {"translated/001.tex:5", "translated/002.tex:5"}


# ── Per-book allowlist ──────────────────────────────────────────────────────


def test_known_commands_txt_extends_the_vocabulary(tmp_path: Path) -> None:
    body = GOOD + "Field: $E = \\varkappa H$.\n" + SENTINEL + "\n"
    kit = _kit(tmp_path, body)
    (kit / "chunks" / "001.md").write_text(SOURCE + "场 $E = \\varkappa H$ 。\n", encoding="utf-8")
    assert [f.check for f in _review(kit)] == ["unknown-command"]
    assert "known_commands.txt" in _review(kit)[0].message

    (kit / "known_commands.txt").write_text(
        "# amssymb's variant kappa; amssymb is loaded by the preamble\n\\varkappa\n",
        encoding="utf-8",
    )
    assert _review(kit) == []


# ── Waivers ─────────────────────────────────────────────────────────────────


def _waive(kit: Path, *waivers: dict) -> None:
    (kit / "lint_waivers.json").write_text(json.dumps(list(waivers)), encoding="utf-8")


def test_a_waiver_downgrades_the_finding_to_info_with_its_reason(tmp_path: Path) -> None:
    kit = _kit(tmp_path, GOOD)
    _waive(
        kit,
        {
            "check": "truncated-sentinel",
            "file": "translated/001.tex",
            "line_excerpt": "",
            "reason": "translated by hand before the sentinel existed",
            "by": "agent",
        },
    )
    report = lint_kit(kit)
    assert not report.needs_review
    (waived,) = [f for f in report.findings if f.check == "truncated-sentinel"]
    assert waived.severity is Severity.INFO
    assert "waived by agent: translated by hand" in waived.message


def test_a_waiver_needs_an_excerpt_that_matches(tmp_path: Path) -> None:
    kit = _kit(tmp_path, GOOD)
    _waive(
        kit,
        {"check": "truncated-sentinel", "file": "translated/001.tex",
         "line_excerpt": "something else entirely", "reason": "r", "by": "agent"},
    )
    report = lint_kit(kit)
    assert report.needs_review
    assert any(f.check == "waivers" and "matched nothing" in f.message for f in report.findings)


def test_a_waiver_without_a_reason_is_refused(tmp_path: Path) -> None:
    kit = _kit(tmp_path, GOOD)
    _waive(kit, {"check": "truncated-sentinel", "file": "translated/001.tex", "by": "agent"})
    checks = [f.check for f in _review(kit)]
    assert "truncated-sentinel" in checks and "waivers" in checks


def test_waivers_apply_to_fidelity_findings_too(tmp_path: Path) -> None:
    kit = _kit(tmp_path, GOOD + "\\[\nq = 10^{-2}\n\\]\n" + SENTINEL + "\n")
    (kit / "chunks" / "001.md").write_text(SOURCE + "\n$$\nq = 10^{-20}\n$$\n", encoding="utf-8")
    assert "fidelity-numbers" in {f.check for f in _review(kit)}

    _waive(kit, {"check": "fidelity-numbers", "file": "translated/001.tex",
                 "line_excerpt": "", "reason": "the source misprints the exponent; "
                 "checked against the page image", "by": "agent"})
    assert "fidelity-numbers" not in {f.check for f in _review(kit)}


# ── by: agent ───────────────────────────────────────────────────────────────

BOOK = "# 静电场\n\n## 12.1 电荷\n\n正文。\n\n# 大气电学\n\n## G.1 晴天\n\n正文。\n"


class _Never:
    def generate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("the LLM must not be asked about a decided row")


def test_an_agent_entry_in_structure_json_is_kept_and_shown(tmp_path: Path) -> None:
    cache = tmp_path / "structure.json"
    cache.write_text(
        json.dumps(
            {
                "headings": {
                    "静电场": {"kind": "chapter", "number": 12, "by": "agent", "reason": "12.x"},
                    "大气电学": {"kind": "reading", "by": "agent", "reason": "G.x"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = analyze_structure(BOOK, llm=_Never(), cache_path=cache, force=True)

    assert [(d.kind, d.decided_by) for d in report.divisions] == [
        ("chapter", "agent"), ("reading", "agent")
    ]
    assert "[agent]" in render_outline(report)
    saved = json.loads(cache.read_text(encoding="utf-8"))["headings"]
    assert saved["大气电学"]["by"] == "agent"


def test_an_agent_figure_record_is_never_overwritten(tmp_path: Path) -> None:
    text = "\\chapter{磁力}\n\n![](images/glyph.png)\n\n正文。\n"
    agent = FigureRecord(keep=False, by="agent", label="decoration", reason="drop cap")
    report = build_figures(text, existing={"glyph.png": agent}, reclassify=True)

    assert report.records["glyph.png"] == agent
    assert report.flagged == []


def test_force_profile_keeps_an_agent_edited_profile(tmp_path: Path) -> None:
    (tmp_path / "merged.md").write_text(BOOK, encoding="utf-8")
    (tmp_path / "book_profile.json").write_text(
        json.dumps({"subject": "physics", "by": "agent"}), encoding="utf-8"
    )
    profile = profile_book(tmp_path / "merged.md", tmp_path, llm=_Never(), force=True)
    assert profile.subject == "physics" and profile.by == "agent"
