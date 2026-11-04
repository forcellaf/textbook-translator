"""Meaning verification: reading the agents' files, locating pages, the report.

The agents themselves are never run here; their files are written by hand,
including the mistakes a real agent makes (an undoubled backslash, a quoted
excerpt that is not in the file, a formula it fixed although told not to).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pymupdf
import pytest

from src.verify import (
    compact,
    find_line,
    load_verification,
    locate_pages,
    prepare_page_checks,
    verification_report,
)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

SOURCE = """# 第1章 电场

电荷之间存在相互作用力。同种电荷相互排斥，异种电荷相互吸引。

库仑定律的表达式为

$$
F = k \\frac{q_1 q_2}{r^2}
$$

真空中的静电力常量为 $k = 9.0 \\times 10^{9}$，它不是一个无量纲的数。
"""

TRANSLATION = """\\chapter{The Electric Field}

Charges exert forces on one another. Like charges repel each other.

Coulomb's law reads
\\[
F = k \\frac{q_1 q_2}{r^3}
\\]
In vacuum the electrostatic constant is $k = 9.0 \\times 10^{9}$, which is not a dimensionless number.
% END-OF-CHUNK
"""


def _kit(tmp_path: Path) -> Path:
    work = tmp_path / "book"
    kit = work / "kit"
    for name in ("chunks", "translated", "verify"):
        (kit / name).mkdir(parents=True)
    (kit / "chunks" / "001.md").write_text(SOURCE, encoding="utf-8")
    (kit / "translated" / "001.tex").write_text(TRANSLATION, encoding="utf-8")
    (kit / "chunks" / "002.md").write_text("第二块。\n", encoding="utf-8")
    (kit / "translated" / "002.tex").write_text("Second chunk.\n", encoding="utf-8")
    return kit


def _write(path: Path, data: dict | str) -> None:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


FINDINGS = {
    "chunk": "001",
    "findings": [
        {
            "id": 1,
            "kind": "omission",
            "severity": "major",
            "action": "fixed",
            "source_excerpt": "异种电荷相互吸引。",
            "translation_excerpt": "Like charges repel each other.",
            "fixed_text": "Like charges repel each other. Unlike charges attract.",
            "explanation": "The sentence about unlike charges was dropped.",
        },
        {
            "id": 2,
            "kind": "formula",
            "severity": "major",
            "action": "page-check",
            "source_excerpt": "F = k \\frac{q_1 q_2}{r^2}",
            "translation_excerpt": "F = k \\frac{q_1 q_2}{r^3}",
            "explanation": "The exponent of r is 2 in the source, 3 in the translation.",
        },
        {
            "id": 3,
            "kind": "terminology",
            "severity": "minor",
            "action": "left",
            "source_excerpt": "静电力常量",
            "translation_excerpt": "electrostatic constant",
            "explanation": "Usually 'Coulomb constant' in English.",
        },
    ],
}


# ── Matching ────────────────────────────────────────────────────────────────


def test_compact_ignores_spacing_and_dollars() -> None:
    assert compact("$1 0 ^ { - 2 }$") == compact("10^{-2}")


def test_find_line_reports_the_line_and_misses_cleanly() -> None:
    assert find_line(TRANSLATION, "F = k \\frac{q_1 q_2}{r^3}") == 7
    assert find_line(TRANSLATION, "Unlike charges attract.") is None
    assert find_line(TRANSLATION, "   ") is None


def test_an_undoubled_backslash_in_the_agents_json_still_matches() -> None:
    # json.loads('"\\frac"') with one backslash gives a form feed + "rac".
    broken = json.loads('"F = k \\frac{q_1 q_2}{r^3}"'.replace("\\\\", "\\"))
    assert "\x0c" in broken
    assert find_line(TRANSLATION, broken) == 7


# ── Reading the files ───────────────────────────────────────────────────────


def test_a_valid_result_is_read_with_locations(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    translated = kit / "translated" / "001.tex"
    translated.write_text(
        TRANSLATION.replace("repel each other.", "repel each other. Unlike charges attract."),
        encoding="utf-8",
    )
    _write(kit / "verify" / "001.json", FINDINGS)

    result = load_verification(kit, "001")

    assert result is not None and result.problems == []
    fixed, formula, term = result.findings
    assert fixed.location == "translated/001.tex:3" and fixed.flags == []
    assert formula.line == 7 and formula.flags == []
    assert term.flags == []
    assert [f.id for f in result.pending_page_checks()] == [2]
    assert load_verification(kit, "002") is None  # not verified yet


def test_unconfirmed_excerpts_are_flagged_not_dropped(tmp_path: Path) -> None:
    kit = _kit(tmp_path)  # the fix was never applied to the file
    data = json.loads(json.dumps(FINDINGS))
    data["findings"][2]["source_excerpt"] = "这句话不在源文中"
    _write(kit / "verify" / "001.json", data)

    result = load_verification(kit, "001")

    assert result is not None and len(result.findings) == 3
    assert "fixed text not found" in result.findings[0].flags[0]
    assert "source excerpt not found" in result.findings[2].flags[0]


def test_rule_breaking_findings_are_problems(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _write(
        kit / "verify" / "001.json",
        {
            "findings": [
                {"id": 1, "kind": "formula", "severity": "major", "action": "fixed",
                 "fixed_text": "x", "explanation": "fixed a formula"},
                {"id": 2, "kind": "omission", "severity": "minor", "action": "page-check",
                 "explanation": "minor page check"},
                {"id": 3, "kind": "style", "severity": "minor", "action": "left",
                 "explanation": "not a kind"},
                {"id": 4, "kind": "omission", "severity": "major", "action": "fixed",
                 "explanation": "no fixed text"},
            ]
        },
    )

    problems = load_verification(kit, "001").problems  # type: ignore[union-attr]

    assert any("mathematics goes to a page check" in p for p in problems)
    assert any("only a major finding gets a page check" in p for p in problems)
    assert any("kind 'style'" in p for p in problems)
    assert any("without fixed_text" in p for p in problems)


def test_malformed_json_is_a_problem(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _write(kit / "verify" / "001.json", '{"findings": [ "\\url" ]}')
    result = load_verification(kit, "001")
    assert result is not None and "not valid JSON" in result.problems[0]


def test_a_page_check_resolves_its_finding(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _write(kit / "verify" / "001.json", FINDINGS)
    translated = kit / "translated" / "001.tex"
    translated.write_text(TRANSLATION.replace("{r^3}", "{r^2}"), encoding="utf-8")
    _write(
        kit / "verify" / "001.pages.json",
        {
            "chunk": "001",
            "checks": [
                {"finding": 2, "pages": [12], "verdict": "translation-error", "action": "fixed",
                 "page_reads": "F = k \\frac{q_1 q_2}{r^2}",
                 "translation_fix": {"old": "{r^3}", "new": "F = k \\frac{q_1 q_2}{r^2}"},
                 "explanation": "The page prints r squared."},
                {"finding": 3, "verdict": "no-error", "explanation": "not a page check"},
                {"finding": 2, "verdict": "parse-error", "action": "fixed",
                 "explanation": "a parse error needs a source fix"},
            ],
        },
    )

    result = load_verification(kit, "001")

    assert result is not None
    assert result.pending_page_checks() == []
    assert result.page_checks[0].flags == [] and result.page_checks[0].pages == [12]
    assert any("no page-check finding with that id" in p for p in result.problems)
    assert any("without a source_fix" in p for p in result.problems)


# ── Locating the page ───────────────────────────────────────────────────────


def _content_list(work: Path) -> None:
    entries = [
        {"type": "text", "text": "电荷之间存在相互作用力。同种电荷相互排斥，异种电荷相互吸引。", "page_idx": 0},
        {"type": "text", "text": "库仑定律的表达式为", "page_idx": 0},
        {"type": "equation", "text": "$$\nF = k \\frac { q _ 1 q _ 2 } { r ^ 2 }\n$$", "page_idx": 1},
        {"type": "text", "text": "真空中的静电力常量为 $k = 9.0 \\times 10^{9}$，它不是一个无量纲的数。",
         "page_idx": 1},
        # The same formula again, much later in the book.
        {"type": "equation", "text": "$$F = k \\frac{q_1 q_2}{r^2}$$", "page_idx": 40},
        {"type": "text", "text": "另一章里的一段完全不同的文字，用来占位。", "page_idx": 40},
    ]
    _write(work / "content_list.json", json.dumps(entries, ensure_ascii=False))


def test_locate_finds_the_page_within_the_chunks_own_span(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _content_list(kit.parent)

    located = locate_pages(kit.parent, SOURCE, "F = k \\frac{q_1 q_2}{r^2}")

    assert located.pages == [2] and located.method == "exact"


def test_locate_matches_fuzzily_and_falls_back_to_interpolation(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _content_list(kit.parent)

    fuzzy = locate_pages(kit.parent, SOURCE, "电荷之间存在相互作用力。同种电荷相互排斥，异种电荷相互吸引！")
    assert fuzzy.method == "fuzzy" and fuzzy.pages[0] == 1

    guessed = locate_pages(kit.parent, SOURCE, "完全找不到的一句话，这句话在任何页面上都没有出现过。")
    assert guessed.method == "interpolated" and guessed.pages

    assert locate_pages(tmp_path, SOURCE, "x").method == "none"  # no content list


def test_prepare_page_checks_renders_original_page_numbers(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    work = kit.parent
    _content_list(work)
    _write(kit / "verify" / "001.json", FINDINGS)
    (work / "parts").mkdir()
    _write(work / "parts" / "manifest.json", {"page_offset": 10, "trimmed": True})
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=200, height=200)
    doc.save(work / "source_body.pdf")
    doc.close()

    payload = prepare_page_checks(kit, work)

    (task,) = payload["tasks"]
    assert task["chunk"] == "001" and task["finding"] == 2
    assert task["pages"] == [12]  # body page 2 + offset 10
    assert Path(task["images"][0]).name == "p0012.png" and Path(task["images"][0]).exists()
    assert payload["chunks"] == ["001"]
    assert json.loads((kit / "verify" / "page_checks.json").read_text(encoding="utf-8")) == payload


# ── The report and the script ───────────────────────────────────────────────


def test_report_lists_what_the_run_log_needs(tmp_path: Path) -> None:
    kit = _kit(tmp_path)
    _write(kit / "verify" / "001.json", FINDINGS)

    summary, text = verification_report(kit)

    assert summary["chunks"] == 2 and summary["verified"] == 1
    assert summary["unverified"] == ["002"]
    assert summary["counts"]["action"] == {"fixed": 1, "page-check": 1, "left": 1}
    assert summary["fixed"][0]["after"].endswith("Unlike charges attract.")
    assert summary["left"][0]["kind"] == "terminology"
    assert summary["page_checks_pending"] == [{"chunk": "001", "finding": 2}]
    assert summary["flagged"] == 1  # the fix is not in the file
    assert "page check: PENDING" in text


def test_script_exits_1_only_for_malformed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = importlib.import_module("verify_kit")
    kit = _kit(tmp_path)
    _write(kit / "verify" / "001.json", FINDINGS)

    assert script.main([str(kit), "report", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert Path(summary["log"]).exists() and summary["findings"] == 3

    _write(kit / "verify" / "002.json", "{not json")
    assert script.main([str(kit), "report", "--json"]) == 1
    assert "002" in json.loads(capsys.readouterr().out)["problems"]
