"""Chinese text inside figures: tasks, specs, rendering, assembly.

pdflatex is never run: `run_pdflatex` is replaced by a fake that writes a
one-page PDF (or fails with a LaTeX-style log), so the tests cover what this
module decides, not what TeX does.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pymupdf
import pytest

from src import figure_text
from src.figure_text import (
    build_tasks,
    figure_text_report,
    latex_errors,
    legend_tex,
    overlay_tex,
    render,
    status_of,
    substitute,
    translated_images,
    validate_spec,
)
from src.figures import FIGURES_NAME, FigureRecord, save_figures
from src.kit import assemble

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

MAPPING = {
    "IMG_0000": "images/circuit.png",
    "IMG_0001": "images/photo.png",
    "IMG_0002": "images/plain.png",
    "IMG_0003": "images/manual.png",
}
RECORDS = {
    "circuit.png": FigureRecord(number="3.1", width=0.5, chinese_text=True),
    "photo.png": FigureRecord(number="3.2", width=0.8, chinese_text=True, keep=False),
    "plain.png": FigureRecord(number="3.3", chinese_text=False),
    "manual.png": FigureRecord(number="3.4", by="manual"),  # never asked
}
TRANSLATION = (
    "\\chapter{Circuits}\n\nThe source drives the current through the switch.\n\n"
    "\\bookfig{IMG_0000}{A simple circuit}\n\n\\bookfig{IMG_0002}{Plain}\n\n"
    "\\bookfig{IMG_0003}{Manual}\n% END-OF-CHUNK\n"
)


def _png(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.draw_rect(pymupdf.Rect(10, 10, 190, 90))
    page.get_pixmap(dpi=36).save(path)
    doc.close()


def _kit(tmp_path: Path) -> Path:
    kit = tmp_path / "book" / "kit"
    for name in ("chunks", "translated", "images"):
        (kit / name).mkdir(parents=True)
    for path in MAPPING.values():
        _png(kit / path)
    (kit / "chunks" / "001.md").write_text(
        "# 第3章 电路\n\n电源通过开关驱动电流。\n\n![](IMG_0000)\n图 3.1 简单电路\n", encoding="utf-8"
    )
    (kit / "translated" / "001.tex").write_text(TRANSLATION, encoding="utf-8")
    (kit / "image_map.json").write_text(json.dumps(MAPPING), encoding="utf-8")
    save_figures(kit / FIGURES_NAME, RECORDS)
    return kit


def _spec(kit: Path, data: dict) -> None:
    directory = kit / "figure_text"
    directory.mkdir(exist_ok=True)
    stem = Path(data["image"]).stem
    (directory / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


OVERLAY = {
    "image": "circuit.png",
    "method": "overlay",
    "labels": [
        {"zh": "电源", "en": "source", "box": [0.1, 0.2, 0.3, 0.3]},
        {"zh": "开关", "en": "switch $S$ (100%)", "box": [0.6, 0.7, 0.9, 0.8], "size": "small"},
    ],
    "accepted": False,
    "notes": "two labels overlaid",
}


@pytest.fixture
def fake_latex(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """pdflatex that succeeds (writing a one-page PDF) and records its input."""
    compiled: list[Path] = []

    def fake(tex: Path) -> tuple[bool, list[str]]:
        compiled.append(tex)
        doc = pymupdf.open()
        doc.new_page(width=300, height=150)
        doc.save(tex.with_suffix(".pdf"))
        doc.close()
        return True, []

    monkeypatch.setattr(figure_text, "run_pdflatex", fake)
    return compiled


# ── Tasks ───────────────────────────────────────────────────────────────────


def test_tasks_are_the_kept_images_with_chinese_text(tmp_path: Path) -> None:
    kit = _kit(tmp_path)

    payload = build_tasks(kit)

    (task,) = payload["tasks"]  # photo.png is dropped, plain.png has no text
    assert task["image"] == "circuit.png" and task["stem"] == "circuit"
    assert task["token"] == "IMG_0000" and task["chunk"] == "001" and task["number"] == "3.1"
    assert task["target_width_cm"] == 8.0  # 0.5 of the 16 cm text width
    assert "The source drives" in task["translated_context"]
    assert "电源通过开关" in task["source_context"]
    assert task["status"] == "todo" and payload["todo"] == ["circuit"]
    assert payload["not_asked"] == ["manual.png"]
    assert (kit / "figure_text" / "tasks.json").exists()


# ── Specs ───────────────────────────────────────────────────────────────────


def test_a_valid_spec_of_each_method_passes() -> None:
    assert validate_spec(OVERLAY) == []
    assert validate_spec({"method": "tikz", "labels": [{"zh": "电源", "en": "source"}]}) == []
    assert validate_spec({"method": "legend", "labels": [{"en": "source", "where": "upper left"}]}) == []


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ({"method": "paint", "labels": [{"en": "x"}]}, "method 'paint'"),
        ({"method": "tikz", "labels": []}, "non-empty list"),
        ({"method": "tikz", "labels": [{"zh": "电源"}]}, "no 'en' text"),
        ({"method": "tikz", "labels": [{"en": "电源 source"}]}, "still contains Chinese"),
        ({"method": "tikz", "labels": [{"en": "x", "size": "huge"}]}, "size 'huge'"),
        ({"method": "overlay", "labels": [{"en": "x"}]}, "'box' must be"),
        ({"method": "overlay", "labels": [{"en": "x", "box": [0.5, 0.2, 0.4, 0.3]}]}, "'box' must be"),
        ({"method": "overlay", "labels": [{"en": "x", "box": [0, 0, 1.2, 1]}]}, "'box' must be"),
        ({"method": "legend", "labels": [{"en": "x"}]}, "needs 'where'"),
    ],
)
def test_bad_specs_are_explained(spec: dict, problem: str) -> None:
    assert any(problem in p for p in validate_spec(spec))


def test_overlay_paints_each_box_and_flips_y_for_tikz() -> None:
    tex = overlay_tex(OVERLAY, "../images/circuit.png", 8.0)

    assert r"\includegraphics[width=8.00cm]{../images/circuit.png}" in tex
    # Box [0.1, 0.2, 0.3, 0.3] from the top is y 0.7..0.8 from the bottom.
    assert r"\fill[white] (0.1000,0.7000) rectangle (0.3000,0.8000);" in tex
    assert r"at (0.2000,0.7500) {source}" in tex
    assert r"\node[font=\small" in tex and r"switch $S$ (100\%)" in tex
    assert r"\begin{document}" in tex and tex.rstrip().endswith(r"\end{document}")


def test_legend_names_each_label_by_where_it_is() -> None:
    spec = {
        "method": "legend",
        "labels": [{"en": "source", "where": "upper left"}, {"en": "switch", "where": "on the right"}],
    }
    tex = legend_tex(spec, "../images/circuit.png", 8.0)

    assert r"\begin{minipage}{8.00cm}" in tex
    assert "upper left: source; on the right: switch." in tex


# ── Rendering ───────────────────────────────────────────────────────────────


def test_render_writes_compiles_and_previews_an_overlay(tmp_path: Path, fake_latex: list[Path]) -> None:
    kit = _kit(tmp_path)
    build_tasks(kit)
    _spec(kit, OVERLAY)

    result = render(kit, "circuit")

    assert result.ok and result.errors == []
    assert Path(result.preview).exists() and Path(result.pdf).exists()
    tex = (kit / "figure_text" / "circuit.tex").read_text(encoding="utf-8")
    assert "width=8.00cm]{../images/circuit.png}" in tex  # the task's width, a relative path
    assert fake_latex == [kit / "figure_text" / "circuit.tex"]
    assert status_of(kit, "circuit.png") == "rejected"  # rendered, not yet accepted


def test_an_accepted_spec_is_used_until_it_changes(tmp_path: Path, fake_latex: list[Path]) -> None:
    kit = _kit(tmp_path)
    build_tasks(kit)
    _spec(kit, OVERLAY)
    render(kit, "circuit")
    _spec(kit, {**OVERLAY, "accepted": True, "notes": "checked"})  # accepting is not a change

    assert status_of(kit, "circuit.png") == "done"
    assert translated_images(kit) == {"circuit.png": "figure_text/circuit.pdf"}
    mapping, swapped = substitute(MAPPING, kit)
    assert swapped == 1 and mapping["IMG_0000"] == "figure_text/circuit.pdf"
    assert mapping["IMG_0002"] == "images/plain.png"

    changed = {**OVERLAY, "accepted": True}
    changed["labels"] = [{**OVERLAY["labels"][0], "en": "battery"}]
    _spec(kit, changed)
    assert status_of(kit, "circuit.png") == "stale"
    assert translated_images(kit) == {}


def test_tikz_needs_its_source_without_chinese(tmp_path: Path, fake_latex: list[Path]) -> None:
    kit = _kit(tmp_path)
    spec = {"image": "circuit.png", "method": "tikz", "labels": [{"zh": "电源", "en": "source"}]}
    _spec(kit, spec)

    assert "write the standalone document" in render(kit, "circuit").errors[0]

    tex = kit / "figure_text" / "circuit.tex"
    tex.write_text("\\documentclass{standalone}\\begin{document}电源\\end{document}", encoding="utf-8")
    assert "still contains Chinese" in render(kit, "circuit").errors[0]
    assert fake_latex == []

    tex.write_text("\\documentclass{standalone}\\begin{document}source\\end{document}", encoding="utf-8")
    assert render(kit, "circuit").ok
    _spec(kit, {**spec, "accepted": True})
    assert status_of(kit, "circuit.png") == "done"
    tex.write_text("\\documentclass{standalone}\\begin{document}changed\\end{document}", encoding="utf-8")
    assert status_of(kit, "circuit.png") == "stale"  # the TikZ source is part of the hash


def test_a_failed_compile_is_reported_and_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kit = _kit(tmp_path)
    _spec(kit, {**OVERLAY, "accepted": True})
    monkeypatch.setattr(figure_text, "run_pdflatex", lambda tex: (False, ["Undefined control sequence  [l.9 \\foo]"]))

    result = render(kit, "circuit")

    assert not result.ok and "Undefined control sequence" in result.errors[0]
    assert status_of(kit, "circuit.png") == "failed"
    assert translated_images(kit) == {}


def test_latex_errors_are_read_with_their_line() -> None:
    log = "This is pdfTeX\n! Undefined control sequence.\nl.12 \\foo\n         {bar}\n! Emergency stop.\n"
    assert latex_errors(log) == ["Undefined control sequence.  [l.12 \\foo]", "Emergency stop."]


def test_without_pdflatex_the_render_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(figure_text.shutil, "which", lambda name: None)
    tex = tmp_path / "x.tex"
    tex.write_text("x", encoding="utf-8")
    assert figure_text.run_pdflatex(tex) == (False, ["pdflatex not found on PATH"])


# ── Assembly, report, script ────────────────────────────────────────────────


def test_assembly_uses_the_english_pdf_for_a_finished_figure(tmp_path: Path, fake_latex: list[Path]) -> None:
    kit = _kit(tmp_path)
    build_tasks(kit)
    _spec(kit, OVERLAY)
    render(kit, "circuit")
    _spec(kit, {**OVERLAY, "accepted": True})

    path, _ = assemble(kit)
    body = path.read_text(encoding="utf-8")

    assert r"\bookfigx{0.50}{figure_text/circuit.pdf}{3.1}{A simple circuit}" in body
    assert "{images/plain.png}" in body


def test_report_and_script(
    tmp_path: Path,
    fake_latex: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    kit = _kit(tmp_path)
    summary, text = figure_text_report(kit)
    assert summary["images"] == 1 and summary["done"] == 0
    assert summary["not_done"][0]["status"] == "todo" and summary["not_asked"] == ["manual.png"]
    assert "Never asked" in text

    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = importlib.import_module("figure_text")
    _spec(kit, OVERLAY)
    assert script.main([str(kit), "render", "circuit", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    _spec(kit, {**OVERLAY, "accepted": True})

    assert script.main([str(kit), "report", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["done"] == 1 and report["methods"] == {"overlay": 1}
    assert Path(report["log"]).exists()

    assert script.main([str(kit), "tasks", "--json"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert tasks["todo"] == [] and "translated_context" not in tasks["tasks"][0]
