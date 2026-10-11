"""Tests for src.figures and the figure handling in assembly and lint.

No real API call is made: the classifier is a `FakeVisionLLM` implementing
`VisionLLM`. Fixtures follow the reference book's real signatures -- its
caption forms (`图 18.3`, `图18.3`, `图 G.2`), its sub-panel runs, the
decorative glyph that opened chapter 18 and became "Figure 18.1", and the
layout of a MinerU content list.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pymupdf
import pytest
from tenacity import wait_none

from src.figures import (
    BY_CAPTION,
    BY_CLASSIFIER,
    BY_DEFAULT,
    BY_MANUAL,
    DEFAULT_WIDTH,
    FIGURES_NAME,
    MODE_CLASSIFY,
    MODE_DRY_RUN,
    MODE_OFF,
    REVIEW_NAME,
    FigureRecord,
    apply_linked_widths,
    build_figures,
    caption_number,
    link_images,
    linked_entries,
    load_figures,
    save_figures,
    scan_markdown,
    split_number,
    strip_dropped_images,
    write_review,
)
from src.kit import apply_figures, assemble
from src.latex import assemble_document
from src.lint import Severity, lint_kit
from src.llm.base import VisionLLM

# ── Helpers ─────────────────────────────────────────────────────────────────

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _fake_png(name: str) -> bytes:
    """Bytes that sniff as PNG and identify the image to the fake."""
    return PNG_HEADER + name.encode()


class FakeVisionLLM(VisionLLM):
    """Scripted classifier: a reply (or exception) per image name."""

    def __init__(self, replies: dict[str, object]) -> None:
        self.replies = replies
        self.seen: list[str] = []

    def generate_with_images(self, system_prompt, user_text, images, temperature=0.0):
        (data, mime), = images
        assert mime == "image/png"
        name = data[len(PNG_HEADER) :].decode()
        self.seen.append(name)
        reply = self.replies[name]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _verdict(label: str, confidence: float) -> str:
    return json.dumps({"label": label, "confidence": confidence, "reason": "test"})


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.figures._RETRY_WAIT", wait_none())


def _images(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "images"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(_fake_png(name))
    return directory


# The reference book's chapter 18 opening: the glyph after the heading, then
# numbered figures, a lost caption (18.4) and a sub-panel run.
CHAPTER_18 = """\\chapter{磁力}

![](images/glyph.jpg)

磁场对其中的运动电荷,根据洛伦兹力公式有磁力的作用。

![](images/f1.jpg)
图 18.1 带电粒子在磁场中的运动

![](images/f2.jpg)
图18.2 回旋加速器

![](images/f3.jpg)
图 18.3 霍尔效应

如图18.4所示，设导线截面积为 $S$ 。

![](images/lost.jpg)

![](images/a.jpg)
(a)

![](images/b.jpg)
(b)
图 18.5 磁聚焦(a)与磁镜(b)
"""


# ── Numbers ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("caption", "number"),
    [
        ("图 18.3 霍尔效应", "18.3"),
        ("图18.3 霍尔效应", "18.3"),
        ("图 G.2 雷暴云中的电荷分布", "G.2"),
        ("图 12.1(a) 质子内电荷分布", "12.1"),
        ("图19.24 思考题19.11用图", "19.24"),
        # eee_practice: three levels, exercise and appendix prefixes, a prime.
        # A two-level pattern read 图 1.2.1 as 1.2 and collapsed 606 figures
        # onto 49 numbers.
        ("图 1.2.1 电压源模型和电流源模型", "1.2.1"),
        ("图13.3.13 TL084的传输特性曲线", "13.3.13"),
        ("图 2.4.27(a)", "2.4.27"),
        ("图 11.3.11(续)", "11.3.11"),
        ("图 T1.1", "T1.1"),
        ("图T3.2", "T3.2"),
        ("图A1.8-1", "A1.8-1"),
        ("图 A5.7-1 状态转换图", "A5.7-1"),
        ("图 23.3.3' 习题 23.2 图 2", "23.3.3'"),
    ],
)
def test_caption_numbers_in_the_reference_books_forms(caption: str, number: str) -> None:
    assert caption_number(caption) == number


@pytest.mark.parametrize(
    ("number", "parts"),
    [
        ("18.3", ("18", 3)),
        ("G.2", ("G", 2)),
        ("1.2.1", ("1.2", 1)),
        ("T1.4", ("T1", 4)),
        ("A1.8-1", ("A1.8", 1)),
        ("23.3.3'", ("23.3", 3)),
    ],
)
def test_numbers_split_into_sequence_and_position_at_any_depth(number, parts) -> None:
    assert split_number(number) == parts


@pytest.mark.parametrize(
    "line",
    [
        # Prose that opens with a figure reference is not a caption.
        "图12.4画出了两个点电荷 $q_{1}$ 和 $q_{2}$ 对第三个点电荷的作用力的叠加情况。",
        "如图12.7所示。",
        "(a)",
        "(12.15)",
    ],
)
def test_lines_that_are_not_captions(line: str) -> None:
    assert caption_number(line) is None


def test_number_from_image_caption_is_preferred_over_the_markdown() -> None:
    text = "![](images/x.jpg)\n图 18.9 from the markdown\n"
    entries = [{"type": "image", "img_path": "images/x.jpg", "image_caption": ["图 18.3 霍尔效应"]}]

    report = build_figures(text, entries=entries, mode=MODE_OFF)

    assert report.records["x.jpg"].number == "18.3"


def test_number_from_the_markdown_when_image_caption_is_empty() -> None:
    text = "![](images/x.jpg)  \n图 18.3 霍尔效应\n"
    entries = [{"type": "image", "img_path": "images/x.jpg", "image_caption": []}]

    report = build_figures(text, entries=entries, mode=MODE_OFF)

    assert report.records["x.jpg"].number == "18.3"


# Verbatim from a real vlm archive of the reference book's first 8 pages
# (hashes shortened): figure 12.1's panels came back as *charts*, with the
# caption in `chart_caption`; 12.2's as images. Both lists mix the panel
# labels with the caption. Body text sets the reference width.
REAL_MD = """电量取正值，负电荷电量取负值。

![](images/ad0ce35b.jpg)
(a)

![](images/5c361123.jpg)
(b)
图 12.1 质子内(a)与中子内(b)电荷分布图

## 2. 电荷的量子性

![](images/da53e64b.jpg)
(a)

![](images/00fbc737.jpg)
(b)
图 12.2 氢分子(a)与氦原子(b)结构示意图
"""
BODY_REAL = "作为电磁学的开始,本章讲解静止电荷相互作用的规律。" * 3
REAL_ENTRIES = [
    {"type": "text", "text": BODY_REAL, "bbox": [63, 313, 905, 491], "page_idx": 2},
    {"type": "text", "text": BODY_REAL, "bbox": [65, 600, 902, 700], "page_idx": 3},
    {"type": "text", "text": "12.1 电荷", "text_level": 2, "bbox": [67, 512, 211, 535], "page_idx": 2},
    {"type": "header", "text": "第3篇 电磁学", "bbox": [264, 23, 416, 41], "page_idx": 1},
    {"type": "chart", "img_path": "images/ad0ce35b.jpg", "content": "", "chart_caption": ["(a)"],
     "chart_footnote": [], "bbox": [256, 66, 529, 202], "page_idx": 3},
    {"type": "chart", "img_path": "images/5c361123.jpg", "content": "",
     "chart_caption": ["(b)", "图 12.1 质子内(a)与中子内(b)电荷分布图"], "chart_footnote": [],
     "bbox": [539, 65, 797, 208], "page_idx": 3},
    {"type": "image", "img_path": "images/da53e64b.jpg", "image_caption": ["(a)"],
     "image_footnote": [], "content": "", "bbox": [241, 558, 450, 658], "page_idx": 4},
    {"type": "image", "img_path": "images/00fbc737.jpg",
     "image_caption": ["(b)", "图 12.2 氢分子(a)与氦原子(b)结构示意图"], "image_footnote": [],
     "content": "", "bbox": [506, 553, 731, 663], "page_idx": 4},
]


def test_real_archive_charts_and_images_get_numbers_panels_and_widths() -> None:
    # Without the markdown captions, so the numbers must come from the lists.
    text = "\n".join(line for line in REAL_MD.split("\n") if not line.startswith("图"))

    records = build_figures(text, entries=REAL_ENTRIES, mode=MODE_OFF).records

    assert records["5c361123.jpg"].number == "12.1"  # from chart_caption
    assert records["00fbc737.jpg"].number == "12.2"
    assert records["ad0ce35b.jpg"].panel_of == "12.1"
    assert records["da53e64b.jpg"].panel_of == "12.2"
    # Measured against the ~838-wide text block; none fell back to 0.6.
    widths = {name: r.width for name, r in records.items()}
    assert widths == {
        "ad0ce35b.jpg": 0.33, "5c361123.jpg": 0.31, "da53e64b.jpg": 0.25, "00fbc737.jpg": 0.27
    }
    assert all(r.by == BY_CAPTION and not r.flags for r in records.values())


def test_sub_panel_label_is_not_absorbed_into_the_number() -> None:
    report = build_figures(CHAPTER_18, mode=MODE_OFF)

    assert report.records["b.jpg"].number == "18.5"
    # Panel (a) is part of figure 18.5 but does not carry its number.
    assert report.records["a.jpg"].number is None
    assert report.records["a.jpg"].panel_of == "18.5"
    assert report.records["a.jpg"].by == BY_DEFAULT  # no bbox in this fixture


def test_trailing_panel_after_its_caption_belongs_to_that_figure() -> None:
    """The parser sometimes puts the caption between the panels."""
    text = "![](images/a.jpg)  \n(a)  \n图 23.1 圆孔衍射\n\n![](images/b.jpg)  \n(b)\n\n正文。\n"

    report = build_figures(text, mode=MODE_OFF)

    assert report.records["a.jpg"].number == "23.1"
    assert report.records["b.jpg"].panel_of == "23.1"


def test_chapter_opening_image_is_marked_as_such() -> None:
    sites = {s.name: s for s in scan_markdown(CHAPTER_18)}

    assert sites["glyph.jpg"].after_heading == "chapter: 磁力"
    assert sites["f1.jpg"].after_heading is None
    split = scan_markdown("## 第18章\n\n## 磁力\n\n![](images/g.jpg)\n")
    assert split[0].after_heading == "chapter: 第18章"


# ── Widths ──────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = 1566.0, 2246.0  # the reference PDF's page, in points
BODY = "物体能产生电磁现象,现在都归因于物体带上了电荷以及这些电荷的运动。" * 3


def _layout(scale_x: float = 1.0, scale_y: float = 1.0) -> list[dict]:
    """Two pages of body text and three images, in PDF points by default."""

    def box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
        return [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]

    entries: list[dict] = []
    for page in range(2):
        for row in range(5):
            y = 300 + row * 120
            entries.append(
                {"type": "text", "text": BODY, "bbox": box(150, y, 1416, y + 100), "page_idx": page}
            )
        # A heading and a short line: narrower, and must not pull the median.
        entries.append(
            {"type": "text", "text": "12.1 电荷", "text_level": 1,
             "bbox": box(150, 200, 400, 240), "page_idx": page}
        )
        entries.append({"type": "text", "text": "短行。", "bbox": box(150, 900, 300, 940),
                        "page_idx": page})
    entries += [
        {"type": "image", "img_path": "images/half.jpg", "bbox": box(150, 1000, 783, 1400),
         "image_caption": [], "page_idx": 0},
        {"type": "image", "img_path": "images/tiny.jpg", "bbox": box(700, 1000, 760, 1060),
         "image_caption": [], "page_idx": 1},
        {"type": "image", "img_path": "images/wide.jpg", "bbox": box(60, 1100, 1500, 1500),
         "image_caption": [], "page_idx": 1},
    ]
    return entries


WIDTH_TEXT = "![](images/half.jpg)\n\n![](images/tiny.jpg)\n\n![](images/wide.jpg)\n\n![](images/none.jpg)\n"


def test_widths_are_the_same_in_absolute_and_normalized_coordinates() -> None:
    absolute = build_figures(WIDTH_TEXT, entries=_layout(), mode=MODE_OFF).records
    normalized = build_figures(
        WIDTH_TEXT, entries=_layout(1000 / PAGE_W, 1000 / PAGE_H), mode=MODE_OFF
    ).records

    assert absolute["half.jpg"].width == pytest.approx(0.5, abs=0.01)
    for name in ("half.jpg", "tiny.jpg", "wide.jpg"):
        assert absolute[name].width == normalized[name].width


def test_widths_are_clamped_at_both_ends() -> None:
    records = build_figures(WIDTH_TEXT, entries=_layout(), mode=MODE_OFF).records

    assert records["tiny.jpg"].width == 0.20
    assert records["wide.jpg"].width == 0.90


def test_no_bbox_falls_back_to_the_default_and_is_reported(tmp_path: Path) -> None:
    report = build_figures(WIDTH_TEXT, entries=_layout(), mode=MODE_OFF)
    record = report.records["none.jpg"]

    assert record.width == DEFAULT_WIDTH
    assert record.by == BY_DEFAULT
    assert "no-bbox" in record.flags
    assert "none.jpg" in report.defaulted

    review = write_review(tmp_path, report.records).read_text(encoding="utf-8")
    assert "none.jpg" in review


# ── Classification ──────────────────────────────────────────────────────────


def _classify(tmp_path: Path, replies: dict[str, object], **kwargs):
    names = ["glyph.jpg", "f1.jpg", "f2.jpg", "f3.jpg", "lost.jpg", "a.jpg", "b.jpg"]
    fake = FakeVisionLLM(replies)
    report = build_figures(
        CHAPTER_18, images_dir=_images(tmp_path, *names), vision=fake, **kwargs
    )
    return report, fake


FIGURE_LOST = _verdict("figure", 0.9)


def test_confident_decoration_is_dropped(tmp_path: Path) -> None:
    report, _ = _classify(
        tmp_path, {"glyph.jpg": _verdict("decoration", 0.95), "lost.jpg": FIGURE_LOST}
    )
    glyph = report.records["glyph.jpg"]

    assert glyph.keep is False
    assert glyph.by == BY_CLASSIFIER
    assert (glyph.label, glyph.confidence) == ("decoration", 0.95)
    assert report.dropped == ["glyph.jpg"]


def test_unconfident_decoration_is_kept_and_flagged(tmp_path: Path) -> None:
    report, _ = _classify(
        tmp_path, {"glyph.jpg": _verdict("decoration", 0.7), "lost.jpg": FIGURE_LOST}
    )

    assert report.records["glyph.jpg"].keep is True
    assert "low-confidence" in report.records["glyph.jpg"].flags


@pytest.mark.parametrize(
    ("reply", "flag"),
    [
        ("Sure! It looks decorative to me.", "malformed-reply"),
        ('{"label": "ornament", "confidence": 0.99}', "malformed-reply"),
        ('{"label": "decoration"}', "malformed-reply"),
        (RuntimeError("HTTP 503"), "classifier-error"),
    ],
)
def test_every_failure_keeps_the_image(tmp_path: Path, reply: object, flag: str) -> None:
    report, _ = _classify(tmp_path, {"glyph.jpg": reply, "lost.jpg": FIGURE_LOST})
    glyph = report.records["glyph.jpg"]

    assert glyph.keep is True
    assert flag in glyph.flags
    assert glyph.label is None  # not cached: the next run asks again


def test_a_permanent_api_failure_stops_classifying_and_keeps_everything(tmp_path: Path) -> None:
    report, fake = _classify(
        tmp_path, {"glyph.jpg": ValueError("HTTP 402: out of credit"), "lost.jpg": FIGURE_LOST}
    )

    assert fake.seen == ["glyph.jpg"]
    assert report.stopped
    assert all(r.keep for r in report.records.values())


def test_confident_decoration_in_a_numbering_gap_is_kept(tmp_path: Path) -> None:
    """18.4 has no caption: the image where it should be is probably it."""
    report, _ = _classify(
        tmp_path,
        {"glyph.jpg": _verdict("decoration", 0.95), "lost.jpg": _verdict("decoration", 0.97)},
    )
    lost = report.records["lost.jpg"]

    assert lost.keep is True
    assert "numbering-gap" in lost.flags
    assert report.records["glyph.jpg"].keep is False  # 18.1 follows it: no gap


def test_captioned_images_and_panels_never_reach_the_classifier(tmp_path: Path) -> None:
    _, fake = _classify(tmp_path, {"glyph.jpg": FIGURE_LOST, "lost.jpg": FIGURE_LOST})

    assert sorted(fake.seen) == ["glyph.jpg", "lost.jpg"]


def test_dry_run_labels_but_drops_nothing(tmp_path: Path) -> None:
    report, _ = _classify(
        tmp_path,
        {"glyph.jpg": _verdict("decoration", 0.95), "lost.jpg": FIGURE_LOST},
        mode=MODE_DRY_RUN,
    )
    glyph = report.records["glyph.jpg"]

    assert glyph.keep is True
    assert glyph.label == "decoration"
    assert "would-drop" in glyph.flags
    assert report.dropped == []
    assert report.would_drop == ["glyph.jpg"]
    rendered = report.render()
    assert "would drop (dry run): 1" in rendered
    assert "decoration 1" in rendered


def test_no_classify_keeps_every_uncaptioned_image_for_review(tmp_path: Path) -> None:
    report, fake = _classify(tmp_path, {}, mode=MODE_OFF)

    assert fake.seen == []
    assert all(r.keep for r in report.records.values())
    assert {"glyph.jpg", "lost.jpg"} <= set(report.flagged)


def test_cached_labels_are_reused_without_a_call(tmp_path: Path) -> None:
    first, _ = _classify(
        tmp_path,
        {"glyph.jpg": _verdict("decoration", 0.95), "lost.jpg": FIGURE_LOST},
        mode=MODE_DRY_RUN,
    )
    second, fake = _classify(tmp_path, {}, existing=first.records, mode=MODE_CLASSIFY)

    assert fake.seen == []
    assert second.records["glyph.jpg"].keep is False


def test_reclassify_asks_again_but_leaves_manual_entries_alone(tmp_path: Path) -> None:
    first, _ = _classify(
        tmp_path,
        {"glyph.jpg": _verdict("fragment", 0.7), "lost.jpg": FIGURE_LOST},
        mode=MODE_DRY_RUN,
    )
    existing = dict(first.records)
    existing["lost.jpg"] = FigureRecord(keep=False, by=BY_MANUAL, reason="mine")

    second, fake = _classify(
        tmp_path,
        {"glyph.jpg": _verdict("decoration", 0.95)},
        existing=existing,
        reclassify=True,
    )

    assert fake.seen == ["glyph.jpg"]
    assert second.records["glyph.jpg"].label == "decoration"
    assert second.records["glyph.jpg"].keep is False
    assert second.records["lost.jpg"] == existing["lost.jpg"]


def test_manual_entries_survive_a_rerun(tmp_path: Path) -> None:
    manual = FigureRecord(number="18.0", width=0.33, keep=True, by=BY_MANUAL, reason="mine")
    path = tmp_path / FIGURES_NAME
    save_figures(path, {"glyph.jpg": manual})

    report, fake = _classify(
        tmp_path, {"lost.jpg": FIGURE_LOST}, existing=load_figures(path)
    )
    save_figures(path, report.records)

    assert "glyph.jpg" not in fake.seen
    assert load_figures(path)["glyph.jpg"] == manual


def test_dropped_images_are_removed_from_the_markdown() -> None:
    records = {"glyph.jpg": FigureRecord(keep=False), "f1.jpg": FigureRecord(number="18.1")}

    text, removed = strip_dropped_images(CHAPTER_18, records)

    assert removed == 1
    assert "glyph.jpg" not in text
    assert "![](images/f1.jpg)" in text
    assert "\\chapter{磁力}\n\n\n磁场" in text  # only the image line went


# ── Review sheet ────────────────────────────────────────────────────────────


def test_review_sheet_lists_dropped_first_with_relative_thumbnails(tmp_path: Path) -> None:
    records = {
        "kept.jpg": FigureRecord(number="1.1", by=BY_CAPTION),
        "flag.jpg": FigureRecord(flags=["low-confidence"], label="decoration", confidence=0.7),
        "gone.jpg": FigureRecord(keep=False, label="decoration", confidence=0.95, by=BY_CLASSIFIER),
    }

    html = write_review(tmp_path, records, {"IMG_0007": "images/flag.jpg"}).read_text(
        encoding="utf-8"
    )

    assert (tmp_path / REVIEW_NAME).exists()
    assert html.index("gone.jpg") < html.index("flag.jpg")
    assert "kept.jpg" not in html
    assert 'src="images/gone.jpg"' in html
    assert "IMG_0007" in html
    assert "http" not in html


# ── Assembly ────────────────────────────────────────────────────────────────

MAPPING = {f"IMG_000{i}": f"images/{name}" for i, name in enumerate(
    ["fig.jpg", "plain.jpg", "gone.jpg", "pa.jpg", "pb.jpg"]
)}
RECORDS = {
    "fig.jpg": FigureRecord(number="18.3", width=0.42, by=BY_CAPTION),
    "plain.jpg": FigureRecord(width=0.3, by=BY_CLASSIFIER, label="figure"),
    "gone.jpg": FigureRecord(keep=False, by=BY_CLASSIFIER, label="decoration"),
    "pa.jpg": FigureRecord(width=0.6, panel_of="18.4", by=BY_CAPTION),
    "pb.jpg": FigureRecord(number="18.4", width=0.6, by=BY_CAPTION),
}


def test_assembly_rewrites_a_nested_brace_caption_intact() -> None:
    text = r"\bookfig{IMG_0000}{Field of $E_{x}$ near {the} edge}"

    out, problems = apply_figures(text, MAPPING, RECORDS)

    assert out == r"\bookfigx{0.42}{IMG_0000}{18.3}{Field of $E_{x}$ near {the} edge}"
    assert problems == []


def test_assembly_gives_an_uncaptioned_figure_no_number() -> None:
    out, _ = apply_figures(r"\bookfig{IMG_0001}{}", MAPPING, RECORDS)

    assert out == r"\bookfigx{0.30}{IMG_0001}{}{}"


def test_assembly_removes_a_dropped_figure() -> None:
    out, _ = apply_figures("before\n\n\\bookfig{IMG_0002}{}\n\nafter", MAPPING, RECORDS)

    assert "IMG_0002" not in out and "bookfig" not in out
    assert out.startswith("before") and out.endswith("after")


def test_assembly_ignores_a_width_the_model_emitted() -> None:
    out, _ = apply_figures(r"\bookfig[0.85]{IMG_0000}{Hall effect}", MAPPING, RECORDS)

    assert out == r"\bookfigx{0.42}{IMG_0000}{18.3}{Hall effect}"


def test_assembly_sizes_a_pair_to_fit_and_shares_its_number() -> None:
    out, problems = apply_figures(
        r"\bookfigtwo{IMG_0003}{(a)}{IMG_0004}{(b)}{Focusing}", MAPPING, RECORDS
    )

    # 0.6 + 0.6 is over the 0.96 budget, so both scale to 0.48.
    assert out == r"\bookfigtwox{0.48}{IMG_0003}{(a)}{0.48}{IMG_0004}{(b)}{18.4}{Focusing}"
    assert problems == []


def test_assembly_keeps_a_pair_with_one_dropped_panel_and_reports_it() -> None:
    out, problems = apply_figures(
        r"\bookfigtwo{IMG_0002}{(a)}{IMG_0000}{(b)}{Pair}", MAPPING, RECORDS
    )

    assert out.startswith(r"\bookfigtwox{") and "IMG_0002" in out and "IMG_0000" in out
    assert len(problems) == 1 and "IMG_0002" in problems[0]


def test_assembly_reports_a_figure_with_no_record() -> None:
    out, problems = apply_figures(r"\bookfig{IMG_0000}{x}", MAPPING, {})

    assert out == r"\bookfigx{0.60}{IMG_0000}{}{x}"
    assert problems and "no figures.json record" in problems[0]


def _kit(tmp_path: Path, translation: str, records: dict[str, FigureRecord]) -> Path:
    kit = tmp_path / "kit"
    (kit / "chunks").mkdir(parents=True)
    (kit / "translated").mkdir()
    (kit / "images").mkdir()
    for path in MAPPING.values():
        (kit / path).write_bytes(_fake_png(path))
    tokens = " ".join(f"![]({t})" for t in MAPPING)
    (kit / "chunks" / "001.md").write_text(tokens, encoding="utf-8")
    (kit / "translated" / "001.tex").write_text(translation, encoding="utf-8")
    (kit / "image_map.json").write_text(json.dumps(MAPPING), encoding="utf-8")
    save_figures(kit / FIGURES_NAME, records)
    return kit


def test_assemble_applies_figures_json_then_restores_paths(tmp_path: Path) -> None:
    kit = _kit(
        tmp_path,
        "\\chapter{Magnetic Force}\n\n\\bookfig{IMG_0000}{Hall}\n\n\\bookfig{IMG_0002}{}\n",
        RECORDS,
    )

    path, problems = assemble(kit)
    body = path.read_text(encoding="utf-8").split("\\begin{document}")[1]

    assert problems == []
    assert r"\bookfigx{0.42}{images/fig.jpg}{18.3}{Hall}" in body
    assert "gone.jpg" not in body


# ── Lint ────────────────────────────────────────────────────────────────────


def _figure_findings(kit: Path):
    return [f for f in lint_kit(kit).findings if f.check in ("figures", "figure-numbers")]


def test_lint_reports_a_figure_with_no_record(tmp_path: Path) -> None:
    kit = _kit(tmp_path, r"\bookfig{IMG_0000}{Hall}", {})

    (finding,) = _figure_findings(kit)
    assert finding.severity is Severity.REVIEW
    assert "IMG_0000" in finding.context


def test_lint_notes_dropped_tokens_and_numbering_as_info(tmp_path: Path) -> None:
    records = dict(RECORDS)
    records["plain.jpg"] = FigureRecord(number="18.3", by=BY_CAPTION)  # a duplicate
    kit = _kit(
        tmp_path,
        "\\bookfig{IMG_0000}{a} \\bookfig{IMG_0001}{b} \\bookfig{IMG_0002}{}"
        " \\bookfigtwo{IMG_0003}{(a)}{IMG_0004}{(b)}{c}",
        records,
    )

    findings = {f.message.split(" ", 1)[1][:30]: f for f in _figure_findings(kit)}

    assert all(f.severity is Severity.INFO for f in findings.values())
    messages = " | ".join(f"{f.message} {f.context}" for f in findings.values())
    assert "dropped image(s) still appear" in messages and "IMG_0002" in messages
    assert "used twice" in messages and "18.3" in messages
    # 18.1 and 18.2 are missing below 18.3 and 18.4.
    assert "missing" in messages and "18.1, 18.2" in messages


# ── Linking a re-parse ──────────────────────────────────────────────────────


def _draw(path: Path, shapes: list[tuple[float, float, float, float]], size=(240, 180)) -> None:
    """A real PNG of black rectangles on white, rendered by pymupdf."""
    doc = pymupdf.open()
    page = doc.new_page(width=120, height=90)
    for rect in shapes:
        page.draw_rect(pymupdf.Rect(*rect), color=(0, 0, 0), fill=(0, 0, 0))
    zoom = size[0] / 120
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(str(path))
    doc.close()


SHAPES = {
    "one": [(10, 10, 40, 80), (60, 30, 110, 50)],
    "two": [(5, 60, 115, 85)],
    "three": [(80, 5, 100, 85), (10, 40, 30, 45)],
}


def _reparse(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    old = tmp_path / "old" / "images"
    new = tmp_path / "new" / "images"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    _draw(old / "same.jpg", SHAPES["one"])
    _draw(new / "same.jpg", SHAPES["one"])
    _draw(old / "renamed_old.jpg", SHAPES["two"])
    _draw(new / "renamed_new.jpg", SHAPES["two"], size=(300, 225))  # re-cropped larger
    _draw(old / "orphan.jpg", SHAPES["three"])
    text = [{"type": "text", "text": BODY, "bbox": [100, 100, 900, 150], "page_idx": 0}]
    images = [
        {"type": "image", "img_path": "images/same.jpg", "bbox": [100, 200, 500, 400]},
        {"type": "image", "img_path": "images/renamed_new.jpg", "bbox": [100, 500, 340, 700]},
    ]
    return old, new, text + images


def test_link_matches_exact_names_then_difference_hashes(tmp_path: Path) -> None:
    old, new, entries = _reparse(tmp_path)

    result = link_images(["same.jpg", "renamed_old.jpg", "orphan.jpg"], old, entries, new)

    assert result.matches["same.jpg"] == ("exact", "same.jpg")
    assert result.matches["renamed_old.jpg"] == ("dhash", "renamed_new.jpg")
    assert result.unmatched == ["orphan.jpg"]
    assert result.counts() == {"exact": 1, "dhash": 1, "unmatched": 1}


def test_linked_widths_size_matches_and_fall_back_for_the_rest(tmp_path: Path) -> None:
    old, new, entries = _reparse(tmp_path)
    result = link_images(["same.jpg", "renamed_old.jpg", "orphan.jpg"], old, entries, new)
    records = {
        "same.jpg": FigureRecord(number="1.1", flags=["no-bbox"]),
        "renamed_old.jpg": FigureRecord(number="1.2", flags=["no-bbox"]),
        "orphan.jpg": FigureRecord(number="1.3", flags=["no-bbox"]),
        "mine.jpg": FigureRecord(width=0.77, by=BY_MANUAL),
    }

    measured = apply_linked_widths(records, linked_entries(entries, result))

    assert measured == 2
    assert records["same.jpg"].width == 0.5
    assert records["same.jpg"].by == BY_CAPTION
    assert records["renamed_old.jpg"].width == 0.3
    assert records["orphan.jpg"].width == DEFAULT_WIDTH
    assert records["orphan.jpg"].by == BY_DEFAULT
    assert records["mine.jpg"].width == 0.77
    review = write_review(tmp_path, records).read_text(encoding="utf-8")
    assert "orphan.jpg" in review and "same.jpg" not in review


# ── Compiles as specified ───────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_pdflatex_prints_source_numbers_and_never_the_counter(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _draw(images / "fig.png", SHAPES["one"])
    body = (
        "\\setcounter{chapter}{17}\n\\chapter{Magnetic Force}\n\n"
        "\\bookfigx{0.30}{fig.png}{}{}\n\n"
        "\\bookfigx{0.42}{fig.png}{18.3}{Hall effect with $E_{x}$}\n\n"
        "\\bookfigtwox{0.40}{fig.png}{(a)}{0.50}{fig.png}{(b)}{18.5}{A pair}\n"
    )
    tex = assemble_document([body], tmp_path / "book.tex")

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:]

    with pymupdf.open(tmp_path / "book.pdf") as pdf:
        text = " ".join(page.get_text() for page in pdf)
    assert "Figure 18.3" in text
    assert "Figure 18.5" in text
    assert "Figure 18.1" not in text
    assert "Figure 18.2" not in text
