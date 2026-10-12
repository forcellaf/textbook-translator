"""Tests for src.footnotes: page_footnote blocks put back into merged.md.

The content list is synthetic but shaped like the VLM backend's (normalized
0-1000 bboxes, global ``page_idx``), and covers the three kinds physics_2
lost: a real footnote at its marker, body text mis-tagged as a footnote, and
a footnote derivation whose display equation was stranded in the body.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.footnotes import missing_footnote_blocks, restore_footnotes, split_markers


def _e(kind: str, page: int, top: int, **extra: object) -> dict:
    return {"type": kind, "bbox": [80, top, 920, top + 40], "page_idx": page, **extra}


ENTRIES = [
    # Page 0: a footnote at its marker, and an example statement mis-tagged
    # as a footnote, with body text below it.
    _e("header", 0, 30, text="第13章 电势"),
    _e("text", 0, 80, text="13.1 电势", text_level=1),
    _e("text", 0, 120, text="电场力做功与路径无关①，因此可以定义电势能。"),
    _e("equation", 0, 200, text="$$\nW = q U\n$$", text_format="latex"),
    _e("text", 0, 300, text="例题前的正文段落，说明下面的例题。"),
    _e("page_footnote", 0, 400, text="例 13.3 一个点电荷 $q$ 位于原点，求距离 $r$ 处的电势。"),
    _e("text", 0, 500, text="解：由库仑定律可得电势的表达式。"),
    _e("page_footnote", 0, 850, text="① 这一结论对任何静电场都成立。"),
    _e("page_number", 0, 950, text="12"),
    # Page 1: a footnote derivation whose equation stayed in the body.
    _e("text", 1, 100, text="第二页正文，引入势能$^{②}$的概念并给出结论。"),
    _e("page_footnote", 1, 780, text="②势能的推导如下："),
    _e("equation", 1, 830, text="$$\nU = - \\int E \\cdot d l\n$$", text_format="latex"),
    _e("page_footnote", 1, 900, text="其中积分沿任意路径。"),
    _e("page_number", 1, 960, text="13"),
    # Page 2: two footnotes in one block; "由①式" refers to an equation.
    _e("text", 2, 100, text="第三页正文，含有两个脚注③和④。"),
    _e("page_footnote", 2, 860, text="③第一个脚注。④第二个脚注，由①式可得。"),
]

# What full.md gives: every block above except header, page number and
# page_footnote.
MERGED = """# 13.1 电势

电场力做功与路径无关①，因此可以定义电势能。

$$
W = q U
$$

例题前的正文段落，说明下面的例题。

解：由库仑定律可得电势的表达式。

第二页正文，引入势能$^{②}$的概念并给出结论。

$$
U = - \\int E \\cdot d l
$$

第三页正文，含有两个脚注③和④。
"""

RESTORED = """# 13.1 电势

电场力做功与路径无关①，因此可以定义电势能。

① 这一结论对任何静电场都成立。

$$
W = q U
$$

例题前的正文段落，说明下面的例题。

例 13.3 一个点电荷 $q$ 位于原点，求距离 $r$ 处的电势。

解：由库仑定律可得电势的表达式。

第二页正文，引入势能$^{②}$的概念并给出结论。

② 势能的推导如下：
$$
U = - \\int E \\cdot d l
$$
其中积分沿任意路径。

第三页正文，含有两个脚注③和④。

③ 第一个脚注。

④ 第二个脚注，由①式可得。
"""


def test_footnotes_are_restored_at_their_markers() -> None:
    text, report = restore_footnotes(MERGED, ENTRIES)

    assert text == RESTORED
    assert report.footnote_blocks == 5
    assert report.at_marker == 4
    assert report.unmarked == 1
    assert report.moved_into_footnotes == 1
    assert report.marker_not_found == 0


def test_restoring_twice_changes_nothing() -> None:
    once, _ = restore_footnotes(MERGED, ENTRIES)
    twice, report = restore_footnotes(once, ENTRIES)

    assert twice == once
    assert not report.changed
    assert report.already_present == 5


def test_a_footnote_whose_marker_is_gone_goes_after_the_body_above() -> None:
    entries = [
        _e("text", 0, 100, text="正文第一段，标记被删掉了。"),
        _e("text", 0, 300, text="正文第二段，也没有标记。"),
        _e("page_footnote", 0, 850, text="①孤立的脚注文字。"),
        _e("text", 1, 100, text="下一页的正文。"),
    ]
    merged = "正文第一段，标记被删掉了。\n\n正文第二段，也没有标记。\n\n下一页的正文。\n"

    text, report = restore_footnotes(merged, entries)

    assert text == (
        "正文第一段，标记被删掉了。\n\n正文第二段，也没有标记。\n\n"
        "① 孤立的脚注文字。\n\n下一页的正文。\n"
    )
    assert report.marker_not_found == 1


def test_without_footnotes_the_markdown_is_returned_untouched() -> None:
    merged = "# 标题\n\n\n正文。\n"
    text, report = restore_footnotes(merged, [_e("text", 0, 100, text="正文。")])

    assert text is merged
    assert report.footnote_blocks == 0


def test_split_markers() -> None:
    assert split_markers("③第一个脚注。④第二个，由①式可得。") == [
        (3, "第一个脚注。"),
        (4, "第二个，由①式可得。"),
    ]
    assert split_markers("续上页的文字。\n$^{①}$ 新脚注") == [
        (None, "续上页的文字。"),
        (1, "新脚注"),
    ]
    assert split_markers("\\textcircled{2} 另一种写法") == [(2, "另一种写法")]


def test_missing_footnote_blocks_counts_what_the_parse_lost() -> None:
    assert len(missing_footnote_blocks(MERGED, ENTRIES)) == 5
    assert missing_footnote_blocks(RESTORED, ENTRIES) == []


def test_script_reports_by_default_and_writes_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "scripts"))
    script = importlib.import_module("restore_footnotes")
    (tmp_path / "merged.md").write_text(MERGED, encoding="utf-8")
    (tmp_path / "content_list.json").write_text(json.dumps(ENTRIES), encoding="utf-8")

    assert script.main([str(tmp_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["page_footnote_blocks"] == 5
    assert summary["missing_from_merged"] == 5
    assert summary["missing_pages"] == [0, 1, 2]
    assert not summary["written"]
    assert (tmp_path / "merged.md").read_text(encoding="utf-8") == MERGED

    assert script.main([str(tmp_path), "--write"]) == 0
    assert (tmp_path / "merged.md").read_text(encoding="utf-8") == RESTORED
    assert (tmp_path / "merged.before_footnotes.md").read_text(encoding="utf-8") == MERGED
