"""Tests for src.profiler's heading classification.

The parser gets levels 1 and 2 right but flattens everything below: on the
test book, 20 headings at `#` and 422 at `##`, with sections, subsections and
worked examples all landing on `##`. Classification is an LLM call because
heading conventions vary between books -- but *applying* the result is pure,
and that is where the interesting behaviour lives, so it is tested without a
provider.

No test here makes a real API call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.base import BaseLLM
from src.profiler import (
    BookProfile,
    apply_heading_levels,
    classify_headings,
    extract_headings,
    profile_book,
)

# The real shape: sections, subsections and examples all flattened onto `##`.
FLAT_BOOK = (
    "# 第3篇 电磁学\n\n"
    "## 第12章 电场\n\n"
    "## 12.1 电荷\n\n"
    "正文一。\n\n"
    "## 1. 电荷的种类\n\n"
    "正文二。\n\n"
    "## 例12.3\n\n"
    "解：正文三。\n\n"
    "## 提要\n\n"
    "本章要点。\n"
)

CLASSIFICATION = {
    "heading_levels": {
        "第3篇 电磁学": 1,
        "第12章 电场": 1,
        "12.1 电荷": 2,
        "1. 电荷的种类": 3,
        "例12.3": 4,
        "提要": 2,
    },
    "non_headings": [],
    "fragments": [],
}


class FakeLLM(BaseLLM):
    """Returns ``response`` (or raises it, if it is an exception)."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        if isinstance(self.response, Exception):
            raise self.response
        return str(self.response)


class ScriptedLLM(BaseLLM):
    """Answers each call from ``responses`` in order, repeating the last."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def _profile(**kwargs: object) -> BookProfile:
    return BookProfile(**kwargs)  # type: ignore[arg-type]


# ── Extraction ──────────────────────────────────────────────────────────────


def test_extract_headings_returns_distinct_strings_in_order() -> None:
    assert extract_headings(FLAT_BOOK) == [
        "第3篇 电磁学",
        "第12章 电场",
        "12.1 电荷",
        "1. 电荷的种类",
        "例12.3",
        "提要",
    ]


def test_repeated_headings_are_sent_once() -> None:
    """`提要` and `习题` recur once per chapter; 19 copies each buys nothing."""
    text = "## 提要\n\na\n\n## 习题\n\nb\n\n## 提要\n\nc\n\n## 习题\n\nd\n"
    assert extract_headings(text) == ["提要", "习题"]


def test_the_whole_heading_list_is_a_small_prompt() -> None:
    """~4,000 characters for a 500-page book, about 0.5% of it."""
    llm = FakeLLM(json.dumps(CLASSIFICATION))
    classify_headings(FLAT_BOOK, llm)

    _, user_text, _ = llm.calls[0]
    assert len(user_text) < len(FLAT_BOOK)
    assert user_text.splitlines() == extract_headings(FLAT_BOOK)


# ── Application (pure, no LLM) ──────────────────────────────────────────────


def test_flat_input_plus_a_classification_produces_correct_levels() -> None:
    profile = _profile(heading_levels=tuple(CLASSIFICATION["heading_levels"].items()))
    result, counts = apply_heading_levels(FLAT_BOOK, profile)

    assert "# 第3篇 电磁学" in result
    assert "# 第12章 电场" in result
    assert "## 12.1 电荷" in result
    assert "### 1. 电荷的种类" in result
    assert "#### 例12.3" in result
    assert "## 提要" in result
    assert counts["relevelled"] == 3  # 第12章, 1. 电荷的种类, 例12.3
    assert counts["demoted"] == 0


def test_body_text_is_demoted_to_a_paragraph() -> None:
    text = "## 12.1 电荷\n\n## 电荷 $\\rightarrow$ 电荷\n\n正文。\n"
    profile = _profile(
        heading_levels=(("12.1 电荷", 2),),
        non_headings=("电荷 $\\rightarrow$ 电荷",),
    )
    result, counts = apply_heading_levels(text, profile)

    assert "## 电荷 $\\rightarrow$ 电荷" not in result
    assert "电荷 $\\rightarrow$ 电荷" in result
    assert counts["demoted"] == 1


def test_an_unclassified_heading_keeps_its_parsed_level() -> None:
    """Never guess: a heading the profile says nothing about is left alone."""
    profile = _profile(heading_levels=(("12.1 电荷", 3),))
    result, counts = apply_heading_levels(FLAT_BOOK, profile)

    assert "### 12.1 电荷" in result
    assert "## 例12.3" in result  # untouched
    assert counts["unchanged"] == 5


def test_an_empty_profile_changes_nothing() -> None:
    result, counts = apply_heading_levels(FLAT_BOOK, BookProfile.generic())

    assert result == FLAT_BOOK
    assert counts["relevelled"] == 0


def test_application_is_idempotent() -> None:
    profile = _profile(heading_levels=tuple(CLASSIFICATION["heading_levels"].items()))
    once, _ = apply_heading_levels(FLAT_BOOK, profile)
    twice, counts = apply_heading_levels(once, profile)

    assert twice == once
    assert counts["relevelled"] == 0


def test_body_text_is_never_touched() -> None:
    profile = _profile(heading_levels=(("12.1 电荷", 3),))
    result, _ = apply_heading_levels(FLAT_BOOK, profile)

    assert "正文一。" in result
    assert "解：正文三。" in result


# ── Classification call ─────────────────────────────────────────────────────


def test_classification_parses_a_clean_response() -> None:
    levels, non_headings = classify_headings(FLAT_BOOK, FakeLLM(json.dumps(CLASSIFICATION)))

    assert dict(levels) == CLASSIFICATION["heading_levels"]
    assert non_headings == ()


def test_classification_tolerates_code_fences_and_preamble() -> None:
    raw = f"Here you go:\n```json\n{json.dumps(CLASSIFICATION)}\n```"
    levels, _ = classify_headings(FLAT_BOOK, FakeLLM(raw))

    assert dict(levels) == CLASSIFICATION["heading_levels"]


def test_classification_accepts_a_list_of_objects() -> None:
    payload = {
        "heading_levels": [
            {"heading": "12.1 电荷", "level": 2},
            {"heading": "例12.3", "level": 4},
        ]
    }
    levels, _ = classify_headings(FLAT_BOOK, FakeLLM(json.dumps(payload)))

    assert dict(levels) == {"12.1 电荷": 2, "例12.3": 4}


def test_hallucinated_headings_are_dropped() -> None:
    """A heading string that is not in the document cannot be applied."""
    payload = {"heading_levels": {"12.1 电荷": 2, "A Heading Not In The Book": 1}}
    levels, _ = classify_headings(FLAT_BOOK, FakeLLM(json.dumps(payload)))

    assert dict(levels) == {"12.1 电荷": 2}


def test_out_of_range_levels_are_dropped_not_clamped() -> None:
    payload = {"heading_levels": {"12.1 电荷": 2, "例12.3": 99, "提要": "two"}}
    levels, _ = classify_headings(FLAT_BOOK, FakeLLM(json.dumps(payload)))

    assert dict(levels) == {"12.1 电荷": 2}


def test_reported_fragments_are_logged_not_merged(caplog) -> None:
    payload = {"heading_levels": {"提要": 2}, "fragments": ["提要"]}
    levels, _ = classify_headings(FLAT_BOOK, FakeLLM(json.dumps(payload)))

    assert dict(levels) == {"提要": 2}  # still classified, never joined
    assert any("fragment" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "response",
    [RuntimeError("provider is down"), "not json at all", json.dumps(["a", "list"])],
)
def test_any_failure_degrades_to_leaving_headings_alone(response: object, caplog) -> None:
    levels, non_headings = classify_headings(FLAT_BOOK, FakeLLM(response))

    assert levels == ()
    assert non_headings == ()
    assert any(r.levelno >= 30 for r in caplog.records)
    # And the conservative default is a genuine no-op.
    assert apply_heading_levels(FLAT_BOOK, BookProfile.generic())[0] == FLAT_BOOK


# ── Round trip through the cached profile ───────────────────────────────────


def test_profile_book_caches_heading_levels_and_they_survive_reload(
    tmp_path: Path,
) -> None:
    merged = tmp_path / "normalized.md"
    merged.write_text(FLAT_BOOK, encoding="utf-8")

    book_profile = json.dumps({"subject": "physics", "glossary": []})
    llm = ScriptedLLM([book_profile, json.dumps(CLASSIFICATION)])

    profile = profile_book(merged, tmp_path, llm=llm)

    assert len(llm.calls) == 2  # one for the book, one for the headings
    assert dict(profile.heading_levels) == CLASSIFICATION["heading_levels"]

    # The cache is hand-editable JSON, and reloads without losing the levels.
    reloaded = profile_book(merged, tmp_path, llm=llm)
    assert len(llm.calls) == 2  # cached: no further calls
    assert dict(reloaded.heading_levels) == CLASSIFICATION["heading_levels"]


def test_a_heading_failure_does_not_cost_the_glossary(tmp_path: Path) -> None:
    """The two calls are isolated: the glossary took a full book read."""
    merged = tmp_path / "normalized.md"
    merged.write_text(FLAT_BOOK, encoding="utf-8")

    book_profile = json.dumps(
        {"subject": "physics", "glossary": [{"source": "电荷", "target": "electric charge"}]}
    )
    llm = ScriptedLLM([book_profile, "the heading call returned garbage"])

    profile = profile_book(merged, tmp_path, llm=llm)

    assert profile.subject == "physics"
    assert profile.glossary == (("电荷", "electric charge"),)
    assert profile.heading_levels == ()


def test_with_headings_false_makes_only_one_call(tmp_path: Path) -> None:
    merged = tmp_path / "normalized.md"
    merged.write_text(FLAT_BOOK, encoding="utf-8")
    llm = ScriptedLLM([json.dumps({"subject": "physics"})])

    profile = profile_book(merged, tmp_path, llm=llm, with_headings=False)

    assert len(llm.calls) == 1
    assert profile.heading_levels == ()
