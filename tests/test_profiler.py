"""Tests for src.profiler (book profiling).

No real API calls: the LLM is always a fake that returns scripted JSON.
The LaTeX assets, scanning and assembly live in tests/test_latex.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.base import BaseLLM
from src.profiler import (
    PROFILE_FILENAME,
    BookProfile,
    profile_book,
    profile_to_prompt_block,
)


class FakeLLM(BaseLLM):
    """Returns ``response`` (or raises it, if it is an exception) and counts calls."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        self.calls.append((system_prompt, user_text, temperature))
        if isinstance(self.response, Exception):
            raise self.response
        return str(self.response)

    @property
    def call_count(self) -> int:
        return len(self.calls)


VALID_PROFILE_JSON = json.dumps(
    {
        "subject": "mathematics",
        "subfield": "real analysis",
        "education_level": "undergraduate",
        "audience": "first-year mathematics students",
        "register": "formal academic prose",
        "notation_notes": "Uses $\\varepsilon$-$\\delta$ definitions throughout.",
        "structural_elements": ["definition", "theorem", "proof", "exercise"],
        "glossary": [
            {"source": "导数", "target": "derivative"},
            {"source": "极限", "target": "limit"},
        ],
        "latex_documentclass": "book",
        "latex_packages": ["amsthm"],
        "summary": "An introduction to real analysis.",
    },
    ensure_ascii=False,
)


def _write_book(tmp_path: Path, text: str = "# 第一章\n\n本章介绍导数的定义。\n") -> Path:
    merged = tmp_path / "merged.md"
    merged.write_text(text, encoding="utf-8")
    return merged


# ── BookProfile.from_dict ───────────────────────────────────────────────────


def test_from_dict_parses_object_style_glossary() -> None:
    profile = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    assert profile.subject == "mathematics"
    assert profile.glossary == (("导数", "derivative"), ("极限", "limit"))
    assert profile.structural_elements == ("definition", "theorem", "proof", "exercise")
    assert profile.latex_packages == ("amsthm",)


def test_from_dict_parses_list_style_glossary() -> None:
    profile = BookProfile.from_dict({"glossary": [["导数", "derivative"], ["极限", "limit"]]})

    assert profile.glossary == (("导数", "derivative"), ("极限", "limit"))


def test_from_dict_skips_junk_glossary_entries() -> None:
    profile = BookProfile.from_dict(
        {
            "glossary": [
                {"source": "导数", "target": "derivative"},
                {"source": "极限"},  # missing target
                ["only-one-element"],
                ["a", "b", "c"],
                "not an entry at all",
                {"source": "", "target": "empty source"},
                42,
                None,
                ["积分", "integral"],
            ]
        }
    )

    assert profile.glossary == (("导数", "derivative"), ("积分", "integral"))


def test_from_dict_tolerates_missing_keys_and_wrong_types() -> None:
    profile = BookProfile.from_dict({"subject": "physics", "glossary": "not a list"})

    assert profile.subject == "physics"
    assert profile.glossary == ()
    assert profile.education_level == BookProfile.generic().education_level
    assert profile.latex_documentclass == "book"


def test_from_dict_coerces_a_bare_string_to_a_one_tuple() -> None:
    profile = BookProfile.from_dict(
        {"latex_packages": "amsthm", "structural_elements": "theorem"}
    )

    assert profile.latex_packages == ("amsthm",)
    assert profile.structural_elements == ("theorem",)


def test_from_dict_of_non_dict_is_generic() -> None:
    assert BookProfile.from_dict(["nope"]) == BookProfile.generic()  # type: ignore[arg-type]


def test_to_json_round_trips_through_from_dict() -> None:
    original = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    assert BookProfile.from_dict(json.loads(original.to_json())) == original


# ── profile_book ────────────────────────────────────────────────────────────


def test_profile_book_parses_and_caches(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    llm = FakeLLM(VALID_PROFILE_JSON)

    profile = profile_book(merged, tmp_path, llm=llm)

    assert profile.subject == "mathematics"
    # Two calls: one to characterize the book, one to classify its heading
    # levels. Both are cached in the same book_profile.json. See
    # tests/test_heading_classification.py for the second one's behaviour.
    assert llm.call_count == 2
    assert (tmp_path / PROFILE_FILENAME).exists()


def test_second_call_uses_the_cache_without_hitting_the_llm(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    second_llm = FakeLLM(VALID_PROFILE_JSON)
    cached = profile_book(merged, tmp_path, llm=second_llm)

    assert second_llm.call_count == 0
    assert cached.glossary == (("导数", "derivative"), ("极限", "limit"))


def test_force_re_profiles_despite_the_cache(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    second_llm = FakeLLM(VALID_PROFILE_JSON)
    profile_book(merged, tmp_path, llm=second_llm, force=True)

    # Both the book call and the heading-classification call are re-made.
    assert second_llm.call_count == 2


def test_fenced_and_prefixed_json_is_still_parsed(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    llm = FakeLLM(f"Here is the profile:\n```json\n{VALID_PROFILE_JSON}\n```\nHope that helps!")

    profile = profile_book(merged, tmp_path, llm=llm)

    assert profile.subject == "mathematics"


def test_unparseable_json_falls_back_to_generic_and_is_not_cached(
    tmp_path: Path, caplog
) -> None:
    merged = _write_book(tmp_path)

    with caplog.at_level("WARNING"):
        profile = profile_book(merged, tmp_path, llm=FakeLLM("I'm afraid I can't do that."))

    assert profile == BookProfile.generic()
    assert not (tmp_path / PROFILE_FILENAME).exists()
    assert any("profiling failed" in record.message.lower() for record in caplog.records)


def test_raising_llm_falls_back_to_generic_without_raising(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)

    profile = profile_book(merged, tmp_path, llm=FakeLLM(RuntimeError("API down")))

    assert profile == BookProfile.generic()
    assert not (tmp_path / PROFILE_FILENAME).exists()


def test_missing_source_file_falls_back_to_generic(tmp_path: Path) -> None:
    profile = profile_book(tmp_path / "nope.md", tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    assert profile == BookProfile.generic()


def test_a_failed_profile_does_not_poison_the_next_run(tmp_path: Path) -> None:
    merged = _write_book(tmp_path)
    profile_book(merged, tmp_path, llm=FakeLLM(RuntimeError("API down")))

    recovered = profile_book(merged, tmp_path, llm=FakeLLM(VALID_PROFILE_JSON))

    assert recovered.subject == "mathematics"


def test_large_books_are_sampled_not_sent_whole(tmp_path: Path) -> None:
    body = "\n\n".join(f"## 第{i}节\n\n本节内容重复出现。" * 40 for i in range(200))
    merged = _write_book(tmp_path, body)
    llm = FakeLLM(VALID_PROFILE_JSON)

    profile_book(merged, tmp_path, llm=llm)

    sample = llm.calls[0][1]
    assert len(sample) < len(body)
    assert len(sample) <= 30_000


# ── profile_to_prompt_block ─────────────────────────────────────────────────


def test_prompt_block_lists_required_terminology() -> None:
    profile = BookProfile.from_dict(json.loads(VALID_PROFILE_JSON))

    block = profile_to_prompt_block(profile)

    assert "Required terminology" in block
    assert "导数 -> derivative" in block
    assert "极限 -> limit" in block
    assert "undergraduate" in block


def test_prompt_block_caps_the_glossary_and_stays_compact() -> None:
    profile = BookProfile(glossary=tuple((f"术语{i}", f"term{i}") for i in range(100)))

    block = profile_to_prompt_block(profile, max_glossary=5)

    assert "术语4 -> term4" in block
    assert "术语5 -> term5" not in block
    assert len(block.splitlines()) < 15


def test_prompt_block_of_a_generic_profile_has_no_terminology_section() -> None:
    block = profile_to_prompt_block(BookProfile.generic())

    assert "Required terminology" not in block
    assert "BOOK CONTEXT" in block
