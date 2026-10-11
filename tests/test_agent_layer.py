"""The agent layer's files match the decided model / effort / tool table.

The orchestrator skill and the subagents are configuration, not code, so a
typo there (a tool an agent must not have, the wrong model) would only show
up in a real, expensive run. These checks read the frontmatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / ".claude" / "agents"

EXPECTED = {
    "page-range-finder": ("claude-sonnet-5-5", "medium", {"Read", "Write"}),
    "stage2-preparer": ("claude-sonnet-5-5", "medium", {"Bash", "Read", "Write", "Grep", "Agent"}),
    "llm-answerer-sonnet": ("claude-sonnet-5-5", "medium", {"Read", "Write"}),
    "figure-classifier": ("claude-haiku-5-5", "low", {"Read", "Write"}),
    "chunk-translator": ("claude-haiku-5-5", "medium", {"Read", "Write"}),
    "output-digester": ("claude-sonnet-5-5", "medium", {"Read", "Grep"}),
}


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    block = text.split("---\n", 2)[1]
    return {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in block.splitlines() if ":" in line)
    }


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_agent_has_its_model_effort_and_only_its_tools(name: str) -> None:
    meta = _frontmatter(AGENTS / f"{name}.md")
    model, effort, tools = EXPECTED[name]
    assert meta["name"] == name
    assert meta["model"] == model
    assert meta["effort"] == effort
    assert {t.strip() for t in meta["tools"].split(",")} == tools


def test_the_skill_is_named_and_described() -> None:
    meta = _frontmatter(ROOT / ".claude" / "skills" / "translate-book" / "SKILL.md")
    assert meta["name"] == "translate-book"
    assert "/translate-book" in meta["description"]


def test_run_settings_deny_edits_to_the_tool_but_shared_settings_do_not() -> None:
    shared = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    overlay = json.loads((ROOT / ".claude" / "run-settings.json").read_text(encoding="utf-8"))
    assert "deny" not in shared["permissions"]  # dev sessions can still edit code
    for path in ("src", "assets", "tests"):
        assert f"Edit(./{path}/**)" in overlay["permissions"]["deny"]
    assert int(overlay["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]) >= 64_000
