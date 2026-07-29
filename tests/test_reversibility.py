"""Tests for the reversibility signal, written before the implementation.

Pins the API surface:
- ActionCategory: the four classification buckets from PROJECT.md
- HARD_OVERRIDE_CATEGORIES: the set {irreversible-write, external-effect}
  that Policy will later force to ask_human and exclude from the cache
- ReversibilitySignal(registry=..., fallback=...): static registry lookup,
  falling through to an injectable FallbackClassifier for unknown tools
- HeuristicFallbackClassifier: the M1 stand-in for the future LLM
  classifier (the Anthropic adapter doesn't land until M4)
"""

import asyncio

import pytest

from threshold.signals.base import Signal
from threshold.signals.reversibility import (
    HARD_OVERRIDE_CATEGORIES,
    ActionCategory,
    HeuristicFallbackClassifier,
    ReversibilitySignal,
)
from threshold.types import ProposedAction


def make_action(tool_name: str, arguments: dict | None = None) -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments=arguments or {},
        agent_reasoning="test",
    )


def run(coro):
    return asyncio.run(coro)


class StubFallbackClassifier:
    """Test double: records what it was asked, returns a fixed category."""

    def __init__(self, category: ActionCategory):
        self.category = category
        self.calls: list[tuple[str, dict]] = []

    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory:
        self.calls.append((tool_name, arguments))
        return self.category


# --- static registry: one representative tool per category ---

REGISTRY_CASES = [
    ("read_file", ActionCategory.READ_ONLY),
    ("write_file", ActionCategory.REVERSIBLE_WRITE),
    ("delete_file", ActionCategory.IRREVERSIBLE_WRITE),
    ("send_email", ActionCategory.EXTERNAL_EFFECT),
]


@pytest.mark.parametrize("tool_name,expected_category", REGISTRY_CASES)
def test_registry_hit_classifies_correctly(tool_name, expected_category):
    signal = ReversibilitySignal()
    category, source = run(signal.classify(make_action(tool_name)))
    assert category == expected_category
    assert source == "registry"


@pytest.mark.parametrize("tool_name,expected_category", REGISTRY_CASES)
def test_registry_hit_produces_expected_signal_result(tool_name, expected_category):
    signal = ReversibilitySignal()
    result = run(signal.score(make_action(tool_name)))
    assert result.name == "reversibility"
    assert 0.0 <= result.score <= 1.0
    assert result.cost_ms >= 0
    assert "registry" in result.reason
    assert expected_category.value in result.reason


def test_unknown_tool_falls_through_to_fallback_classifier():
    fallback = StubFallbackClassifier(ActionCategory.EXTERNAL_EFFECT)
    signal = ReversibilitySignal(fallback=fallback)
    action = make_action("some_tool_not_in_any_registry", {"x": 1})

    category, source = run(signal.classify(action))

    assert category == ActionCategory.EXTERNAL_EFFECT
    assert source == "fallback"
    assert fallback.calls == [("some_tool_not_in_any_registry", {"x": 1})]


def test_unknown_tool_signal_result_reflects_fallback_source():
    fallback = StubFallbackClassifier(ActionCategory.READ_ONLY)
    signal = ReversibilitySignal(fallback=fallback)
    result = run(signal.score(make_action("totally_novel_tool")))
    assert "fallback" in result.reason


# --- hard-override categories: what Policy will key off of ---

def test_hard_override_categories_are_exactly_irreversible_and_external():
    assert HARD_OVERRIDE_CATEGORIES == {
        ActionCategory.IRREVERSIBLE_WRITE,
        ActionCategory.EXTERNAL_EFFECT,
    }


@pytest.mark.parametrize(
    "tool_name",
    ["delete_file", "send_email"],
)
def test_hard_override_tools_classify_into_hard_override_set(tool_name):
    signal = ReversibilitySignal()
    category, _ = run(signal.classify(make_action(tool_name)))
    assert category in HARD_OVERRIDE_CATEGORIES


@pytest.mark.parametrize(
    "tool_name",
    ["read_file", "write_file"],
)
def test_non_hard_override_tools_are_excluded_from_hard_override_set(tool_name):
    signal = ReversibilitySignal()
    category, _ = run(signal.classify(make_action(tool_name)))
    assert category not in HARD_OVERRIDE_CATEGORIES


# --- default heuristic fallback (the M1 stand-in for an LLM classifier) ---

HEURISTIC_CASES = [
    ("get_weather_forecast", ActionCategory.READ_ONLY),
    ("send_slack_notification", ActionCategory.EXTERNAL_EFFECT),
    ("update_local_cache", ActionCategory.REVERSIBLE_WRITE),
    ("purge_old_sessions", ActionCategory.IRREVERSIBLE_WRITE),
    ("launch_the_thing", ActionCategory.IRREVERSIBLE_WRITE),  # no keyword match -> fail-safe default
    # "message" alone is not an EXTERNAL_EFFECT keyword, so this must not
    # false-positive to external-effect just because it mentions a message
    ("delete_message", ActionCategory.IRREVERSIBLE_WRITE),
    # whole-token matching: "widget" contains "get" as a substring but must
    # not match the READ_ONLY "get" keyword
    ("reset_widget_config", ActionCategory.REVERSIBLE_WRITE),
    # the reversible/irreversible boundary is NOT hard-override-protected
    # on the reversible side -- if a milder keyword ("edit") won over a
    # destructive one ("delete") also present, this would score 0.35 and
    # never force ask_human. Most-severe-match-wins must catch this.
    ("edit_and_delete_entry", ActionCategory.IRREVERSIBLE_WRITE),
    # same principle one severity level down: "update" must beat "get"
    ("get_and_update_settings", ActionCategory.REVERSIBLE_WRITE),
]


@pytest.mark.parametrize("tool_name,expected_category", HEURISTIC_CASES)
def test_heuristic_fallback_classifies_by_keyword(tool_name, expected_category):
    classifier = HeuristicFallbackClassifier()
    category = run(classifier.classify(tool_name, {}))
    assert category == expected_category


def test_reversibility_signal_is_a_signal():
    assert isinstance(ReversibilitySignal(), Signal)


def test_reversibility_signal_uses_heuristic_fallback_by_default():
    # no fallback injected -> falls back to HeuristicFallbackClassifier,
    # so an unknown but keyword-y tool name still classifies sensibly
    signal = ReversibilitySignal()
    category, source = run(signal.classify(make_action("delete_orphaned_records")))
    assert source == "fallback"
    assert category == ActionCategory.IRREVERSIBLE_WRITE
