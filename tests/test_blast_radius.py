"""Tests for the blast radius signal, written before the implementation.

Pins the API surface:
- BlastRadiusSignal(estimator=None).score(action) -> SignalResult
- Parsing order: list length > count-like key > wildcard-ambiguous > scalar-implies-one
- BlastRadiusEstimator protocol, injectable, with HeuristicBlastRadiusEstimator
  as the M2 stand-in for a future LLM estimate (fallback score 0.6)
"""

import asyncio

import pytest

from threshold.signals.base import Signal
from threshold.signals.blast_radius import BlastRadiusSignal, HeuristicBlastRadiusEstimator
from threshold.types import ProposedAction


def make_action(arguments: dict, tool_name: str = "some_tool") -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments=arguments, agent_reasoning="test")


def run(coro):
    return asyncio.run(coro)


class StubEstimator:
    """Test double: records what it was asked, returns a fixed score."""

    def __init__(self, score: float):
        self.score = score
        self.calls: list[tuple[str, dict]] = []

    async def estimate(self, tool_name: str, arguments: dict) -> float:
        self.calls.append((tool_name, arguments))
        return self.score


# --- parsing: list length ---

def test_list_argument_length_becomes_the_count():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"recipients": ["a@x.com", "b@x.com", "c@x.com"]})))
    assert result.score == pytest.approx(0.3)  # count 3 -> bucket [2,9]


def test_longest_list_wins_when_multiple_lists_present():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"to": ["a@x.com"], "cc": [f"u{i}@x.com" for i in range(15)]})))
    assert result.score == pytest.approx(0.6)  # count 15 -> bucket [10,99]


# --- parsing: count-like key ---

def test_count_like_key_becomes_the_count():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"symbol": "ACME", "quantity": 500}, tool_name="execute_trade")))
    assert result.score == pytest.approx(1.0)  # count 500 -> bucket [100, inf)


def test_list_takes_priority_over_a_coincidentally_present_count_key():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"ids": [1, 2, 3], "count": 3})))
    assert result.score == pytest.approx(0.3)  # both agree here, but list must be the one consulted


# --- parsing: scalar-only implies exactly one entity ---

def test_single_scalar_argument_implies_count_of_one():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"path": "/tmp/x"})))
    assert result.score == pytest.approx(0.0)


# --- parsing: wildcard strings are ambiguous, not a count of one ---

def test_wildcard_string_argument_is_ambiguous_not_a_single_entity():
    fallback = StubEstimator(score=0.6)
    signal = BlastRadiusSignal(estimator=fallback)
    result = run(signal.score(make_action({"path": "/data/*.csv"})))
    assert fallback.calls == [("some_tool", {"path": "/data/*.csv"})]
    assert result.score == pytest.approx(0.6)


# --- parsing: empty arguments fall back ---

def test_empty_arguments_fall_back_to_estimator():
    fallback = StubEstimator(score=0.6)
    signal = BlastRadiusSignal(estimator=fallback)
    result = run(signal.score(make_action({})))
    assert fallback.calls == [("some_tool", {})]


# --- bucket boundaries ---

BUCKET_CASES = [
    (1, 0.0),
    (2, 0.3),
    (9, 0.3),
    (10, 0.6),
    (99, 0.6),
    (100, 1.0),
]


@pytest.mark.parametrize("count,expected_score", BUCKET_CASES)
def test_bucket_boundaries(count, expected_score):
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"ids": list(range(count))})))
    assert result.score == pytest.approx(expected_score)


# --- default heuristic estimator ---

def test_default_heuristic_estimator_returns_point_six():
    estimator = HeuristicBlastRadiusEstimator()
    score = run(estimator.estimate("anything", {}))
    assert score == pytest.approx(0.6)


def test_signal_uses_heuristic_estimator_by_default():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({})))  # empty args -> must fall back
    assert result.score == pytest.approx(0.6)


# --- shape/metadata ---

def test_blast_radius_signal_is_a_signal():
    assert isinstance(BlastRadiusSignal(), Signal)


def test_signal_result_has_expected_name_and_nonnegative_cost():
    signal = BlastRadiusSignal()
    result = run(signal.score(make_action({"path": "/tmp/x"})))
    assert result.name == "blast_radius"
    assert result.cost_ms >= 0
