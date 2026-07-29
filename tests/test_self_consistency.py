"""Tests for the self-consistency signal, written before the implementation.

Pins the API surface:
- SelfConsistencySignal(similarity_scorer=None).score(action) -> SignalResult
- Sample set is [action, *action.alternative_actions]; score = 1 - avg
  pairwise similarity across ALL pairs in that set (not just alternates
  vs. the primary action)
- No alternatives at all -> MISSING_DATA_SCORE (0.5), not 0.0 -- absence
  of data must not silently read as "checked, looks consistent"
- SimilarityScorer protocol, injectable, with HeuristicSimilarityScorer
  (tool_name equality + argument-overlap) as the M2 stand-in for a
  future embedding-based comparison
"""

import asyncio

import pytest

from threshold.signals.base import Signal
from threshold.signals.self_consistency import (
    HeuristicSimilarityScorer,
    SelfConsistencySignal,
)
from threshold.types import ProposedAction


def make_action(tool_name: str, arguments: dict | None = None, alternatives: list[ProposedAction] | None = None) -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments=arguments or {},
        agent_reasoning="test",
        alternative_actions=alternatives or [],
    )


def run(coro):
    return asyncio.run(coro)


class StubSimilarityScorer:
    """Test double: records the pairs it was asked about, returns a fixed score."""

    def __init__(self, score: float):
        self.score = score
        self.calls: list[tuple[ProposedAction, ProposedAction]] = []

    async def similarity(self, a: ProposedAction, b: ProposedAction) -> float:
        self.calls.append((a, b))
        return self.score


# --- missing data ---

def test_no_alternatives_returns_missing_data_score_not_zero():
    signal = SelfConsistencySignal()
    action = make_action("read_file", {"path": "/x"})
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.5)
    assert "no alternative" in result.reason.lower()
    assert result.informative is False  # policy must exclude this from the weighted average


def test_real_computation_is_marked_informative():
    alt = make_action("read_file", {"path": "/x"})
    primary = make_action("read_file", {"path": "/x"}, alternatives=[alt])
    signal = SelfConsistencySignal()
    result = run(signal.score(primary))
    assert result.informative is True


# --- real computation with the default heuristic scorer ---

def test_single_identical_alternative_gives_full_agreement_zero_score():
    alt = make_action("read_file", {"path": "/x"})
    primary = make_action("read_file", {"path": "/x"}, alternatives=[alt])
    signal = SelfConsistencySignal()
    result = run(signal.score(primary))
    assert result.score == pytest.approx(0.0)


def test_single_alternative_with_different_tool_gives_full_disagreement():
    alt = make_action("delete_file", {"path": "/x"})
    primary = make_action("read_file", {"path": "/x"}, alternatives=[alt])
    signal = SelfConsistencySignal()
    result = run(signal.score(primary))
    assert result.score == pytest.approx(1.0)


def test_single_alternative_with_partial_argument_overlap():
    alt = make_action("write_file", {"a": 1, "b": 3})
    primary = make_action("write_file", {"a": 1, "b": 2}, alternatives=[alt])
    signal = SelfConsistencySignal()
    result = run(signal.score(primary))
    # pairs_a={("a",1),("b",2)}, pairs_b={("a",1),("b",3)} -> jaccard 1/3
    assert result.score == pytest.approx(2 / 3, abs=1e-6)


def test_multiple_alternatives_average_pairwise_across_the_full_sample_set():
    alt1 = make_action("read_file", {})  # agrees with primary
    alt2 = make_action("delete_file", {})  # disagrees with both primary and alt1
    primary = make_action("read_file", {}, alternatives=[alt1, alt2])
    signal = SelfConsistencySignal()
    result = run(signal.score(primary))
    # pairs: (primary,alt1)=1.0, (primary,alt2)=0.0, (alt1,alt2)=0.0 -> avg 1/3
    assert result.score == pytest.approx(1 - 1 / 3, abs=1e-6)


# --- injectable scorer (DI plumbing, independent of heuristic details) ---

def test_uses_injected_similarity_scorer_and_calls_it_for_every_pair():
    stub = StubSimilarityScorer(score=0.4)
    alt1 = make_action("read_file", {})
    alt2 = make_action("write_file", {})
    primary = make_action("read_file", {}, alternatives=[alt1, alt2])
    signal = SelfConsistencySignal(similarity_scorer=stub)
    result = run(signal.score(primary))
    assert len(stub.calls) == 3  # 3 samples -> C(3,2) = 3 pairs
    assert result.score == pytest.approx(1 - 0.4)


# --- default heuristic scorer, tested directly ---

def test_heuristic_scorer_identical_tool_and_args_is_one():
    scorer = HeuristicSimilarityScorer()
    a = make_action("read_file", {"path": "/x"})
    b = make_action("read_file", {"path": "/x"})
    assert run(scorer.similarity(a, b)) == pytest.approx(1.0)


def test_heuristic_scorer_different_tool_is_zero():
    scorer = HeuristicSimilarityScorer()
    a = make_action("read_file", {"path": "/x"})
    b = make_action("delete_file", {"path": "/x"})
    assert run(scorer.similarity(a, b)) == pytest.approx(0.0)


def test_heuristic_scorer_both_empty_args_is_one():
    scorer = HeuristicSimilarityScorer()
    a = make_action("read_file", {})
    b = make_action("read_file", {})
    assert run(scorer.similarity(a, b)) == pytest.approx(1.0)


def test_heuristic_scorer_empty_vs_nonempty_args_is_zero():
    scorer = HeuristicSimilarityScorer()
    a = make_action("read_file", {})
    b = make_action("read_file", {"path": "/x"})
    assert run(scorer.similarity(a, b)) == pytest.approx(0.0)


# --- shape/metadata ---

def test_self_consistency_signal_is_a_signal():
    assert isinstance(SelfConsistencySignal(), Signal)


def test_signal_result_has_expected_name_and_nonnegative_cost():
    signal = SelfConsistencySignal()
    result = run(signal.score(make_action("read_file", {})))
    assert result.name == "self_consistency"
    assert result.cost_ms >= 0
