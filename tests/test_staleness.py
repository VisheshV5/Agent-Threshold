"""Tests for the staleness signal, written before the implementation.

Pins the API surface:
- StalenessSignal().score(action) -> SignalResult, no constructor args --
  unlike the other signals, this one is a pure deterministic function of
  ProposedAction.context_age_steps and needs no injectable fallback
- Empty context_age_steps -> score 0.0 ("nothing depends on prior
  context"), NOT a cautious fallback -- this data is already present on
  every ProposedAction, not something that had to be actively gathered
- Multiple tracked facts -> uses the single MOST STALE one (max steps),
  not an average
- Bucket boundaries on "steps since last verified"
"""

import asyncio

import pytest

from escalation.signals.base import Signal
from escalation.signals.staleness import StalenessSignal
from escalation.types import ProposedAction


def make_action(context_age_steps: dict[str, int] | None = None) -> ProposedAction:
    return ProposedAction(
        tool_name="some_tool",
        arguments={},
        agent_reasoning="test",
        context_age_steps=context_age_steps or {},
    )


def run(coro):
    return asyncio.run(coro)


def test_empty_context_age_steps_is_not_stale():
    signal = StalenessSignal()
    result = run(signal.score(make_action()))
    assert result.score == pytest.approx(0.0)
    assert "no tracked facts" in result.reason.lower()


BUCKET_CASES = [
    (0, 0.0),
    (1, 0.0),
    (2, 0.3),
    (4, 0.3),
    (5, 0.6),
    (14, 0.6),
    (15, 1.0),
]


@pytest.mark.parametrize("steps,expected_score", BUCKET_CASES)
def test_bucket_boundaries(steps, expected_score):
    signal = StalenessSignal()
    result = run(signal.score(make_action({"account_balance": steps})))
    assert result.score == pytest.approx(expected_score)


def test_uses_the_most_stale_fact_not_average_or_min():
    signal = StalenessSignal()
    result = run(signal.score(make_action({"user_role": 1, "account_balance": 20})))
    assert result.score == pytest.approx(1.0)  # driven by the stalest (20), not the freshest (1)
    assert "account_balance" in result.reason


def test_signal_is_a_signal_with_expected_name_and_cost():
    signal = StalenessSignal()
    result = run(signal.score(make_action({"x": 3})))
    assert isinstance(signal, Signal)
    assert result.name == "staleness"
    assert result.cost_ms >= 0
