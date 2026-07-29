"""Tests for the policy skeleton, written before the implementation.

Pins the API surface:
- Policy(reversibility=..., weights=..., threshold=..., extra_signals=None)
- Policy.from_profile("conservative"|"balanced"|"autonomous")
- Policy.evaluate(action) -> Decision, async (signals are async)
- Hard override: reversibility category in HARD_OVERRIDE_CATEGORIES always
  forces verdict == "ask_human", independent of aggregate_score vs threshold
- Weighted aggregation must fail loudly if a signal has no configured weight,
  not silently zero it out
"""

import asyncio

import pytest

from escalation.policy.policy import Policy
from escalation.policy.profiles import PROFILE_THRESHOLDS
from escalation.signals.base import Signal
from escalation.signals.reversibility import ReversibilitySignal
from escalation.types import ProposedAction, SignalResult


def make_action(tool_name: str) -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments={}, agent_reasoning="test")


def run(coro):
    return asyncio.run(coro)


class StubSignal(Signal):
    """Fixed-score test double, to test aggregation math independent of any real signal."""

    def __init__(self, name: str, score: float, informative: bool = True):
        self.name = name
        self._score = score
        self._informative = informative

    async def score(self, action: ProposedAction) -> SignalResult:
        return SignalResult(
            name=self.name, score=self._score, reason="stub", cost_ms=0, informative=self._informative
        )


# --- profiles ---

def test_profile_thresholds_are_ordered_conservative_to_autonomous():
    assert PROFILE_THRESHOLDS["conservative"] < PROFILE_THRESHOLDS["balanced"] < PROFILE_THRESHOLDS["autonomous"]


def test_from_profile_rejects_unknown_profile():
    with pytest.raises(ValueError):
        Policy.from_profile("yolo")


# --- plain threshold behavior (no hard override involved) ---

def test_below_threshold_proceeds():
    policy = Policy.from_profile("balanced")  # threshold 0.25, read_file aggregate 0.1385
    decision = run(policy.evaluate(make_action("read_file")))
    assert decision.verdict == "proceed"
    assert decision.hard_override_triggered is False
    assert decision.question is None


def test_at_or_above_threshold_asks_human_without_hard_override():
    policy = Policy.from_profile("conservative")  # threshold 0.05, write_file aggregate 0.2731
    decision = run(policy.evaluate(make_action("write_file")))  # reversible-write, not hard override
    assert decision.verdict == "ask_human"
    assert decision.hard_override_triggered is False  # threshold-driven, not override-driven


def test_below_threshold_with_a_looser_profile_still_proceeds():
    # write_file's aggregate (0.2731) clears conservative (0.05) and balanced
    # (0.25) but not autonomous (0.6) -- autonomous is deliberately the
    # profile that tolerates this
    policy = Policy.from_profile("autonomous")
    decision = run(policy.evaluate(make_action("write_file")))
    assert decision.verdict == "proceed"
    assert decision.hard_override_triggered is False


# --- hard override: the actual point of this test file ---

def test_hard_override_forces_ask_human_even_when_aggregate_is_below_threshold():
    # threshold is deliberately set ABOVE delete_file's own score (0.75),
    # so a pure weighted-threshold policy would say "proceed" here.
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.9)
    decision = run(policy.evaluate(make_action("delete_file")))  # irreversible-write, score 0.75

    assert decision.aggregate_score == pytest.approx(0.75)
    assert decision.aggregate_score < policy.threshold  # the weighted score alone would proceed
    assert decision.verdict == "ask_human"  # but the override fires anyway
    assert decision.hard_override_triggered is True


def test_hard_override_fires_even_under_the_autonomous_profile():
    # autonomous threshold (0.6) is above delete_file's aggregate (0.4269) --
    # the most permissive preset would otherwise wave this through.
    policy = Policy.from_profile("autonomous")
    decision = run(policy.evaluate(make_action("delete_file")))
    assert decision.aggregate_score < policy.threshold
    assert decision.verdict == "ask_human"
    assert decision.hard_override_triggered is True


def test_hard_override_does_not_silently_inflate_aggregate_score():
    # overriding the verdict must not overwrite the honest weighted score
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.9)
    decision = run(policy.evaluate(make_action("delete_file")))
    assert decision.aggregate_score == pytest.approx(0.75)  # not bumped to 1.0


def test_external_effect_also_hard_overrides():
    policy = Policy.from_profile("balanced")
    decision = run(policy.evaluate(make_action("send_email")))
    assert decision.verdict == "ask_human"
    assert decision.hard_override_triggered is True


# --- weighted aggregation across multiple signals ---

def test_aggregate_score_is_the_weighted_average_of_all_signals():
    reversibility = ReversibilitySignal()  # read_file -> 0.0
    stub = StubSignal(name="stub", score=1.0)
    policy = Policy(
        reversibility=reversibility,
        weights={"reversibility": 0.5, "stub": 0.5},
        threshold=0.9,
        extra_signals=[stub],
    )
    decision = run(policy.evaluate(make_action("read_file")))
    assert decision.aggregate_score == pytest.approx(0.5)
    assert len(decision.signals) == 2


def test_missing_weight_for_an_active_signal_raises():
    reversibility = ReversibilitySignal()
    stub = StubSignal(name="stub", score=1.0)
    policy = Policy(
        reversibility=reversibility,
        weights={"reversibility": 1.0},  # "stub" has no configured weight
        threshold=0.9,
        extra_signals=[stub],
    )
    with pytest.raises(ValueError, match="stub"):
        run(policy.evaluate(make_action("read_file")))


def test_missing_weight_still_raises_even_when_signal_is_uninformative():
    # a signal missing from the weights dict is a real config gap
    # regardless of whether it happens to be uninformative on this call
    reversibility = ReversibilitySignal()
    stub = StubSignal(name="stub", score=1.0, informative=False)
    policy = Policy(
        reversibility=reversibility,
        weights={"reversibility": 1.0},
        threshold=0.9,
        extra_signals=[stub],
    )
    with pytest.raises(ValueError, match="stub"):
        run(policy.evaluate(make_action("read_file")))


def test_non_informative_signal_is_excluded_from_the_weighted_average():
    reversibility = ReversibilitySignal()  # read_file -> 0.0
    informative_stub = StubSignal(name="informative_stub", score=1.0, informative=True)
    uninformative_stub = StubSignal(name="uninformative_stub", score=1.0, informative=False)
    policy = Policy(
        reversibility=reversibility,
        weights={"reversibility": 0.5, "informative_stub": 0.25, "uninformative_stub": 0.25},
        threshold=0.9,
        extra_signals=[informative_stub, uninformative_stub],
    )
    decision = run(policy.evaluate(make_action("read_file")))
    # if uninformative_stub were included: (0.5*0 + 0.25*1 + 0.25*1) / 1.0 = 0.5
    # excluded instead: (0.5*0 + 0.25*1) / (0.5+0.25) = 0.25/0.75 = 1/3
    assert decision.aggregate_score == pytest.approx(1 / 3)
    assert len(decision.signals) == 3  # still logged, just excluded from the average


# --- human question generation (minimal, per earlier sign-off) ---

def test_ask_human_via_hard_override_recommends_abort():
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.9)
    decision = run(policy.evaluate(make_action("delete_file")))
    assert decision.question is not None
    assert decision.question.recommended_option == "abort"
    assert "irreversible-write" in decision.question.concern


def test_ask_human_via_threshold_only_recommends_proceed():
    policy = Policy.from_profile("conservative")  # threshold 0.05, write_file aggregate 0.2731
    decision = run(policy.evaluate(make_action("write_file")))  # not hard override
    assert decision.question is not None
    assert decision.question.recommended_option == "proceed"


def test_proceed_verdict_has_no_question():
    policy = Policy.from_profile("balanced")
    decision = run(policy.evaluate(make_action("read_file")))
    assert decision.question is None
