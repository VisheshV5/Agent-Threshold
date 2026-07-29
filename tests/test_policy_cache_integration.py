"""Tests for Policy's integration with DecisionCache, written before the
implementation.

Pins the API surface:
- Policy(..., cache=None) -- optional, no behavior change when absent
- Policy.evaluate(action): if a cached resolution exists for a
  non-hard-override action, the verdict comes from the cache (no
  question, no threshold comparison), and Decision.cache_hit is True
- Policy.record_resolution(action, resolution) -- writes to the cache,
  but REFUSES (raises) for a hard-override action, no exceptions,
  matching the earlier decision that hard-override categories are
  never cached
- A cache lookup never even happens for hard-override actions, not
  just that recording is blocked
"""

import asyncio

import pytest

from escalation.decision_cache import DecisionCache
from escalation.policy.policy import Policy
from escalation.signals.reversibility import ReversibilitySignal
from escalation.types import ProposedAction


def make_action(tool_name: str, arguments: dict | None = None) -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments=arguments or {}, agent_reasoning="test")


def run(coro):
    return asyncio.run(coro)


def test_no_cache_configured_behaves_exactly_as_before():
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.1)
    decision = run(policy.evaluate(make_action("write_file")))
    assert decision.cache_hit is False


def test_cache_miss_proceeds_through_normal_threshold_logic():
    cache = DecisionCache()
    policy = Policy(
        reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.1, cache=cache
    )
    decision = run(policy.evaluate(make_action("write_file")))
    assert decision.cache_hit is False
    assert decision.verdict == "ask_human"  # 0.35 >= 0.1, normal threshold logic


def test_recorded_resolution_is_used_on_a_later_identical_call():
    cache = DecisionCache()
    policy = Policy(
        reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.1, cache=cache
    )
    action = make_action("write_file", {"path": "/tmp/x"})

    run(policy.record_resolution(action, "proceed"))
    decision = run(policy.evaluate(action))

    assert decision.cache_hit is True
    assert decision.verdict == "proceed"  # from cache, even though 0.35 >= 0.1 would normally ask
    assert decision.question is None


def test_record_resolution_refuses_a_hard_override_action():
    cache = DecisionCache()
    policy = Policy(
        reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.9, cache=cache
    )
    with pytest.raises(ValueError, match="hard.override"):
        run(policy.record_resolution(make_action("delete_file"), "proceed"))


def test_hard_override_actions_never_consult_the_cache():
    # even if something were cached under the same key (shouldn't be
    # possible via record_resolution, but verified defensively at the
    # read side too), a hard-override action must still always ask
    cache = DecisionCache()
    cache.record(make_action("delete_file", {"path": "/x"}), "proceed")
    policy = Policy(
        reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.9, cache=cache
    )
    decision = run(policy.evaluate(make_action("delete_file", {"path": "/x"})))
    assert decision.cache_hit is False
    assert decision.verdict == "ask_human"
    assert decision.hard_override_triggered is True
