"""Policy.evaluate() must call reversibility.classify() exactly once.

Found via a sanity check while wiring the demo: evaluate() calls
classify() directly for the hard-override check, and ReversibilitySignal
.score() calls classify() again internally -- harmless when it's a free
heuristic, but AnthropicFallbackClassifier makes classify() a real,
billed API call, so calling it twice silently doubles the cost of every
non-registry classification.
"""

import asyncio

from escalation.policy.policy import Policy
from escalation.signals.reversibility import ReversibilitySignal
from escalation.types import ProposedAction


def run(coro):
    return asyncio.run(coro)


class CountingReversibilitySignal(ReversibilitySignal):
    def __init__(self):
        super().__init__()
        self.classify_calls = 0

    async def classify(self, action):
        self.classify_calls += 1
        return await super().classify(action)


def test_classify_is_called_exactly_once_per_evaluate_for_a_registry_hit():
    reversibility = CountingReversibilitySignal()
    policy = Policy(reversibility=reversibility, weights={"reversibility": 1.0}, threshold=0.5)
    run(policy.evaluate(ProposedAction(tool_name="read_file", arguments={}, agent_reasoning="test")))
    assert reversibility.classify_calls == 1


def test_classify_is_called_exactly_once_per_evaluate_for_a_fallback_hit():
    reversibility = CountingReversibilitySignal()
    policy = Policy(reversibility=reversibility, weights={"reversibility": 1.0}, threshold=0.5)
    run(policy.evaluate(ProposedAction(tool_name="some_unregistered_tool", arguments={}, agent_reasoning="test")))
    assert reversibility.classify_calls == 1


def test_hard_override_and_score_are_still_consistent_with_a_single_classify_call():
    reversibility = CountingReversibilitySignal()
    policy = Policy(reversibility=reversibility, weights={"reversibility": 1.0}, threshold=0.9)
    decision = run(policy.evaluate(ProposedAction(tool_name="delete_file", arguments={}, agent_reasoning="test")))
    assert reversibility.classify_calls == 1
    assert decision.hard_override_triggered is True
    reversibility_result = next(s for s in decision.signals if s.name == "reversibility")
    assert reversibility_result.score == 0.75  # IRREVERSIBLE_WRITE, from the single classify() call
    assert "irreversible-write" in reversibility_result.reason
