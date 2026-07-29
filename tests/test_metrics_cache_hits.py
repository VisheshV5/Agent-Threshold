"""Cache hits must be tracked separately in the metrics, per PROJECT.md.

The static scenario suite doesn't naturally produce cache hits (each
scenario is independent), so these results are constructed directly
rather than run through the eval harness.
"""

from escalation.eval.metrics import compute_metrics
from escalation.eval.runner import ScenarioResult
from escalation.eval.scenario import Scenario
from escalation.types import Decision, ProposedAction, SignalResult


def make_result(should_escalate: bool, verdict: str, cache_hit: bool) -> ScenarioResult:
    scenario = Scenario(
        id="s",
        description="test",
        category="safe",
        action=ProposedAction(tool_name="read_file", arguments={}, agent_reasoning="test"),
        should_escalate=should_escalate,
    )
    decision = Decision(
        verdict=verdict,
        aggregate_score=0.5,
        signals=[SignalResult(name="reversibility", score=0.0, reason="test", cost_ms=0)],
        cache_hit=cache_hit,
    )
    return ScenarioResult(scenario=scenario, decision=decision)


def test_cache_hits_are_counted():
    results = [
        make_result(should_escalate=False, verdict="proceed", cache_hit=True),
        make_result(should_escalate=True, verdict="ask_human", cache_hit=True),
        make_result(should_escalate=False, verdict="proceed", cache_hit=False),
    ]
    metrics = compute_metrics(results)
    assert metrics.cache_hits == 2


def test_cache_hit_rate_is_a_fraction_of_total():
    results = [
        make_result(should_escalate=False, verdict="proceed", cache_hit=True),
        make_result(should_escalate=False, verdict="proceed", cache_hit=False),
        make_result(should_escalate=False, verdict="proceed", cache_hit=False),
        make_result(should_escalate=False, verdict="proceed", cache_hit=False),
    ]
    metrics = compute_metrics(results)
    assert metrics.cache_hit_rate == 0.25


def test_cache_hit_rate_is_zero_with_no_results():
    metrics = compute_metrics([])
    assert metrics.cache_hit_rate == 0.0
    assert metrics.cache_hits == 0
