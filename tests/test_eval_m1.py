import asyncio

from escalation.eval.data.scenarios_m1 import SCENARIOS
from escalation.eval.metrics import compute_metrics, hard_override_breakdown
from escalation.eval.runner import run_scenarios
from escalation.policy.policy import Policy


def run(coro):
    return asyncio.run(coro)


def test_scenarios_span_all_four_categories():
    assert len(SCENARIOS) == 13  # 10 from M1 + 3 added in M2
    assert {s.category for s in SCENARIOS} == {"safe", "dangerous", "ambiguous", "adversarial"}


def test_scenario_ids_are_unique():
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_all_scenarios_pass_under_balanced_policy():
    policy = Policy.from_profile("balanced")
    results = run(run_scenarios(policy, SCENARIOS))
    failures = [r.scenario.id for r in results if not r.correct]
    assert failures == [], f"scenarios failing: {failures}"


def test_metrics_are_perfect_under_the_full_three_signal_policy():
    policy = Policy.from_profile("balanced")
    results = run(run_scenarios(policy, SCENARIOS))
    metrics = compute_metrics(results)
    assert metrics.total == 13
    assert metrics.false_alarm_rate == 0.0
    assert metrics.miss_rate == 0.0
    assert metrics.accuracy == 1.0


def test_hard_override_breakdown_splits_six_and_seven():
    policy = Policy.from_profile("balanced")
    results = run(run_scenarios(policy, SCENARIOS))
    breakdown = hard_override_breakdown(results)
    assert breakdown["hard_override"].total == 6
    assert breakdown["threshold_sensitive"].total == 7
    assert breakdown["hard_override"].accuracy == 1.0
    assert breakdown["threshold_sensitive"].accuracy == 1.0


def test_blast_radius_and_self_consistency_scenarios_escalate_without_hard_override():
    policy = Policy.from_profile("balanced")
    results = run(run_scenarios(policy, SCENARIOS))
    by_id = {r.scenario.id: r for r in results}

    for scenario_id in ("ambiguous_003", "ambiguous_004"):
        r = by_id[scenario_id]
        assert r.decision.hard_override_triggered is False
        assert r.decision.verdict == "ask_human"
        assert r.decision.aggregate_score >= policy.threshold
