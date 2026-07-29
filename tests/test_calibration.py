"""Tests for the calibration curve sweep.

Structural facts about the current 13-scenario suite, verified against
actual sweep output rather than assumed: the balanced threshold (0.5)
sits right at the edge of a real tradeoff now. Below ~0.35, false alarms
come from the threshold-sensitive negatives (safe_001/003 at 0.20,
safe_004 at 0.32, safe_002/ambiguous_002 at 0.34). Above 0.54, the two
scenarios that only blast_radius/self_consistency catch (ambiguous_003,
ambiguous_004, both aggregate 0.54) start being missed -- these are the
first misses this suite can produce, since every OTHER should_escalate
scenario is hard-override protected and threshold-invariant.
"""

import asyncio

from escalation.eval.calibration import sweep_thresholds
from escalation.eval.data.scenarios_m1 import SCENARIOS


def run(coro):
    return asyncio.run(coro)


def test_sweep_returns_points_sorted_by_threshold():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.8, 0.1, 0.5]))
    assert [p.threshold for p in points] == [0.1, 0.5, 0.8]


def test_miss_rate_is_zero_up_to_and_including_the_balanced_threshold():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0, 0.2, 0.34, 0.5]))
    assert all(p.metrics.miss_rate == 0.0 for p in points)


def test_miss_rate_becomes_nonzero_once_threshold_clears_the_signal_earned_scenarios():
    # ambiguous_003/004 (aggregate 0.54) are the only should_escalate=True
    # scenarios without a hard override, so they're the only source of a
    # miss anywhere in this suite
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.55, 0.8, 1.0]))
    assert all(p.metrics.miss_rate == 0.25 for p in points)
    assert all(p.metrics.misses == 2 for p in points)


def test_false_alarm_rate_is_maximal_at_threshold_zero():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0]))
    assert points[0].metrics.false_alarm_rate == 1.0


def test_false_alarm_rate_reaches_zero_once_threshold_clears_all_negative_clusters():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.35, 0.5, 1.0]))
    assert all(p.metrics.false_alarm_rate == 0.0 for p in points)


def test_false_alarm_rate_is_non_increasing_as_threshold_rises():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0, 0.1, 0.25, 0.4, 0.6, 1.0]))
    rates = [p.metrics.false_alarm_rate for p in points]
    assert rates == sorted(rates, reverse=True)


def test_balanced_threshold_sits_at_the_edge_of_the_tradeoff():
    # at 0.50: no false alarms, no misses yet. one step past it (0.55):
    # false alarms are still clear, but misses appear. Demonstrates the
    # curve, not just the single balanced-profile point.
    at_threshold = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.5]))[0]
    just_past = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.55]))[0]
    assert at_threshold.metrics.false_alarm_rate == 0.0
    assert at_threshold.metrics.miss_rate == 0.0
    assert just_past.metrics.miss_rate > at_threshold.metrics.miss_rate


def test_default_threshold_grid_spans_zero_to_one():
    points = run(sweep_thresholds(SCENARIOS, "balanced"))
    thresholds = [p.threshold for p in points]
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0
    assert len(thresholds) >= 10
