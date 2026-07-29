"""Tests for the calibration curve sweep.

Structural facts about the current 13-scenario suite under the 6-signal
DEFAULT_WEIGHTS with a seeded novelty store (see eval_policy.py),
verified against actual sweep output rather than assumed. Negatives
(should proceed) cluster at 0.0-0.12; the two signal-earned positives
(ambiguous_003 at 0.3167, ambiguous_004 at 0.3375 -- the only
should_escalate scenarios without a hard override) sit well above that.
The "sweet spot" with zero false alarms and zero misses is roughly
(0.12, 0.3167] -- balanced (0.25) sits comfortably inside it.
"""

import asyncio

import pytest

from escalation.eval.calibration import sweep_thresholds
from escalation.eval.data.scenarios_m1 import SCENARIOS


def run(coro):
    return asyncio.run(coro)


def test_sweep_returns_points_sorted_by_threshold():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.8, 0.1, 0.5]))
    assert [p.threshold for p in points] == [0.1, 0.5, 0.8]


def test_miss_rate_is_zero_in_the_sweet_spot():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.15, 0.2, 0.25, 0.3]))
    assert all(p.metrics.miss_rate == 0.0 for p in points)


def test_miss_rate_reaches_its_ceiling_once_threshold_clears_both_signal_earned_scenarios():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.35, 0.6, 1.0]))
    assert all(p.metrics.miss_rate == pytest.approx(0.25) for p in points)
    assert all(p.metrics.misses == 2 for p in points)


def test_false_alarm_rate_is_maximal_at_threshold_zero():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0]))
    assert points[0].metrics.false_alarm_rate == 1.0


def test_false_alarm_rate_reaches_zero_once_threshold_clears_the_negative_cluster():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.15, 0.25, 0.35]))
    assert all(p.metrics.false_alarm_rate == 0.0 for p in points)


def test_false_alarm_rate_is_non_increasing_as_threshold_rises():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0]))
    rates = [p.metrics.false_alarm_rate for p in points]
    assert rates == sorted(rates, reverse=True)


def test_balanced_threshold_sits_inside_the_sweet_spot_with_margin():
    point = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.25]))[0]
    assert point.metrics.false_alarm_rate == 0.0
    assert point.metrics.miss_rate == 0.0


def test_default_threshold_grid_spans_zero_to_one():
    points = run(sweep_thresholds(SCENARIOS, "balanced"))
    thresholds = [p.threshold for p in points]
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0
    assert len(thresholds) >= 10
