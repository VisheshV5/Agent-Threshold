"""Tests for the calibration curve sweep.

Structural facts about the full 40-scenario suite under the 6-signal
DEFAULT_WEIGHTS with a seeded novelty store, verified against actual
sweep output rather than assumed. Among threshold-sensitive scenarios
(not hard-override protected), negatives top out at 0.2367
(ambiguous_010) and the lowest signal-earned positive sits at 0.2992
(ambiguous_012) -- the sweet spot with zero false alarms and zero
misses is (0.2367, 0.2992]. Balanced (0.25) sits inside it with margin
on both sides, but it's a narrower window than the 13-scenario suite
had, and misses climb steadily above it as more positives fall below
threshold, reaching a ceiling of 44% at 1.0 (the 11 threshold-sensitive
positives that never get hard-override protection).
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
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.24, 0.25, 0.29]))
    assert all(p.metrics.miss_rate == 0.0 for p in points)


def test_miss_rate_climbs_and_then_plateaus_above_the_sweet_spot():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.3, 0.4, 0.5, 0.75, 1.0]))
    rates = [p.metrics.miss_rate for p in points]
    assert rates == sorted(rates)  # non-decreasing as threshold rises
    assert rates[0] > 0.0  # first miss appears just past the sweet spot
    assert points[-2].metrics.miss_rate == points[-1].metrics.miss_rate  # plateaued by threshold 1.0


def test_false_alarm_rate_is_maximal_at_threshold_zero():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0]))
    assert points[0].metrics.false_alarm_rate == 1.0


def test_false_alarm_rate_reaches_zero_in_the_sweet_spot():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.24, 0.25, 0.29]))
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
