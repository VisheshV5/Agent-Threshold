"""Tests for the calibration curve sweep, written before the implementation.

Pins the API surface:
- sweep_thresholds(scenarios, profile, thresholds=None) -> list[CalibrationPoint]
- Each CalibrationPoint has .threshold and .metrics (a Metrics from eval.metrics)
- Reuses the same signal instances across all thresholds rather than
  reconstructing them per point

Structural predictions for the M1 10-scenario suite, verified rather than
assumed: every should_escalate=True scenario is a hard-override case (see
the M1 metrics writeup), so miss_rate must be 0.0 across the ENTIRE sweep
-- threshold literally cannot affect a hard-override verdict. Only the 4
threshold-sensitive scenarios (all should_escalate=False) can produce
false alarms, and their aggregate scores cluster at two values, so
false_alarm_rate should step down in two stages as threshold rises.
"""

import asyncio

from escalation.eval.calibration import sweep_thresholds
from escalation.eval.data.scenarios_m1 import SCENARIOS


def run(coro):
    return asyncio.run(coro)


def test_sweep_returns_points_sorted_by_threshold():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.8, 0.1, 0.5]))
    assert [p.threshold for p in points] == [0.1, 0.5, 0.8]


def test_miss_rate_is_zero_across_the_entire_sweep():
    # every should_escalate=True scenario in the M1 suite is hard-override
    # protected, so no threshold value can ever produce a miss
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0, 0.2, 0.34, 0.5, 0.8, 1.0]))
    assert all(p.metrics.miss_rate == 0.0 for p in points)


def test_false_alarm_rate_is_maximal_at_threshold_zero():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0]))
    # at threshold 0.0, every non-hard-override scenario's aggregate >= 0 -> escalates
    assert points[0].metrics.false_alarm_rate == 1.0


def test_false_alarm_rate_reaches_zero_once_threshold_clears_both_clusters():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.5, 1.0]))
    assert all(p.metrics.false_alarm_rate == 0.0 for p in points)


def test_false_alarm_rate_is_non_increasing_as_threshold_rises():
    points = run(sweep_thresholds(SCENARIOS, "balanced", thresholds=[0.0, 0.1, 0.25, 0.4, 0.6, 1.0]))
    rates = [p.metrics.false_alarm_rate for p in points]
    assert rates == sorted(rates, reverse=True)


def test_default_threshold_grid_spans_zero_to_one():
    points = run(sweep_thresholds(SCENARIOS, "balanced"))
    thresholds = [p.threshold for p in points]
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0
    assert len(thresholds) >= 10
