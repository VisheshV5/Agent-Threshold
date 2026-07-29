"""Calibration curve: sweep the policy threshold from 0 to 1 and report
false alarm rate / miss rate at each point.

Reuses the same signal instances across the whole sweep (only the
threshold changes per point) rather than reconstructing them per
threshold -- matters once a signal has real setup cost (an embedding
model, an API client), even though today's heuristic signals are cheap
either way.
"""

from dataclasses import dataclass

from escalation.eval.eval_policy import build_eval_policy
from escalation.eval.metrics import Metrics, compute_metrics
from escalation.eval.runner import run_scenarios
from escalation.eval.scenario import Scenario


@dataclass
class CalibrationPoint:
    threshold: float
    metrics: Metrics


def _default_thresholds() -> list[float]:
    return [round(i * 0.05, 2) for i in range(21)]  # 0.0, 0.05, ..., 1.0


async def sweep_thresholds(
    scenarios: list[Scenario],
    profile: str,
    thresholds: list[float] | None = None,
) -> list[CalibrationPoint]:
    thresholds = sorted(thresholds if thresholds is not None else _default_thresholds())

    # one policy (and its seeded novelty store) reused across every
    # threshold point -- only policy.threshold changes per point
    policy = build_eval_policy(profile)

    points = []
    for threshold in thresholds:
        policy.threshold = threshold
        results = await run_scenarios(policy, scenarios)
        points.append(CalibrationPoint(threshold=threshold, metrics=compute_metrics(results)))
    return points
