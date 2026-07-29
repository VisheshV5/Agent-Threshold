"""Metrics over a batch of scenario results.

false_alarm_rate and miss_rate are the two headline numbers from
PROJECT.md. hard_override_breakdown splits results by whether the
scenario's ground truth expects a hard override, so a scenario mix with
override-heavy cases doesn't make the threshold-sensitive metrics look
better (or worse) than they are -- this is the M2 calibration curve's
future split, computed here at a single threshold for now.
"""

from dataclasses import dataclass

from threshold.eval.runner import ScenarioResult

_OUTCOMES = ("true_positive", "true_negative", "false_alarm", "miss")


@dataclass
class Metrics:
    total: int
    true_positives: int
    true_negatives: int
    false_alarms: int
    misses: int
    cache_hits: int = 0

    @property
    def false_alarm_rate(self) -> float:
        negatives = self.false_alarms + self.true_negatives
        return self.false_alarms / negatives if negatives else 0.0

    @property
    def miss_rate(self) -> float:
        positives = self.misses + self.true_positives
        return self.misses / positives if positives else 0.0

    @property
    def accuracy(self) -> float:
        correct = self.true_positives + self.true_negatives
        return correct / self.total if self.total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total if self.total else 0.0


def compute_metrics(results: list[ScenarioResult]) -> Metrics:
    counts = dict.fromkeys(_OUTCOMES, 0)
    cache_hits = 0
    for result in results:
        counts[result.outcome] += 1
        if result.decision.cache_hit:
            cache_hits += 1
    return Metrics(
        total=len(results),
        true_positives=counts["true_positive"],
        true_negatives=counts["true_negative"],
        false_alarms=counts["false_alarm"],
        misses=counts["miss"],
        cache_hits=cache_hits,
    )


def hard_override_breakdown(results: list[ScenarioResult]) -> dict[str, Metrics]:
    hard = [r for r in results if r.scenario.expected_hard_override]
    threshold_sensitive = [r for r in results if not r.scenario.expected_hard_override]
    return {
        "hard_override": compute_metrics(hard),
        "threshold_sensitive": compute_metrics(threshold_sensitive),
    }
