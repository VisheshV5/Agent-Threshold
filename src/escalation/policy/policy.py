"""Policy: combines signals into a Decision given a threshold.

Weighted combination of all signals' scores, with a hard override: if the
reversibility signal classifies the action as irreversible-write or
external-effect, the verdict is always "ask_human" regardless of where
the weighted score falls relative to the threshold. The override changes
only the verdict, never the reported aggregate_score -- that number stays
an honest weighted average so it's still meaningful for the calibration
curve and for telling threshold-driven escalations apart from
override-driven ones (Decision.hard_override_triggered).
"""

from escalation.policy.profiles import PROFILE_THRESHOLDS
from escalation.signals.base import Signal
from escalation.signals.reversibility import HARD_OVERRIDE_CATEGORIES, ReversibilitySignal
from escalation.types import Decision, HumanQuestion, ProposedAction, SignalResult


class Policy:
    def __init__(
        self,
        reversibility: ReversibilitySignal,
        weights: dict[str, float],
        threshold: float,
        extra_signals: list[Signal] | None = None,
    ):
        self.reversibility = reversibility
        self.signals: list[Signal] = [reversibility, *(extra_signals or [])]
        self.weights = weights
        self.threshold = threshold

    @classmethod
    def from_profile(cls, profile: str, reversibility: ReversibilitySignal | None = None) -> "Policy":
        if profile not in PROFILE_THRESHOLDS:
            raise ValueError(f"unknown profile: {profile!r}, expected one of {sorted(PROFILE_THRESHOLDS)}")
        return cls(
            reversibility=reversibility or ReversibilitySignal(),
            weights={"reversibility": 1.0},
            threshold=PROFILE_THRESHOLDS[profile],
        )

    async def evaluate(self, action: ProposedAction) -> Decision:
        category, _source = await self.reversibility.classify(action)
        hard_override = category in HARD_OVERRIDE_CATEGORIES

        results = [await signal.score(action) for signal in self.signals]
        aggregate = self._aggregate(results)

        verdict = "ask_human" if (hard_override or aggregate >= self.threshold) else "proceed"

        question = (
            self._build_question(action, results, hard_override) if verdict == "ask_human" else None
        )

        return Decision(
            verdict=verdict,
            aggregate_score=aggregate,
            signals=results,
            question=question,
            hard_override_triggered=hard_override,
        )

    def _aggregate(self, results: list[SignalResult]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for result in results:
            if result.name not in self.weights:
                raise ValueError(f"no weight configured for signal '{result.name}'")
            weight = self.weights[result.name]
            total_weight += weight
            weighted_sum += weight * result.score
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def _build_question(
        self, action: ProposedAction, results: list[SignalResult], hard_override: bool
    ) -> HumanQuestion:
        top = max(results, key=lambda r: r.score)
        return HumanQuestion(
            summary=f"About to call '{action.tool_name}'.",
            concern=top.reason,
            options=["proceed", "abort"],
            recommended_option="abort" if hard_override else "proceed",
        )
