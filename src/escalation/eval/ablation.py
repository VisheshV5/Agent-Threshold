"""Per-signal ablation: how much does each signal actually change decisions?

For each signal, builds a policy with that one signal removed (weight
and all) and compares its verdicts against the caller's full policy.
Reuses the SAME signal instances from that policy (not fresh ones) --
critical for novelty specifically, since it carries state (a seeded
NoveltyStore); rebuilding a bare NoveltySignal() from scratch would
compare against an empty store and always show zero impact regardless
of how the caller actually configured things.

Ablating reversibility is a special case: it drives the hard override
as well as the weighted average, so simply dropping its weight would
leave hard override still fully protecting every irreversible-write/
external-effect scenario, understating its true importance. Ablating it
means swapping in a neutral stand-in that always classifies read-only,
so both mechanisms lose its input.
"""

from dataclasses import dataclass

from escalation.eval.metrics import compute_metrics
from escalation.eval.runner import run_scenarios
from escalation.eval.scenario import Scenario
from escalation.policy.policy import Policy
from escalation.signals.reversibility import ActionCategory, ReversibilitySignal


class _NeutralFallbackClassifier:
    """Always read-only -- the ablation stand-in for "no reversibility signal"."""

    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory:
        return ActionCategory.READ_ONLY


def _neutral_reversibility() -> ReversibilitySignal:
    return ReversibilitySignal(registry={}, fallback=_NeutralFallbackClassifier())


def _build_ablated_policy(full_policy: Policy, signal_name: str) -> Policy:
    # Zeroing the weight (rather than removing the dict key) is
    # mathematically identical to full exclusion and works uniformly for
    # every signal, including reversibility -- which Policy always scores
    # regardless of extra_signals, so dropping its key outright would
    # raise "no weight configured" once it's still evaluated.
    weights = dict(full_policy.weights)
    weights[signal_name] = 0.0

    if signal_name == "reversibility":
        reversibility = _neutral_reversibility()
    else:
        reversibility = full_policy.reversibility

    extra_signals = full_policy.signals[1:]  # same instances -- preserves e.g. novelty's seeded store

    return Policy(
        reversibility=reversibility,
        weights=weights,
        threshold=full_policy.threshold,
        extra_signals=extra_signals,
    )


@dataclass
class AblationResult:
    signal_name: str
    verdict_changes: int
    full_accuracy: float
    ablated_accuracy: float
    accuracy_delta: float  # full - ablated; positive means removing it hurt accuracy


async def run_ablation(scenarios: list[Scenario], full_policy: Policy) -> list[AblationResult]:
    full_results = await run_scenarios(full_policy, scenarios)
    full_metrics = compute_metrics(full_results)
    full_verdicts = {r.scenario.id: r.decision.verdict for r in full_results}

    results = []
    for signal_name in full_policy.weights:
        ablated_policy = _build_ablated_policy(full_policy, signal_name)
        ablated_results = await run_scenarios(ablated_policy, scenarios)
        ablated_metrics = compute_metrics(ablated_results)
        verdict_changes = sum(
            1 for r in ablated_results if r.decision.verdict != full_verdicts[r.scenario.id]
        )
        results.append(
            AblationResult(
                signal_name=signal_name,
                verdict_changes=verdict_changes,
                full_accuracy=full_metrics.accuracy,
                ablated_accuracy=ablated_metrics.accuracy,
                accuracy_delta=full_metrics.accuracy - ablated_metrics.accuracy,
            )
        )
    return results
