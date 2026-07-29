"""Per-signal ablation: how much does each signal actually change decisions?

For each signal, builds a policy with that one signal removed (weight
and all) and compares its verdicts against the full policy's. A signal
whose removal changes nothing is currently just along for the ride --
which the current scenario suite may reveal about novelty, staleness,
and thrash specifically, since coverage of them is uneven.

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
from escalation.policy.profiles import DEFAULT_WEIGHTS, PROFILE_THRESHOLDS
from escalation.signals.base import Signal
from escalation.signals.blast_radius import BlastRadiusSignal
from escalation.signals.novelty import NoveltySignal
from escalation.signals.reversibility import ActionCategory, ReversibilitySignal
from escalation.signals.self_consistency import SelfConsistencySignal
from escalation.signals.staleness import StalenessSignal
from escalation.signals.thrash import ThrashSignal


class _NeutralFallbackClassifier:
    """Always read-only -- the ablation stand-in for "no reversibility signal"."""

    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory:
        return ActionCategory.READ_ONLY


def _neutral_reversibility() -> ReversibilitySignal:
    return ReversibilitySignal(registry={}, fallback=_NeutralFallbackClassifier())


def _build_extra_signals() -> dict[str, Signal]:
    return {
        "blast_radius": BlastRadiusSignal(),
        "self_consistency": SelfConsistencySignal(),
        "novelty": NoveltySignal(),
        "staleness": StalenessSignal(),
        "thrash": ThrashSignal(),
    }


def _build_policy_excluding(signal_name: str, profile: str) -> Policy:
    # Policy always includes reversibility as a mandatory signal
    # regardless of extra_signals, so removing a weight key outright
    # would raise "no weight configured" once that signal is still
    # scored. Zeroing the weight is mathematically identical to full
    # exclusion (contributes nothing to either the numerator or
    # denominator) and works uniformly for all six signals.
    weights = dict(DEFAULT_WEIGHTS)
    weights[signal_name] = 0.0

    reversibility = _neutral_reversibility() if signal_name == "reversibility" else ReversibilitySignal()
    extra = _build_extra_signals()

    return Policy(
        reversibility=reversibility,
        weights=weights,
        threshold=PROFILE_THRESHOLDS[profile],
        extra_signals=list(extra.values()),
    )


@dataclass
class AblationResult:
    signal_name: str
    verdict_changes: int
    full_accuracy: float
    ablated_accuracy: float
    accuracy_delta: float  # full - ablated; positive means removing it hurt accuracy


async def run_ablation(scenarios: list[Scenario], profile: str = "balanced") -> list[AblationResult]:
    full_policy = Policy.from_profile(profile)
    full_results = await run_scenarios(full_policy, scenarios)
    full_metrics = compute_metrics(full_results)
    full_verdicts = {r.scenario.id: r.decision.verdict for r in full_results}

    results = []
    for signal_name in DEFAULT_WEIGHTS:
        ablated_policy = _build_policy_excluding(signal_name, profile)
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
