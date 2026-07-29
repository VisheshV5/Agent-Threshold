"""Tests for the per-signal ablation study, written before the implementation.

Pins the API surface:
- run_ablation(scenarios, profile="balanced") -> list[AblationResult],
  one entry per signal in DEFAULT_WEIGHTS
- AblationResult(signal_name, verdict_changes, full_accuracy,
  ablated_accuracy, accuracy_delta)
- Ablating reversibility must ALSO disable the hard override (a neutral
  stand-in that always classifies read-only), not just drop it from the
  weighted average -- otherwise "ablating reversibility" would still
  silently benefit from its hard-override protection
- Predictions about the CURRENT 13-scenario suite, verified rather than
  assumed: ablating blast_radius changes ambiguous_003's verdict,
  ablating self_consistency changes ambiguous_004's verdict, and
  ablating novelty/staleness/thrash changes NOTHING yet, since none of
  the 13 scenarios differentiate on those three signals
"""

import asyncio

from escalation.eval.ablation import run_ablation
from escalation.eval.data.scenarios_m1 import SCENARIOS

ALL_SIGNAL_NAMES = {"reversibility", "blast_radius", "self_consistency", "novelty", "staleness", "thrash"}


def run(coro):
    return asyncio.run(coro)


def test_returns_one_result_per_signal():
    results = run(run_ablation(SCENARIOS, "balanced"))
    assert {r.signal_name for r in results} == ALL_SIGNAL_NAMES


def test_ablating_reversibility_causes_the_largest_accuracy_drop():
    # every hard-override scenario (6 of 13) depends entirely on
    # reversibility for protection -- removing it should be the single
    # most damaging ablation by a wide margin
    results = run(run_ablation(SCENARIOS, "balanced"))
    by_name = {r.signal_name: r for r in results}
    reversibility_drop = by_name["reversibility"].accuracy_delta
    for name in ALL_SIGNAL_NAMES - {"reversibility"}:
        assert reversibility_drop > by_name[name].accuracy_delta


def test_ablating_reversibility_disables_hard_override_not_just_the_weight():
    # if hard override still fired, ablating reversibility would show
    # zero impact on the 6 hard-override scenarios -- it must not
    results = run(run_ablation(SCENARIOS, "balanced"))
    by_name = {r.signal_name: r for r in results}
    assert by_name["reversibility"].verdict_changes >= 6


def test_ablating_blast_radius_flips_ambiguous_003():
    results = run(run_ablation(SCENARIOS, "balanced"))
    by_name = {r.signal_name: r for r in results}
    assert by_name["blast_radius"].verdict_changes >= 1


def test_ablating_self_consistency_flips_ambiguous_004():
    results = run(run_ablation(SCENARIOS, "balanced"))
    by_name = {r.signal_name: r for r in results}
    assert by_name["self_consistency"].verdict_changes >= 1


def test_ablating_novelty_staleness_thrash_changes_nothing_in_current_suite():
    # honest finding: none of the 13 scenarios differentiate on these
    # three signals yet -- that's exactly the gap the 40-scenario suite
    # needs to close
    results = run(run_ablation(SCENARIOS, "balanced"))
    by_name = {r.signal_name: r for r in results}
    for name in ("novelty", "staleness", "thrash"):
        assert by_name[name].verdict_changes == 0
        assert by_name[name].accuracy_delta == 0.0


def test_full_accuracy_matches_across_all_results():
    results = run(run_ablation(SCENARIOS, "balanced"))
    full_accuracies = {r.full_accuracy for r in results}
    assert len(full_accuracies) == 1  # same baseline reported alongside every ablation
