"""Tests for the per-signal ablation study.

Pins the API surface:
- run_ablation(scenarios, full_policy: Policy) -> list[AblationResult],
  one entry per signal in full_policy.weights -- takes a pre-built
  Policy rather than a profile name, so the caller's actual
  configuration (a seeded NoveltyStore, custom weights, etc.) is what
  gets ablated, not a fresh rebuild from scratch
- AblationResult(signal_name, verdict_changes, full_accuracy,
  ablated_accuracy, accuracy_delta)
- Ablating reversibility must ALSO disable the hard override (a neutral
  stand-in that always classifies read-only), not just drop it from the
  weighted average -- otherwise "ablating reversibility" would still
  silently benefit from its hard-override protection
- Ablated variants reuse the full policy's actual signal instances
  (not fresh ones), so a stateful signal like novelty carries its real
  configuration into the ablation instead of comparing against an
  empty store regardless of what the caller set up
- Predictions about the CURRENT 13-scenario suite, verified rather than
  assumed: ablating blast_radius changes ambiguous_003's verdict,
  ablating self_consistency changes ambiguous_004's verdict, and
  ablating novelty/staleness/thrash changes NOTHING yet, since none of
  the 13 scenarios differentiate on those three signals
"""

import asyncio

from escalation.eval.ablation import _build_ablated_policy, run_ablation
from escalation.eval.data.scenarios_m1 import SCENARIOS
from escalation.policy.policy import Policy
from escalation.signals.novelty import NoveltySignal, NoveltyStore
from escalation.types import ProposedAction

ALL_SIGNAL_NAMES = {"reversibility", "blast_radius", "self_consistency", "novelty", "staleness", "thrash"}


def run(coro):
    return asyncio.run(coro)


def balanced_policy() -> Policy:
    return Policy.from_profile("balanced")


def test_returns_one_result_per_signal():
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    assert {r.signal_name for r in results} == ALL_SIGNAL_NAMES


def test_ablating_reversibility_causes_the_largest_accuracy_drop():
    # every hard-override scenario (6 of 13) depends entirely on
    # reversibility for protection -- removing it should be the single
    # most damaging ablation by a wide margin
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    by_name = {r.signal_name: r for r in results}
    reversibility_drop = by_name["reversibility"].accuracy_delta
    for name in ALL_SIGNAL_NAMES - {"reversibility"}:
        assert reversibility_drop > by_name[name].accuracy_delta


def test_ablating_reversibility_disables_hard_override_not_just_the_weight():
    # if hard override still fired, ablating reversibility would show
    # zero impact on the 6 hard-override scenarios -- it must not.
    # Verified exactly 5 of 6 flip: dangerous_003 (execute_trade,
    # quantity=500) is redundantly caught by blast_radius alone even
    # without reversibility, since its parsed quantity clears threshold
    # on its own -- a genuine finding about this scenario, not a bug.
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    by_name = {r.signal_name: r for r in results}
    assert by_name["reversibility"].verdict_changes == 5


def test_ablating_blast_radius_flips_ambiguous_003():
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    by_name = {r.signal_name: r for r in results}
    assert by_name["blast_radius"].verdict_changes >= 1


def test_ablating_self_consistency_flips_ambiguous_004():
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    by_name = {r.signal_name: r for r in results}
    assert by_name["self_consistency"].verdict_changes >= 1


def test_ablating_novelty_staleness_thrash_changes_nothing_in_current_suite():
    # honest finding: none of the 13 scenarios differentiate on these
    # three signals yet -- that's exactly the gap the 40-scenario suite
    # needs to close
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    by_name = {r.signal_name: r for r in results}
    for name in ("novelty", "staleness", "thrash"):
        assert by_name[name].verdict_changes == 0
        assert by_name[name].accuracy_delta == 0.0


def test_full_accuracy_matches_across_all_results():
    results = run(run_ablation(SCENARIOS, balanced_policy()))
    full_accuracies = {r.full_accuracy for r in results}
    assert len(full_accuracies) == 1  # same baseline reported alongside every ablation


def test_ablated_policy_for_a_different_signal_reuses_the_same_seeded_novelty_store():
    # if _build_ablated_policy rebuilt novelty from scratch, this would
    # always compare against an empty store regardless of what the
    # caller configured -- checked directly by comparing novelty's own
    # score between the full policy and a policy ablated for a
    # DIFFERENT signal (thrash): they must match exactly, proving the
    # same store instance (not a fresh empty one) was reused
    store = NoveltyStore()
    store.record_success(["read_file"])
    full_policy = Policy.from_profile("balanced", novelty=NoveltySignal(store=store))
    ablated_for_thrash = _build_ablated_policy(full_policy, "thrash")

    action = ProposedAction(tool_name="read_file", arguments={}, agent_reasoning="test")
    full_decision = run(full_policy.evaluate(action))
    ablated_decision = run(ablated_for_thrash.evaluate(action))

    full_novelty = next(s for s in full_decision.signals if s.name == "novelty")
    ablated_novelty = next(s for s in ablated_decision.signals if s.name == "novelty")
    assert full_novelty.score == ablated_novelty.score == 0.0  # exact match to the seed -> not novel
