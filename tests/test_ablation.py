"""Tests for the per-signal ablation study, run against the full
40-scenario suite.

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

Verified findings on the full 40-scenario suite (via build_eval_policy,
which seeds novelty): every signal now shows real impact --
reversibility still dominates (11 verdict changes; 14 hard-override
scenarios exist, but a few are redundantly caught by other signals too),
staleness 4, self_consistency 3, blast_radius/thrash/novelty 2 each.
None are zero anymore, unlike the original 13-scenario suite where
novelty/staleness/thrash all carried zero weight.
"""

import asyncio

from escalation.eval.ablation import _build_ablated_policy, run_ablation
from escalation.eval.data.scenarios_m1 import SCENARIOS
from escalation.eval.eval_policy import build_eval_policy
from escalation.policy.policy import Policy
from escalation.signals.novelty import NoveltySignal, NoveltyStore
from escalation.types import ProposedAction

ALL_SIGNAL_NAMES = {"reversibility", "blast_radius", "self_consistency", "novelty", "staleness", "thrash"}


def run(coro):
    return asyncio.run(coro)


def test_returns_one_result_per_signal():
    results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
    assert {r.signal_name for r in results} == ALL_SIGNAL_NAMES


def test_ablating_reversibility_causes_the_largest_accuracy_drop():
    # every hard-override scenario (14 of 40) depends on reversibility
    # for protection -- removing it should be the single most damaging
    # ablation by a wide margin
    results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
    by_name = {r.signal_name: r for r in results}
    reversibility_drop = by_name["reversibility"].accuracy_delta
    for name in ALL_SIGNAL_NAMES - {"reversibility"}:
        assert reversibility_drop > by_name[name].accuracy_delta


def test_ablating_reversibility_disables_hard_override_not_just_the_weight():
    # if hard override still fired, ablating reversibility would show
    # zero impact on the 14 hard-override scenarios -- it must not
    results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
    by_name = {r.signal_name: r for r in results}
    assert by_name["reversibility"].verdict_changes >= 10


def test_every_signal_now_carries_real_weight():
    # the point of the 40-scenario expansion: on the original 13,
    # novelty/staleness/thrash all showed exactly zero impact.
    # Every signal must show at least one verdict change now.
    results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
    for r in results:
        assert r.verdict_changes >= 1, f"{r.signal_name} still carries zero weight"
        assert r.accuracy_delta > 0.0


def test_full_accuracy_matches_across_all_results():
    results = run(run_ablation(SCENARIOS, build_eval_policy("balanced")))
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
