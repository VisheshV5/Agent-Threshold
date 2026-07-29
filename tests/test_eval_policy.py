import asyncio

import pytest

from threshold.eval.eval_policy import SEED_TRAJECTORIES, build_eval_policy, build_seeded_novelty_store
from threshold.types import ProposedAction


def run(coro):
    return asyncio.run(coro)


def test_seeded_store_recognizes_a_seed_trajectory_as_not_novel():
    store = build_seeded_novelty_store()
    first_seed = tuple(SEED_TRAJECTORIES[0])
    similarity = run(store.max_similarity(first_seed))
    assert similarity == pytest.approx(1.0)


def test_seeded_store_is_not_empty():
    store = build_seeded_novelty_store()
    assert run(store.max_similarity(("nonexistent_tool_xyz",))) is not None


def test_eval_policy_uses_the_seeded_store_not_an_empty_one():
    policy = build_eval_policy("balanced")
    action = ProposedAction(tool_name="read_file", arguments={}, agent_reasoning="test")
    decision = run(policy.evaluate(action))
    novelty_result = next(s for s in decision.signals if s.name == "novelty")
    assert novelty_result.informative is True  # real computation, not the empty-store fallback
    assert novelty_result.score == pytest.approx(0.0)  # "read_file" is an exact seed match
