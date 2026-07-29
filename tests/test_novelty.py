"""Tests for the novelty signal, written before the implementation.

Pins the API surface:
- NoveltyStore(path=None, similarity_scorer=None): in-memory by default,
  only touches disk if a path is explicitly given (same opt-in pattern
  as DecisionLogger)
- NoveltyStore.record_success(tool_names: list[str]) -- the caller's
  explicit judgment that a trajectory completed successfully, not
  something this library infers
- await NoveltyStore.max_similarity(signature) -> float | None; None
  means the store is empty, not "found nothing similar"
- NoveltySignal(store=None).score(action) -> SignalResult; signature is
  the trajectory's tool_name sequence + the proposed action's tool_name,
  deliberately excluding arguments
- HeuristicTrajectorySimilarityScorer: difflib.SequenceMatcher.ratio()
  over tool-name sequences, the M3 stand-in for a future embedding model
- Empty store falls back to MISSING_DATA_SCORE=0.5, not 0.0 or 1.0
"""

import asyncio

import pytest

from escalation.signals.base import Signal
from escalation.signals.novelty import (
    HeuristicTrajectorySimilarityScorer,
    NoveltySignal,
    NoveltyStore,
)
from escalation.types import ProposedAction, Step


def make_action(tool_name: str, trajectory_tools: list[str] | None = None) -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments={},
        agent_reasoning="test",
        trajectory=[Step(tool_name=t, arguments={}, succeeded=True) for t in (trajectory_tools or [])],
    )


def run(coro):
    return asyncio.run(coro)


class StubNoveltyStore:
    """Test double: records the signature it was asked about, returns a fixed similarity."""

    def __init__(self, similarity: float | None):
        self.similarity = similarity
        self.calls: list[tuple[str, ...]] = []

    async def max_similarity(self, signature: tuple[str, ...]) -> float | None:
        self.calls.append(signature)
        return self.similarity


# --- trajectory signature (tested indirectly via what the store is asked) ---

def test_signature_includes_trajectory_tools_then_the_proposed_action():
    stub = StubNoveltyStore(similarity=0.5)
    signal = NoveltySignal(store=stub)
    action = make_action("write_file", trajectory_tools=["list_dir", "read_file"])
    run(signal.score(action))
    assert stub.calls == [("list_dir", "read_file", "write_file")]


def test_signature_with_no_trajectory_is_just_the_proposed_action():
    stub = StubNoveltyStore(similarity=0.5)
    signal = NoveltySignal(store=stub)
    action = make_action("read_file")
    run(signal.score(action))
    assert stub.calls == [("read_file",)]


# --- NoveltySignal scoring ---

def test_empty_store_returns_missing_data_score():
    signal = NoveltySignal(store=NoveltyStore())
    result = run(signal.score(make_action("read_file")))
    assert result.score == pytest.approx(0.5)
    assert "empty" in result.reason.lower()


def test_exact_match_to_a_past_trajectory_is_not_novel():
    store = NoveltyStore()
    store.record_success(["list_dir", "read_file"])
    signal = NoveltySignal(store=store)
    result = run(signal.score(make_action("read_file", trajectory_tools=["list_dir"])))
    assert result.score == pytest.approx(0.0)


def test_completely_disjoint_trajectory_is_maximally_novel():
    store = NoveltyStore()
    store.record_success(["list_dir", "read_file"])
    signal = NoveltySignal(store=store)
    result = run(signal.score(make_action("execute_trade", trajectory_tools=["check_market_conditions"])))
    assert result.score == pytest.approx(1.0)


def test_nearest_neighbor_not_average_across_the_store():
    store = NoveltyStore()
    store.record_success(["totally", "different", "sequence"])
    store.record_success(["read_file"])  # exact match to the query below
    signal = NoveltySignal(store=store)
    result = run(signal.score(make_action("read_file")))
    assert result.score == pytest.approx(0.0)  # closest match wins, not averaged with the disjoint one


def test_signal_is_a_signal_with_expected_name_and_cost():
    signal = NoveltySignal(store=NoveltyStore())
    result = run(signal.score(make_action("read_file")))
    assert isinstance(signal, Signal)
    assert result.name == "novelty"
    assert result.cost_ms >= 0


# --- NoveltyStore behavior ---

def test_store_with_no_path_does_not_touch_disk(tmp_path):
    store = NoveltyStore()
    store.record_success(["read_file"])
    assert list(tmp_path.iterdir()) == []


def test_store_persists_and_reloads_from_path(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    store1 = NoveltyStore(path=path)
    store1.record_success(["list_dir", "read_file"])
    store1.record_success(["write_file"])

    store2 = NoveltyStore(path=path)  # fresh instance, same file
    similarity = run(store2.max_similarity(("list_dir", "read_file")))
    assert similarity == pytest.approx(1.0)


def test_max_similarity_is_none_for_empty_store():
    store = NoveltyStore()
    assert run(store.max_similarity(("read_file",))) is None


# --- default heuristic scorer, tested directly ---

def test_heuristic_scorer_identical_sequences_is_one():
    scorer = HeuristicTrajectorySimilarityScorer()
    assert run(scorer.similarity(("a", "b"), ("a", "b"))) == pytest.approx(1.0)


def test_heuristic_scorer_disjoint_sequences_is_zero():
    scorer = HeuristicTrajectorySimilarityScorer()
    assert run(scorer.similarity(("a", "b"), ("c", "d"))) == pytest.approx(0.0)


def test_heuristic_scorer_partial_overlap():
    scorer = HeuristicTrajectorySimilarityScorer()
    # one matching token ("read_file") out of 2+2 total -> ratio = 2*1/4 = 0.5
    result = run(scorer.similarity(("read_file", "write_file"), ("read_file", "delete_file")))
    assert result == pytest.approx(0.5)
