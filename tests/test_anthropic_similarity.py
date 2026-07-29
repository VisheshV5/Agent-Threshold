"""Tests for the Anthropic-backed similarity scorers, written before
the implementation.

Pins the API surface:
- AnthropicSimilarityScorer(client).similarity(a: ProposedAction, b: ProposedAction) -> float
  (implements self_consistency.SimilarityScorer)
- AnthropicTrajectorySimilarityScorer(client).similarity(a: tuple[str,...], b: tuple[str,...]) -> float
  (implements novelty.TrajectorySimilarityScorer)
- Both parse a 0-1 score from the response, falling back to 0.5 (genuinely
  uncertain, not "definitely similar" or "definitely different") on an
  unparseable response
"""

import asyncio

import pytest

from escalation.adapters.similarity import AnthropicSimilarityScorer, AnthropicTrajectorySimilarityScorer
from escalation.types import ProposedAction


def run(coro):
    return asyncio.run(coro)


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    async def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return self.response


def make_action(tool_name: str, arguments: dict | None = None) -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments=arguments or {}, agent_reasoning="test")


# --- self-consistency similarity ---

def test_action_similarity_parses_a_score():
    scorer = AnthropicSimilarityScorer(FakeClient("0.9"))
    result = run(scorer.similarity(make_action("read_file", {"path": "/a"}), make_action("read_file", {"path": "/a"})))
    assert result == pytest.approx(0.9)


def test_action_similarity_includes_both_actions_in_the_prompt():
    client = FakeClient("0.5")
    scorer = AnthropicSimilarityScorer(client)
    run(scorer.similarity(make_action("delete_file", {"path": "/a"}), make_action("send_email", {"to": "x@y.com"})))
    assert "delete_file" in client.prompts[0]
    assert "send_email" in client.prompts[0]
    assert "/a" in client.prompts[0]
    assert "x@y.com" in client.prompts[0]


def test_action_similarity_falls_back_to_half_on_unparseable_response():
    scorer = AnthropicSimilarityScorer(FakeClient("hard to say"))
    result = run(scorer.similarity(make_action("read_file"), make_action("write_file")))
    assert result == pytest.approx(0.5)


# --- novelty trajectory similarity ---

def test_trajectory_similarity_parses_a_score():
    scorer = AnthropicTrajectorySimilarityScorer(FakeClient("0.2"))
    result = run(scorer.similarity(("read_file", "write_file"), ("list_dir", "delete_file")))
    assert result == pytest.approx(0.2)


def test_trajectory_similarity_includes_both_sequences_in_the_prompt():
    client = FakeClient("0.5")
    scorer = AnthropicTrajectorySimilarityScorer(client)
    run(scorer.similarity(("read_file", "write_file"), ("list_dir", "delete_file")))
    assert "read_file" in client.prompts[0]
    assert "write_file" in client.prompts[0]
    assert "list_dir" in client.prompts[0]
    assert "delete_file" in client.prompts[0]


def test_trajectory_similarity_falls_back_to_half_on_unparseable_response():
    scorer = AnthropicTrajectorySimilarityScorer(FakeClient("unclear"))
    result = run(scorer.similarity(("read_file",), ("write_file",)))
    assert result == pytest.approx(0.5)
