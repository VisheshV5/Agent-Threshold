"""Self-consistency signal: how much the agent agrees with itself.

Compares the proposed action against ProposedAction.alternative_actions
(pre-sampled candidates for the same decision point -- see types.py for
why this is plain data rather than a live resampling hook). Score is
1 - average pairwise similarity across the full sample set (the
proposed action plus all alternatives), using an injectable
SimilarityScorer. No alternatives available means self-consistency
genuinely could not be assessed, not that it was checked and passed --
so that case scores MISSING_DATA_SCORE (0.5), never 0.0.
"""

import itertools
import time
from typing import Protocol

from escalation.signals.base import Signal
from escalation.types import ProposedAction, SignalResult


class SimilarityScorer(Protocol):
    async def similarity(self, a: ProposedAction, b: ProposedAction) -> float: ...


def _argument_similarity(a: dict, b: dict) -> float:
    pairs_a = {(key, repr(value)) for key, value in a.items()}
    pairs_b = {(key, repr(value)) for key, value in b.items()}
    union = pairs_a | pairs_b
    if not union:
        return 1.0  # both empty -> nothing to disagree on
    intersection = pairs_a & pairs_b
    return len(intersection) / len(union)


class HeuristicSimilarityScorer:
    """M2 stand-in for an embedding-based comparison.

    Structural only: same tool_name and overlapping arguments count as
    agreement, ignoring agent_reasoning entirely -- consistency is about
    whether the agent would take the same action, not how it justifies it.
    """

    async def similarity(self, a: ProposedAction, b: ProposedAction) -> float:
        if a.tool_name != b.tool_name:
            return 0.0
        return _argument_similarity(a.arguments, b.arguments)


class SelfConsistencySignal(Signal):
    name = "self_consistency"

    MISSING_DATA_SCORE = 0.5

    def __init__(self, similarity_scorer: SimilarityScorer | None = None):
        self._similarity_scorer: SimilarityScorer = (
            similarity_scorer if similarity_scorer is not None else HeuristicSimilarityScorer()
        )

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()
        samples = [action, *action.alternative_actions]

        if len(samples) < 2:
            cost_ms = int((time.perf_counter() - start) * 1000)
            return SignalResult(
                name=self.name,
                score=self.MISSING_DATA_SCORE,
                reason="no alternative samples available; self-consistency could not be assessed",
                cost_ms=cost_ms,
                informative=False,
            )

        similarities = [
            await self._similarity_scorer.similarity(x, y) for x, y in itertools.combinations(samples, 2)
        ]
        avg_similarity = sum(similarities) / len(similarities)
        score = 1.0 - avg_similarity
        cost_ms = int((time.perf_counter() - start) * 1000)
        return SignalResult(
            name=self.name,
            score=score,
            reason=(
                f"{len(action.alternative_actions)} alternative sample(s), "
                f"avg pairwise agreement {avg_similarity:.2f} -> self-consistency score {score:.2f}"
            ),
            cost_ms=cost_ms,
        )
