"""Novelty signal: how far the current trajectory is from anything that
has completed successfully before.

Unlike every other signal, this one is stateful -- it compares against a
growing store of past trajectories rather than being a pure function of
the current action. The signature compared is deliberately coarse: the
ordered sequence of tool_names (trajectory + the proposed action),
excluding arguments entirely. That's a real scope limitation, not an
oversight -- a completely ordinary-looking trajectory with unusual
argument values still looks "not novel" here; catching that is
self-consistency's and blast radius's job, not this one's.

"Completed successfully" is never this library's judgment -- the caller
records it explicitly via NoveltyStore.record_success() after a run
finishes well.
"""

import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

from escalation.signals.base import Signal
from escalation.types import ProposedAction, SignalResult


def _trajectory_signature(action: ProposedAction) -> tuple[str, ...]:
    return tuple(step.tool_name for step in action.trajectory) + (action.tool_name,)


class TrajectorySimilarityScorer(Protocol):
    async def similarity(self, a: tuple[str, ...], b: tuple[str, ...]) -> float: ...


class HeuristicTrajectorySimilarityScorer:
    """M3 stand-in for a future embedding-based comparison.

    difflib.SequenceMatcher.ratio() over tool-name sequences: order-
    sensitive, deterministic, no new dependency.
    """

    async def similarity(self, a: tuple[str, ...], b: tuple[str, ...]) -> float:
        return SequenceMatcher(None, a, b).ratio()


class NoveltyStore:
    """In-memory by default; only touches disk if given an explicit path,
    same opt-in persistence pattern as DecisionLogger."""

    def __init__(
        self,
        path: str | Path | None = None,
        similarity_scorer: TrajectorySimilarityScorer | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self._similarity_scorer: TrajectorySimilarityScorer = (
            similarity_scorer if similarity_scorer is not None else HeuristicTrajectorySimilarityScorer()
        )
        self._trajectories: list[tuple[str, ...]] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self._trajectories.append(tuple(record["tool_names"]))

    def record_success(self, tool_names: list[str]) -> None:
        signature = tuple(tool_names)
        self._trajectories.append(signature)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"tool_names": list(signature)}) + "\n")

    async def max_similarity(self, signature: tuple[str, ...]) -> float | None:
        if not self._trajectories:
            return None
        similarities = [
            await self._similarity_scorer.similarity(signature, past) for past in self._trajectories
        ]
        return max(similarities)


class NoveltySignal(Signal):
    name = "novelty"

    MISSING_DATA_SCORE = 0.5

    def __init__(self, store: NoveltyStore | None = None):
        self._store = store if store is not None else NoveltyStore()

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()
        signature = _trajectory_signature(action)
        similarity = await self._store.max_similarity(signature)

        if similarity is None:
            cost_ms = int((time.perf_counter() - start) * 1000)
            return SignalResult(
                name=self.name,
                score=self.MISSING_DATA_SCORE,
                reason="novelty store is empty; nothing to compare against",
                cost_ms=cost_ms,
                informative=False,
            )

        score = 1.0 - similarity
        cost_ms = int((time.perf_counter() - start) * 1000)
        return SignalResult(
            name=self.name,
            score=score,
            reason=f"closest past successful trajectory has similarity {similarity:.2f} -> novelty {score:.2f}",
            cost_ms=cost_ms,
        )
