"""Blast radius signal: how many records/entities/dollars/people an action touches.

Parses a count from the action's arguments where possible, falling
through to an injectable estimator otherwise. The default estimator is a
heuristic -- like reversibility's fallback, a stand-in for the real LLM
estimate that doesn't land until the Anthropic adapter (Milestone 4).
"""

import time
from typing import Protocol

from threshold.signals.base import Signal
from threshold.types import ProposedAction, SignalResult

_COUNT_KEYS = {"count", "quantity", "amount", "num_records", "num_items", "num_entities"}
_WILDCARD_CHARS = ("*", "?")

# Log-scale buckets: (inclusive upper bound, score). Anything above the
# last bound scores 1.0. Tunable constants, same pattern as
# reversibility's CATEGORY_SCORES.
_BUCKETS: list[tuple[int, float]] = [(1, 0.0), (9, 0.3), (99, 0.6)]


def _count_to_score(count: int) -> float:
    for upper, score in _BUCKETS:
        if count <= upper:
            return score
    return 1.0


def _is_wildcard_string(value: object) -> bool:
    return isinstance(value, str) and any(ch in value for ch in _WILDCARD_CHARS)


def _parse_count(arguments: dict) -> int | None:
    if not arguments:
        return None

    list_lengths = [len(v) for v in arguments.values() if isinstance(v, list)]
    if list_lengths:
        return max(list_lengths)

    for key in _COUNT_KEYS:
        value = arguments.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)

    if any(_is_wildcard_string(v) for v in arguments.values()):
        return None  # ambiguous magnitude (e.g. a glob pattern), not a count of one

    return 1  # only plain scalar arguments present -> touches exactly one thing


class BlastRadiusEstimator(Protocol):
    async def estimate(self, tool_name: str, arguments: dict) -> float: ...


class HeuristicBlastRadiusEstimator:
    """M2 stand-in for an LLM estimate.

    Defaults to 0.6 (the [10,99] bucket) rather than 0.0 or 1.0: when the
    magnitude genuinely can't be parsed, assume it's worth a second look
    without assuming worst-case.
    """

    FALLBACK_SCORE = 0.6

    async def estimate(self, tool_name: str, arguments: dict) -> float:
        return self.FALLBACK_SCORE


class BlastRadiusSignal(Signal):
    name = "blast_radius"

    def __init__(self, estimator: BlastRadiusEstimator | None = None):
        self._estimator: BlastRadiusEstimator = estimator if estimator is not None else HeuristicBlastRadiusEstimator()

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()
        count = _parse_count(action.arguments)
        if count is not None:
            score = _count_to_score(count)
            reason = f"parsed {count} entities from arguments -> blast radius {score}"
        else:
            score = await self._estimator.estimate(action.tool_name, action.arguments)
            reason = f"could not parse a count from arguments; fallback estimate {score}"
        cost_ms = int((time.perf_counter() - start) * 1000)
        return SignalResult(name=self.name, score=score, reason=reason, cost_ms=cost_ms)
