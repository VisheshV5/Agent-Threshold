"""Staleness signal: flags when the action depends on a fact fetched
many steps ago that could plausibly have changed since.

context_age_steps is plain data already present on every ProposedAction
(unlike self-consistency/novelty, nothing extra needs to be gathered),
so an empty dict is read as "this decision doesn't depend on any
previously-verified fact" (score 0.0), not as missing data needing a
cautious fallback. Uses the single most-stale fact (max steps since
verified), matching the "worst case drives the score" logic used
elsewhere (blast radius's longest list, novelty's nearest neighbor).
"""

import time

from escalation.signals.base import Signal
from escalation.types import ProposedAction, SignalResult

# Log-scale buckets on "steps since last verified": (inclusive upper bound, score).
_BUCKETS: list[tuple[int, float]] = [(1, 0.0), (4, 0.3), (14, 0.6)]


def _age_to_score(steps: int) -> float:
    for upper, score in _BUCKETS:
        if steps <= upper:
            return score
    return 1.0


class StalenessSignal(Signal):
    name = "staleness"

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()

        if not action.context_age_steps:
            cost_ms = int((time.perf_counter() - start) * 1000)
            return SignalResult(
                name=self.name,
                score=0.0,
                reason="no tracked facts depend on prior context",
                cost_ms=cost_ms,
            )

        stalest_fact, stalest_age = max(action.context_age_steps.items(), key=lambda kv: kv[1])
        score = _age_to_score(stalest_age)
        cost_ms = int((time.perf_counter() - start) * 1000)
        return SignalResult(
            name=self.name,
            score=score,
            reason=f"'{stalest_fact}' last verified {stalest_age} step(s) ago -> staleness {score}",
            cost_ms=cost_ms,
        )
