"""Thrash signal: repeated failed attempts at the same subgoal.

Counts CONSECUTIVE trailing failures of the same tool_name immediately
preceding the proposed action -- not scattered failures elsewhere in the
trajectory, and a succeeded=None (unknown outcome) step breaks the
streak rather than counting as a confirmed failure. An agent about to
retry a tool that just failed several times in a row is an agent that
should be asking for help rather than trying again silently.
"""

import time

from escalation.signals.base import Signal
from escalation.types import ProposedAction, SignalResult

_BUCKETS: list[tuple[int, float]] = [(0, 0.0), (1, 0.3), (2, 0.6)]


def _count_to_score(count: int) -> float:
    for upper, score in _BUCKETS:
        if count <= upper:
            return score
    return 1.0


def _consecutive_trailing_failures(action: ProposedAction) -> int:
    count = 0
    for step in reversed(action.trajectory):
        if step.tool_name == action.tool_name and step.succeeded is False:
            count += 1
        else:
            break
    return count


class ThrashSignal(Signal):
    name = "thrash"

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()
        count = _consecutive_trailing_failures(action)
        score = _count_to_score(count)
        cost_ms = int((time.perf_counter() - start) * 1000)
        reason = (
            f"'{action.tool_name}' failed {count} time(s) in a row immediately before this retry -> thrash {score}"
            if count > 0
            else f"no prior consecutive failures of '{action.tool_name}' -> thrash 0.0"
        )
        return SignalResult(name=self.name, score=score, reason=reason, cost_ms=cost_ms)
