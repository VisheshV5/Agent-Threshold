"""Base interface every signal implements."""

from abc import ABC, abstractmethod

from threshold.types import ProposedAction, SignalResult


class Signal(ABC):
    """An independent scorer: given a proposed action, returns 0-1 plus a reason.

    Async even where a given signal (e.g. reversibility) never awaits
    anything internally, because self-consistency, blast-radius, and
    novelty (M2/M3) all need real LLM/embedding calls.
    """

    name: str

    @abstractmethod
    async def score(self, action: ProposedAction) -> SignalResult: ...
