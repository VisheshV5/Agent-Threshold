"""Anthropic-backed reversibility classifier and blast radius estimator.

These are the real-LLM implementations of the FallbackClassifier and
BlastRadiusEstimator protocols -- the pieces that were stood in for by
HeuristicFallbackClassifier and HeuristicBlastRadiusEstimator since
Milestone 1/2, pending this adapter. Drop-in replacements: same
protocol, same fail-safe philosophy on an unparseable response.
"""

from typing import Protocol

from escalation.adapters.parsing import parse_score
from escalation.signals.blast_radius import HeuristicBlastRadiusEstimator
from escalation.signals.reversibility import ActionCategory

_CATEGORY_BY_VALUE = {category.value: category for category in ActionCategory}
_CATEGORIES_BY_DESCENDING_LENGTH = sorted(ActionCategory, key=lambda c: -len(c.value))


class Completer(Protocol):
    async def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str: ...


def _parse_category(text: str) -> ActionCategory:
    normalized = text.strip().lower()
    if normalized in _CATEGORY_BY_VALUE:
        return _CATEGORY_BY_VALUE[normalized]
    # substring fallback for a noisy response, longest category name
    # first: "irreversible-write" literally contains "reversible-write"
    # as a substring, so checking shorter names first would misfire.
    for category in _CATEGORIES_BY_DESCENDING_LENGTH:
        if category.value in normalized:
            return category
    return ActionCategory.IRREVERSIBLE_WRITE  # fail-safe, same as the heuristic


_CLASSIFY_PROMPT = """Classify this tool call by whether its effect can be undone. Respond with \
ONLY one of these four category names, nothing else: read-only, reversible-write, \
irreversible-write, external-effect.

Tool: {tool_name}
Arguments: {arguments}
"""

_ESTIMATE_PROMPT = """Estimate how many records, entities, dollars, or people this tool call \
affects, on a scale from 0.0 (trivial, a single item) to 1.0 (massive scale, 100+). Respond \
with ONLY a number between 0.0 and 1.0, nothing else.

Tool: {tool_name}
Arguments: {arguments}
"""


class AnthropicFallbackClassifier:
    def __init__(self, client: Completer):
        self._client = client

    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory:
        prompt = _CLASSIFY_PROMPT.format(tool_name=tool_name, arguments=arguments)
        response = await self._client.complete(prompt, max_tokens=20)
        return _parse_category(response)


class AnthropicBlastRadiusEstimator:
    def __init__(self, client: Completer):
        self._client = client

    async def estimate(self, tool_name: str, arguments: dict) -> float:
        prompt = _ESTIMATE_PROMPT.format(tool_name=tool_name, arguments=arguments)
        response = await self._client.complete(prompt, max_tokens=10)
        return parse_score(response, default=HeuristicBlastRadiusEstimator.FALLBACK_SCORE)
