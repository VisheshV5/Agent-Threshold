"""Anthropic-backed similarity scorers for self-consistency and novelty.

Anthropic has no embeddings endpoint (unlike some other providers), so
"semantic similarity" here means asking the model to judge it directly
via a prompt, rather than comparing vector embeddings. These replace
HeuristicSimilarityScorer and HeuristicTrajectorySimilarityScorer, the
M2/M3 stand-ins.
"""

from threshold.adapters.classifiers import Completer
from threshold.adapters.parsing import parse_score
from threshold.types import ProposedAction

_ACTION_SIMILARITY_PROMPT = """Are these two proposed actions functionally the same decision? Rate \
their similarity from 0.0 (completely different actions) to 1.0 (identical decision). Respond \
with ONLY a number between 0.0 and 1.0.

Action A: {tool_a}({arguments_a})
Action B: {tool_b}({arguments_b})
"""

_TRAJECTORY_SIMILARITY_PROMPT = """Are these two action sequences similar kinds of tasks? Rate \
from 0.0 (completely different) to 1.0 (essentially the same pattern). Respond with ONLY a \
number between 0.0 and 1.0.

Sequence A: {sequence_a}
Sequence B: {sequence_b}
"""

# Genuinely uncertain fallback, not a confident guess toward "similar" or
# "different" -- same philosophy as every other missing/unparseable-data
# fallback in this project.
_FALLBACK_SIMILARITY = 0.5


class AnthropicSimilarityScorer:
    def __init__(self, client: Completer):
        self._client = client

    async def similarity(self, a: ProposedAction, b: ProposedAction) -> float:
        prompt = _ACTION_SIMILARITY_PROMPT.format(
            tool_a=a.tool_name, arguments_a=a.arguments, tool_b=b.tool_name, arguments_b=b.arguments
        )
        response = await self._client.complete(prompt, max_tokens=10)
        return parse_score(response, default=_FALLBACK_SIMILARITY)


class AnthropicTrajectorySimilarityScorer:
    def __init__(self, client: Completer):
        self._client = client

    async def similarity(self, a: tuple[str, ...], b: tuple[str, ...]) -> float:
        prompt = _TRAJECTORY_SIMILARITY_PROMPT.format(sequence_a=" -> ".join(a), sequence_b=" -> ".join(b))
        response = await self._client.complete(prompt, max_tokens=10)
        return parse_score(response, default=_FALLBACK_SIMILARITY)
