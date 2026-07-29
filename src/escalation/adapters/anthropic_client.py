"""Thin wrapper around the Anthropic SDK.

Every LLM-backed piece of the adapter package (classifier, estimator,
similarity scorers, agent) depends on this one complete() method rather
than the full SDK surface -- easy to mock in tests, one place to change
if the SDK's shape changes.
"""

from typing import Any, Protocol

from escalation.adapters.retry import create_with_temperature_fallback


class _RawMessages(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _RawClient(Protocol):
    messages: _RawMessages


class AnthropicClient:
    def __init__(self, raw_client: _RawClient, model: str = "claude-sonnet-5"):
        self._raw_client = raw_client
        self._model = model

    async def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        response = await create_with_temperature_fallback(
            self._raw_client.messages,
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
