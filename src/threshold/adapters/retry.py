"""Resilience helper found necessary live, not anticipated in design:
some Anthropic models reject the `temperature` parameter entirely
(400: "temperature is deprecated for this model") rather than silently
ignoring or clamping it. Rather than guess which models do or don't
accept it, retry once without temperature specifically when that's the
rejection reason -- any other error is never masked.
"""

from typing import Any, Protocol

import anthropic


class _RawMessages(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


async def create_with_temperature_fallback(messages_client: _RawMessages, **kwargs: Any) -> Any:
    try:
        return await messages_client.create(**kwargs)
    except anthropic.BadRequestError as exc:
        if "temperature" in kwargs and "temperature" in str(exc).lower():
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
            return await messages_client.create(**fallback_kwargs)
        raise
