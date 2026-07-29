"""Tests for the temperature-rejection retry helper, written before the
implementation.

Found live while running the demo: some Anthropic models reject the
`temperature` parameter entirely (400: "temperature is deprecated for
this model") rather than silently ignoring it. Pins
create_with_temperature_fallback(messages_client, **kwargs): retries
once without temperature specifically when that's the rejection
reason, and never masks any other error.
"""

import asyncio

import anthropic
import pytest

from threshold.adapters.retry import create_with_temperature_fallback


def run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    pass


def _make_bad_request_error(message: str) -> anthropic.BadRequestError:
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, json={"type": "error", "error": {"type": "invalid_request_error", "message": message}})
    return anthropic.BadRequestError(message=message, response=response, body={"error": {"message": message}})


class FakeMessagesRejectingTemperature:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise _make_bad_request_error("`temperature` is deprecated for this model.")
        return _FakeResponse()


class FakeMessagesRejectingSomethingElse:
    async def create(self, **kwargs):
        raise _make_bad_request_error("max_tokens is required.")


class FakeMessagesThatWorks:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


def test_retries_without_temperature_on_the_specific_rejection():
    client = FakeMessagesRejectingTemperature()
    result = run(create_with_temperature_fallback(client, model="m", max_tokens=10, temperature=0.7))
    assert isinstance(result, _FakeResponse)
    assert len(client.calls) == 2
    assert "temperature" in client.calls[0]
    assert "temperature" not in client.calls[1]


def test_does_not_retry_for_an_unrelated_bad_request_error():
    client = FakeMessagesRejectingSomethingElse()
    with pytest.raises(anthropic.BadRequestError):
        run(create_with_temperature_fallback(client, model="m", max_tokens=10, temperature=0.7))


def test_succeeds_on_the_first_try_when_temperature_is_accepted():
    client = FakeMessagesThatWorks()
    result = run(create_with_temperature_fallback(client, model="m", max_tokens=10, temperature=0.7))
    assert isinstance(result, _FakeResponse)
    assert len(client.calls) == 1


def test_does_not_retry_if_temperature_was_never_passed():
    client = FakeMessagesRejectingSomethingElse()
    with pytest.raises(anthropic.BadRequestError):
        run(create_with_temperature_fallback(client, model="m", max_tokens=10))
