"""Tests for the thin Anthropic SDK wrapper, written before the implementation.

Pins the API surface:
- AnthropicClient(raw_client, model).complete(prompt, max_tokens=256, temperature=0.0) -> str
- Delegates to raw_client.messages.create(...) and extracts .content[0].text
- Everything else in the adapter package depends on this one method, not
  the full SDK surface -- easy to mock, one place to change if the SDK
  shape changes.
"""

import asyncio

from escalation.adapters.anthropic_client import AnthropicClient


def run(coro):
    return asyncio.run(coro)


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.response_text)


class _FakeRawClient:
    def __init__(self, response_text: str):
        self.messages = _FakeMessages(response_text)


def test_complete_returns_the_response_text():
    raw = _FakeRawClient("read-only")
    client = AnthropicClient(raw_client=raw, model="claude-sonnet-5")
    result = run(client.complete("classify this"))
    assert result == "read-only"


def test_complete_passes_model_prompt_and_params_through():
    raw = _FakeRawClient("ok")
    client = AnthropicClient(raw_client=raw, model="claude-sonnet-5")
    run(client.complete("hello", max_tokens=50, temperature=0.7))

    call = raw.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 50
    assert call["temperature"] == 0.7
    assert call["messages"] == [{"role": "user", "content": "hello"}]


def test_complete_uses_default_max_tokens_and_temperature():
    raw = _FakeRawClient("ok")
    client = AnthropicClient(raw_client=raw, model="claude-sonnet-5")
    run(client.complete("hello"))

    call = raw.messages.calls[0]
    assert call["max_tokens"] == 256
    assert call["temperature"] == 0.0
