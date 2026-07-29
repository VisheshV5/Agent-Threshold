"""Tests for the Anthropic-backed reversibility classifier and blast
radius estimator, written before the implementation.

Pins the API surface:
- AnthropicFallbackClassifier(client).classify(tool_name, arguments) -> ActionCategory
- AnthropicBlastRadiusEstimator(client).estimate(tool_name, arguments) -> float
- Both take an AnthropicClient (or anything with an async .complete()),
  so tests never touch the real API

Category parsing must check EXACT match first, then substring search
ordered by DESCENDING string length -- "irreversible-write" literally
contains "reversible-write" as a substring (i-[rr-e-v-e-r-s-i-b-l-e]-write),
so naive substring search in the wrong order would misclassify every
irreversible-write response as reversible-write.
"""

import asyncio

import pytest

from threshold.adapters.classifiers import AnthropicBlastRadiusEstimator, AnthropicFallbackClassifier
from threshold.signals.reversibility import ActionCategory


def run(coro):
    return asyncio.run(coro)


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    async def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return self.response


# --- classifier ---

@pytest.mark.parametrize(
    "response,expected",
    [
        ("read-only", ActionCategory.READ_ONLY),
        ("reversible-write", ActionCategory.REVERSIBLE_WRITE),
        ("irreversible-write", ActionCategory.IRREVERSIBLE_WRITE),
        ("external-effect", ActionCategory.EXTERNAL_EFFECT),
        ("READ-ONLY", ActionCategory.READ_ONLY),  # case-insensitive
        ("  reversible-write  ", ActionCategory.REVERSIBLE_WRITE),  # whitespace tolerant
    ],
)
def test_classifier_parses_exact_category_responses(response, expected):
    classifier = AnthropicFallbackClassifier(FakeClient(response))
    category = run(classifier.classify("some_tool", {}))
    assert category == expected


def test_classifier_correctly_distinguishes_irreversible_from_reversible_despite_substring_overlap():
    # "irreversible-write" contains "reversible-write" as a literal
    # substring -- this must NOT be misclassified
    classifier = AnthropicFallbackClassifier(FakeClient("Category: irreversible-write."))
    category = run(classifier.classify("some_tool", {}))
    assert category == ActionCategory.IRREVERSIBLE_WRITE


def test_classifier_handles_noisy_response_with_extra_text():
    classifier = AnthropicFallbackClassifier(FakeClient("This looks like a reversible-write operation."))
    category = run(classifier.classify("some_tool", {}))
    assert category == ActionCategory.REVERSIBLE_WRITE


def test_classifier_fails_safe_on_unparseable_response():
    classifier = AnthropicFallbackClassifier(FakeClient("I'm not sure what this does."))
    category = run(classifier.classify("some_tool", {}))
    assert category == ActionCategory.IRREVERSIBLE_WRITE


def test_classifier_sends_tool_name_and_arguments_in_the_prompt():
    client = FakeClient("read-only")
    classifier = AnthropicFallbackClassifier(client)
    run(classifier.classify("read_file", {"path": "/tmp/x"}))
    assert "read_file" in client.prompts[0]
    assert "/tmp/x" in client.prompts[0]


# --- blast radius estimator ---

def test_estimator_parses_a_plain_number():
    estimator = AnthropicBlastRadiusEstimator(FakeClient("0.8"))
    score = run(estimator.estimate("some_tool", {}))
    assert score == pytest.approx(0.8)


def test_estimator_parses_a_number_embedded_in_text():
    estimator = AnthropicBlastRadiusEstimator(FakeClient("I'd estimate this at about 0.35 given the scale."))
    score = run(estimator.estimate("some_tool", {}))
    assert score == pytest.approx(0.35)


def test_estimator_clamps_out_of_range_values():
    estimator = AnthropicBlastRadiusEstimator(FakeClient("1.5"))
    score = run(estimator.estimate("some_tool", {}))
    assert score == pytest.approx(1.0)


def test_estimator_falls_back_to_default_on_unparseable_response():
    estimator = AnthropicBlastRadiusEstimator(FakeClient("I really can't say."))
    score = run(estimator.estimate("some_tool", {}))
    assert score == pytest.approx(0.6)  # same fallback as HeuristicBlastRadiusEstimator
