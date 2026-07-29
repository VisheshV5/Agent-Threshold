"""Reversibility signal: classifies a tool call by whether its effect can be undone.

Static registry for known tools, falling through to an injectable
FallbackClassifier for unknown ones. The default fallback is a keyword
heuristic (HeuristicFallbackClassifier) -- a stand-in for the real LLM
classifier, which doesn't land until the Anthropic adapter (Milestone 4).
"""

import time
from enum import Enum
from typing import Protocol

from escalation.signals.base import Signal
from escalation.types import ProposedAction, SignalResult


class ActionCategory(str, Enum):
    READ_ONLY = "read-only"
    REVERSIBLE_WRITE = "reversible-write"
    IRREVERSIBLE_WRITE = "irreversible-write"
    EXTERNAL_EFFECT = "external-effect"


# Categories that force a hard override in the policy layer: always
# ask_human regardless of aggregate score, and always excluded from the
# decision cache (never resolved from a past answer, no exceptions).
HARD_OVERRIDE_CATEGORIES = {
    ActionCategory.IRREVERSIBLE_WRITE,
    ActionCategory.EXTERNAL_EFFECT,
}

CATEGORY_SCORES: dict[ActionCategory, float] = {
    ActionCategory.READ_ONLY: 0.0,
    ActionCategory.REVERSIBLE_WRITE: 0.35,
    ActionCategory.IRREVERSIBLE_WRITE: 0.75,
    ActionCategory.EXTERNAL_EFFECT: 1.0,
}

# Tools we know about ahead of time; deterministic, never keyword-guessed.
DEFAULT_REGISTRY: dict[str, ActionCategory] = {
    "read_file": ActionCategory.READ_ONLY,
    "write_file": ActionCategory.REVERSIBLE_WRITE,
    "delete_file": ActionCategory.IRREVERSIBLE_WRITE,
    "send_email": ActionCategory.EXTERNAL_EFFECT,
}


class FallbackClassifier(Protocol):
    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory: ...


# Keywords describe the verb of crossing a boundary (EXTERNAL_EFFECT) or
# of destroying internal state (IRREVERSIBLE_WRITE), never the object
# being acted on -- e.g. "message" alone is not an EXTERNAL_EFFECT
# keyword, so "delete_message" doesn't false-positive to sent-to-a-third-party.
_KEYWORDS: dict[ActionCategory, set[str]] = {
    ActionCategory.READ_ONLY: {
        "read", "get", "list", "search", "fetch", "view", "check", "query",
    },
    ActionCategory.REVERSIBLE_WRITE: {
        "write", "edit", "update", "create", "set", "save", "add", "reset",
    },
    ActionCategory.IRREVERSIBLE_WRITE: {
        "delete", "drop", "remove", "destroy", "purge", "wipe", "truncate", "revoke",
    },
    ActionCategory.EXTERNAL_EFFECT: {
        "send", "post", "publish", "pay", "transfer", "notify", "email", "broadcast", "submit",
    },
}

# Checked most-severe-first: if a tool name contains keywords from more
# than one category (e.g. "edit_and_delete_entry" has both "edit" and
# "delete"), the more dangerous category wins rather than being masked
# by a milder keyword also present.
_CHECK_ORDER = [
    ActionCategory.EXTERNAL_EFFECT,
    ActionCategory.IRREVERSIBLE_WRITE,
    ActionCategory.REVERSIBLE_WRITE,
    ActionCategory.READ_ONLY,
]


class HeuristicFallbackClassifier:
    """M1 stand-in for an LLM classifier: whole-token keyword matching.

    Fails safe: a tool name with no recognized keyword is treated as
    IRREVERSIBLE_WRITE rather than waved through as read-only.
    """

    async def classify(self, tool_name: str, arguments: dict) -> ActionCategory:
        tokens = set(tool_name.lower().split("_"))
        for category in _CHECK_ORDER:
            if tokens & _KEYWORDS[category]:
                return category
        return ActionCategory.IRREVERSIBLE_WRITE


class ReversibilitySignal(Signal):
    name = "reversibility"

    def __init__(
        self,
        registry: dict[str, ActionCategory] | None = None,
        fallback: FallbackClassifier | None = None,
    ):
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._fallback: FallbackClassifier = fallback if fallback is not None else HeuristicFallbackClassifier()

    async def classify(self, action: ProposedAction) -> tuple[ActionCategory, str]:
        if action.tool_name in self._registry:
            return self._registry[action.tool_name], "registry"
        category = await self._fallback.classify(action.tool_name, action.arguments)
        return category, "fallback"

    async def score(self, action: ProposedAction) -> SignalResult:
        start = time.perf_counter()
        category, source = await self.classify(action)
        cost_ms = int((time.perf_counter() - start) * 1000)
        return SignalResult(
            name=self.name,
            score=CATEGORY_SCORES[category],
            reason=f"'{action.tool_name}' classified as {category.value} via {source}",
            cost_ms=cost_ms,
        )
