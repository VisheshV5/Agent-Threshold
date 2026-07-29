"""Shared response-parsing helpers for Anthropic-backed adapters."""

import re

_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def parse_score(text: str, default: float) -> float:
    match = _NUMBER_PATTERN.search(text)
    if not match:
        return default
    try:
        value = float(match.group())
    except ValueError:
        return default
    return max(0.0, min(1.0, value))
