"""Threshold and weight presets for Policy.from_profile().

Profiles share weights; they differ only in how eager the threshold is
to escalate. Hard-override categories apply identically across all
profiles -- that's a global invariant, not something a profile can loosen.
"""

PROFILE_THRESHOLDS: dict[str, float] = {
    "conservative": 0.05,
    "balanced": 0.25,
    "autonomous": 0.6,
}

# reversibility + self_consistency still anchor the aggregate (the spec
# calls self-consistency "the strongest single signal"), diluted from
# 80% combined (when there were 3 signals) to 50% now that 4 more
# genuinely contribute. blast_radius and staleness are next -- both
# deterministic given real data, comparable trust. thrash and novelty
# are lowest: thrash triggers rarely (most actions have zero consecutive
# failures to detect), and novelty is the least mature signal right now
# (heuristic comparison, and typically an empty store in practice).
DEFAULT_WEIGHTS: dict[str, float] = {
    "reversibility": 0.25,
    "self_consistency": 0.25,
    "blast_radius": 0.15,
    "staleness": 0.15,
    "thrash": 0.10,
    "novelty": 0.10,
}
