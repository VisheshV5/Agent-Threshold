"""Threshold and weight presets for Policy.from_profile().

Profiles share weights; they differ only in how eager the threshold is
to escalate. Hard-override categories apply identically across all
profiles -- that's a global invariant, not something a profile can loosen.
"""

PROFILE_THRESHOLDS: dict[str, float] = {
    "conservative": 0.1,
    "balanced": 0.5,
    "autonomous": 0.8,
}

# self_consistency gets equal billing with reversibility (the spec calls
# it "the strongest single signal"); blast_radius gets less weight since
# its fallback (0.6, for arguments it can't parse) is a fairly aggressive
# default that shouldn't dominate the aggregate on its own.
DEFAULT_WEIGHTS: dict[str, float] = {
    "reversibility": 0.4,
    "self_consistency": 0.4,
    "blast_radius": 0.2,
}
