"""Threshold presets for Policy.from_profile().

Profiles share weights for now (only the reversibility signal exists);
they differ only in how eager the threshold is to escalate. Hard-override
categories apply identically across all profiles -- that's a global
invariant, not something a profile can loosen.
"""

PROFILE_THRESHOLDS: dict[str, float] = {
    "conservative": 0.1,
    "balanced": 0.5,
    "autonomous": 0.8,
}
