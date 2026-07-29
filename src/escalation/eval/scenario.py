"""Scenario format: a mock agent situation plus a ground-truth escalation label."""

from typing import Literal

from pydantic import BaseModel, model_validator

from escalation.types import ProposedAction

ScenarioCategory = Literal["safe", "dangerous", "ambiguous", "adversarial"]


class Scenario(BaseModel):
    """One hand-written eval case: a proposed action plus what should happen.

    should_escalate collapses Decision.verdict's three-way outcome into a
    binary for metrics purposes (false alarm rate / miss rate both compare
    against this, not against a specific verdict) -- M1's policy never
    actually produces "abort", so this loses nothing yet.

    expected_hard_override tags whether this scenario's ground truth is
    that a hard-override category (irreversible-write / external-effect)
    should fire, so the M2 calibration curve can report threshold-sensitive
    and hard-override scenarios separately instead of one curve where the
    hard-override cases create a flat floor/ceiling.
    """

    id: str
    description: str
    category: ScenarioCategory
    action: ProposedAction
    should_escalate: bool
    expected_hard_override: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def hard_override_implies_escalate(self) -> "Scenario":
        if self.expected_hard_override and not self.should_escalate:
            raise ValueError("expected_hard_override=True requires should_escalate=True")
        return self
