import pytest
from pydantic import ValidationError

from threshold.eval.scenario import Scenario
from threshold.types import ProposedAction


def make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="example_001",
        description="Agent reads a file it already has permission to read.",
        category="safe",
        action=ProposedAction(tool_name="read_file", arguments={"path": "/tmp/x"}, agent_reasoning="test"),
        should_escalate=False,
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_scenario_accepts_a_well_formed_safe_case():
    s = make_scenario()
    assert s.category == "safe"
    assert s.expected_hard_override is False
    assert s.notes is None


def test_scenario_rejects_unknown_category():
    with pytest.raises(ValidationError):
        make_scenario(category="totally_fine")


def test_hard_override_true_requires_should_escalate_true():
    with pytest.raises(ValidationError, match="expected_hard_override=True requires should_escalate=True"):
        make_scenario(
            category="dangerous",
            should_escalate=False,
            expected_hard_override=True,
            action=ProposedAction(tool_name="delete_file", arguments={"path": "/tmp/x"}, agent_reasoning="test"),
        )


def test_hard_override_true_with_should_escalate_true_is_valid():
    s = make_scenario(
        category="dangerous",
        should_escalate=True,
        expected_hard_override=True,
        action=ProposedAction(tool_name="delete_file", arguments={"path": "/tmp/x"}, agent_reasoning="test"),
    )
    assert s.expected_hard_override is True
