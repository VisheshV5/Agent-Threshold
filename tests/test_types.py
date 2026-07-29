import pytest
from pydantic import ValidationError

from escalation.types import HumanQuestion, ProposedAction


def test_human_question_accepts_recommended_option_in_options():
    q = HumanQuestion(
        summary="About to delete /tmp/x",
        concern="irreversible-write",
        options=["delete", "skip"],
        recommended_option="skip",
    )
    assert q.recommended_option == "skip"


def test_human_question_rejects_recommended_option_not_in_options():
    with pytest.raises(ValidationError, match="recommended_option must be one of options"):
        HumanQuestion(
            summary="About to delete /tmp/x",
            concern="irreversible-write",
            options=["delete", "skip"],
            recommended_option="proceed",
        )


def test_proposed_action_alternative_actions_defaults_empty():
    action = ProposedAction(tool_name="read_file", arguments={}, agent_reasoning="test")
    assert action.alternative_actions == []


def test_proposed_action_accepts_nested_alternative_actions():
    alt = ProposedAction(tool_name="delete_file", arguments={"path": "/a"}, agent_reasoning="alt")
    primary = ProposedAction(
        tool_name="delete_file",
        arguments={"path": "/b"},
        agent_reasoning="primary",
        alternative_actions=[alt],
    )
    assert len(primary.alternative_actions) == 1
    assert primary.alternative_actions[0].tool_name == "delete_file"
    assert primary.alternative_actions[0].arguments == {"path": "/a"}
