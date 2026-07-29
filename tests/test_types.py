import pytest
from pydantic import ValidationError

from escalation.types import HumanQuestion


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
