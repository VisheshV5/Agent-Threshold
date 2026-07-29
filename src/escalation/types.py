"""Core data models shared across signals, policy, and eval."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Step(BaseModel):
    """One tool call already taken earlier in the agent's run."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    succeeded: bool | None = None


class ProposedAction(BaseModel):
    """The action an agent is about to take, plus everything needed to judge it."""

    tool_name: str
    arguments: dict[str, Any]
    agent_reasoning: str
    trajectory: list[Step] = Field(default_factory=list)
    context_age_steps: dict[str, int] = Field(default_factory=dict)
    # Pre-sampled candidates for the same decision point, gathered by
    # whoever constructs this ProposedAction (an adapter re-invoking the
    # agent at temperature, or hand-authored scenario data) -- not a live
    # callback, so this stays plain, JSONL-serializable data. Used by the
    # self-consistency signal; empty when resampling wasn't performed.
    alternative_actions: list["ProposedAction"] = Field(default_factory=list)


class SignalResult(BaseModel):
    """Output of a single signal for a single proposed action."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    cost_ms: int = Field(ge=0)


class HumanQuestion(BaseModel):
    """A structured question to surface when escalating, never open-ended."""

    summary: str
    concern: str
    options: list[str] = Field(min_length=2, max_length=4)
    recommended_option: str

    @model_validator(mode="after")
    def recommended_must_be_an_option(self) -> "HumanQuestion":
        if self.recommended_option not in self.options:
            raise ValueError("recommended_option must be one of options")
        return self


class Decision(BaseModel):
    """The policy's final verdict on a proposed action."""

    verdict: Literal["proceed", "ask_human", "abort"]
    aggregate_score: float = Field(ge=0.0, le=1.0)
    signals: list[SignalResult]
    question: HumanQuestion | None = None
    hard_override_triggered: bool = False
