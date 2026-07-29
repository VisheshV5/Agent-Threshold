"""Tests for the Anthropic-backed agent, written before the implementation.

This is the piece that resolves the architectural question raised when
self-consistency was designed (Milestone 2): the escalation library
only ever sees a ProposedAction, never a live hook into an agent's
inference step. This agent IS that hook -- the boundary where a real
model call happens and gets converted into the library's plain-data
interface (ProposedAction.alternative_actions, populated by actually
resampling here, not a live callback threaded through the core library).

Pins the API surface:
- AnthropicAgent(raw_client, model, system_prompt, tools, task)
- await agent.propose_next_action(trajectory=None, temperature=0.0) -> ProposedAction | None
  (None means the agent proposed no tool call -- it considers itself done)
- await agent.propose_with_alternatives(trajectory=None, n_samples=2, sample_temperature=0.8)
  -> ProposedAction | None, with alternative_actions populated from n_samples
  additional calls at a higher temperature
- Prior trajectory steps are replayed as assistant tool_use / user
  tool_result message pairs, so the model sees real history
"""

import asyncio

from escalation.adapters.agent import AnthropicAgent, ToolSpec
from escalation.types import Step


def run(coro):
    return asyncio.run(coro)


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name: str, input: dict):
        self.name = name
        self.input = input


class _FakeMessage:
    def __init__(self, content: list):
        self.content = content


class _FakeMessages:
    def __init__(self, responses: list[_FakeMessage]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeRawClient:
    def __init__(self, responses: list[_FakeMessage]):
        self.messages = _FakeMessages(responses)


TOOLS = [
    ToolSpec(name="read_file", description="Read a file", input_schema={"type": "object"}),
    ToolSpec(name="delete_file", description="Delete a file", input_schema={"type": "object"}),
]


def make_agent(responses: list[_FakeMessage], task: str = "Clean up temp files.") -> tuple[AnthropicAgent, _FakeRawClient]:
    raw = _FakeRawClient(responses)
    agent = AnthropicAgent(
        raw_client=raw, model="claude-sonnet-5", system_prompt="You are a file management agent.",
        tools=TOOLS, task=task,
    )
    return agent, raw


# --- propose_next_action ---

def test_proposes_an_action_from_a_tool_use_block():
    response = _FakeMessage([
        _FakeTextBlock("I should check what's in the temp directory first."),
        _FakeToolUseBlock("read_file", {"path": "/tmp"}),
    ])
    agent, raw = make_agent([response])
    action = run(agent.propose_next_action())

    assert action is not None
    assert action.tool_name == "read_file"
    assert action.arguments == {"path": "/tmp"}
    assert "temp directory" in action.agent_reasoning


def test_falls_back_to_placeholder_reasoning_when_no_text_block_present():
    response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    agent, raw = make_agent([response])
    action = run(agent.propose_next_action())
    assert action.agent_reasoning  # non-empty placeholder, not a crash


def test_returns_none_when_the_agent_proposes_no_tool_call():
    response = _FakeMessage([_FakeTextBlock("I believe the task is already complete.")])
    agent, raw = make_agent([response])
    action = run(agent.propose_next_action())
    assert action is None


def test_sends_system_prompt_tools_and_task_to_the_api():
    response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    agent, raw = make_agent([response], task="Clean up temp files.")
    run(agent.propose_next_action())

    call = raw.messages.calls[0]
    assert call["system"] == "You are a file management agent."
    assert call["model"] == "claude-sonnet-5"
    assert {t["name"] for t in call["tools"]} == {"read_file", "delete_file"}
    assert call["messages"][0] == {"role": "user", "content": "Clean up temp files."}


def test_prior_trajectory_is_replayed_as_tool_use_and_tool_result_pairs():
    response = _FakeMessage([_FakeToolUseBlock("delete_file", {"path": "/tmp/old.txt"})])
    agent, raw = make_agent([response])
    trajectory = [Step(tool_name="read_file", arguments={"path": "/tmp"}, result=["old.txt"], succeeded=True)]

    action = run(agent.propose_next_action(trajectory=trajectory))

    messages = raw.messages.calls[0]["messages"]
    # user task, then assistant tool_use, then user tool_result
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["name"] == "read_file"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert action.trajectory == trajectory


def test_failed_prior_step_is_marked_as_an_error_tool_result():
    response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    agent, raw = make_agent([response])
    trajectory = [Step(tool_name="read_file", arguments={}, result="permission denied", succeeded=False)]
    run(agent.propose_next_action(trajectory=trajectory))

    tool_result = raw.messages.calls[0]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True


def test_uses_the_given_temperature():
    response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    agent, raw = make_agent([response])
    run(agent.propose_next_action(temperature=0.9))
    assert raw.messages.calls[0]["temperature"] == 0.9


# --- propose_with_alternatives (the self-consistency resampling hook) ---

def test_propose_with_alternatives_populates_alternative_actions():
    primary_response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    alt1_response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    alt2_response = _FakeMessage([_FakeToolUseBlock("delete_file", {"path": "/tmp/x"})])
    agent, raw = make_agent([primary_response, alt1_response, alt2_response])

    action = run(agent.propose_with_alternatives(n_samples=2, sample_temperature=0.8))

    assert action is not None
    assert len(action.alternative_actions) == 2
    assert action.alternative_actions[1].tool_name == "delete_file"


def test_propose_with_alternatives_uses_temperature_zero_for_primary_and_higher_for_samples():
    primary_response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    alt_response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    agent, raw = make_agent([primary_response, alt_response])

    run(agent.propose_with_alternatives(n_samples=1, sample_temperature=0.8))

    assert raw.messages.calls[0]["temperature"] == 0.0  # primary: deterministic
    assert raw.messages.calls[1]["temperature"] == 0.8  # resample: exploratory


def test_propose_with_alternatives_returns_none_if_primary_has_no_tool_call():
    response = _FakeMessage([_FakeTextBlock("Nothing left to do.")])
    agent, raw = make_agent([response])
    action = run(agent.propose_with_alternatives(n_samples=2))
    assert action is None


def test_propose_with_alternatives_drops_a_no_tool_call_sample():
    primary_response = _FakeMessage([_FakeToolUseBlock("read_file", {"path": "/tmp"})])
    alt1_response = _FakeMessage([_FakeTextBlock("Actually I think we're done.")])  # no tool call
    alt2_response = _FakeMessage([_FakeToolUseBlock("delete_file", {"path": "/tmp/x"})])
    agent, raw = make_agent([primary_response, alt1_response, alt2_response])

    action = run(agent.propose_with_alternatives(n_samples=2))

    assert len(action.alternative_actions) == 1  # the no-tool-call sample was dropped
    assert action.alternative_actions[0].tool_name == "delete_file"
