"""Anthropic-backed agent: proposes the next tool call given a task and
history, and can resample the SAME decision point N times at
temperature to populate ProposedAction.alternative_actions.

This is the piece that resolves the architectural question raised when
self-consistency was designed (Milestone 2): the threshold library
only ever sees a ProposedAction, never a live hook into an agent's
inference step, because it has no way to actually re-invoke a model.
This agent IS that hook -- alternative_actions still arrives as plain
data (per the original design), but now it's genuinely populated by
resampling here rather than left for hand-authored scenario data.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from threshold.adapters.retry import create_with_temperature_fallback
from threshold.types import ProposedAction, Step


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class _RawMessages(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _RawClient(Protocol):
    messages: _RawMessages


class AnthropicAgent:
    def __init__(
        self,
        raw_client: _RawClient,
        model: str,
        system_prompt: str,
        tools: list[ToolSpec],
        task: str,
    ):
        self._raw_client = raw_client
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._task = task

    def _build_messages(self, trajectory: list[Step]) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": self._task}]
        for i, step in enumerate(trajectory):
            tool_use_id = f"step_{i}"
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tool_use_id, "name": step.tool_name, "input": step.arguments}
                    ],
                }
            )
            result_text = str(step.result) if step.result is not None else ("ok" if step.succeeded else "error")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_text,
                            "is_error": step.succeeded is False,
                        }
                    ],
                }
            )
        return messages

    async def propose_next_action(
        self, trajectory: list[Step] | None = None, temperature: float = 0.0
    ) -> ProposedAction | None:
        trajectory = trajectory or []
        response = await create_with_temperature_fallback(
            self._raw_client.messages,
            model=self._model,
            max_tokens=1024,
            temperature=temperature,
            system=self._system_prompt,
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self._tools
            ],
            messages=self._build_messages(trajectory),
        )

        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_use_blocks:
            return None  # agent considers itself done -- not an error

        block = tool_use_blocks[0]
        text = " ".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        return ProposedAction(
            tool_name=block.name,
            arguments=block.input,
            agent_reasoning=text or "(no explicit reasoning provided)",
            trajectory=trajectory,
        )

    async def propose_with_alternatives(
        self,
        trajectory: list[Step] | None = None,
        n_samples: int = 2,
        sample_temperature: float = 0.8,
    ) -> ProposedAction | None:
        primary = await self.propose_next_action(trajectory, temperature=0.0)
        if primary is None:
            return None

        alternatives = []
        for _ in range(n_samples):
            sample = await self.propose_next_action(trajectory, temperature=sample_temperature)
            if sample is not None:
                alternatives.append(sample)

        return primary.model_copy(update={"alternative_actions": alternatives})
