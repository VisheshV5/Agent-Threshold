"""Decision cache: once a human resolves a situation, don't ask again.

Keyed by an exact structural match of (tool_name, arguments) -- not
agent_reasoning, matching self-consistency's stance that reasoning
doesn't define the situation, and not a fuzzy/semantic match. A real
embedding-based match risks a false-positive cache hit auto-resolving a
similar-but-different situation without human review; this is the
conservative, exact-match M4 stand-in, same pattern as every other
"real semantic operation" in this project (self-consistency's argument
overlap, novelty's sequence comparison) having a heuristic placeholder
until a real embedding model is wired in.

Hard-override categories (irreversible-write, external-effect) must
never be cached, no exceptions -- enforced by the caller (Policy),
since only Policy knows the reversibility category; this module has no
opinion on that and will happily cache anything it's given.
"""

import json
from pathlib import Path
from typing import Literal

from threshold.types import ProposedAction

Resolution = Literal["proceed", "abort"]


def _situation_key(action: ProposedAction) -> str:
    return json.dumps(
        {"tool_name": action.tool_name, "arguments": action.arguments},
        sort_keys=True,
        default=str,
    )


class DecisionCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._resolutions: dict[str, Resolution] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self._resolutions[record["key"]] = record["resolution"]

    def get(self, action: ProposedAction) -> Resolution | None:
        return self._resolutions.get(_situation_key(action))

    def record(self, action: ProposedAction, resolution: Resolution) -> None:
        key = _situation_key(action)
        self._resolutions[key] = resolution
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "resolution": resolution}) + "\n")
