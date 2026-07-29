"""Persists every Decision (with all its signal scores) to a JSONL file.

Every configured signal's score is written on every decision, even when
it wasn't the one that determined the verdict -- that's the point:
PROJECT.md calls for logging every signal on every decision for
calibration analysis, and this is what makes that data durable once a
real agent is being wrapped (as opposed to the eval harness, which
computes metrics by re-running scenarios in memory and never needed
this).
"""

import json
from pathlib import Path

from threshold.types import Decision, ProposedAction


class DecisionLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: ProposedAction, decision: Decision) -> None:
        record = {
            "action": action.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
