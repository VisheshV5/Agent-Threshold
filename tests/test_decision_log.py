"""Tests for the JSONL decision logger, written before the implementation.

Pins the API surface:
- DecisionLogger(path).log(action, decision) appends one JSON line
- Policy(..., logger=None) never touches disk (default, existing behavior)
- Policy(..., logger=DecisionLogger(...)) logs exactly once per evaluate()
  call, and the logged record contains EVERY configured signal's score --
  not just whichever one drove the verdict -- per the spec's requirement
  to log every signal on every decision even when it didn't change the outcome.
"""

import asyncio
import json

from threshold.decision_log import DecisionLogger
from threshold.policy.policy import Policy
from threshold.signals.reversibility import ReversibilitySignal
from threshold.types import ProposedAction


def make_action(tool_name: str, arguments: dict | None = None) -> ProposedAction:
    return ProposedAction(tool_name=tool_name, arguments=arguments or {}, agent_reasoning="test")


def run(coro):
    return asyncio.run(coro)


def test_log_appends_one_json_line_per_call(tmp_path):
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.5)

    action1 = make_action("read_file", {"path": "/a"})
    action2 = make_action("write_file", {"path": "/b"})
    decision1 = run(policy.evaluate(action1))
    decision2 = run(policy.evaluate(action2))

    logger.log(action1, decision1)
    logger.log(action2, decision2)

    lines = (tmp_path / "decisions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    record1 = json.loads(lines[0])
    record2 = json.loads(lines[1])
    assert record1["action"]["tool_name"] == "read_file"
    assert record2["action"]["tool_name"] == "write_file"


def test_log_creates_parent_directories(tmp_path):
    logger = DecisionLogger(tmp_path / "nested" / "dir" / "decisions.jsonl")
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.5)
    action = make_action("read_file")
    decision = run(policy.evaluate(action))
    logger.log(action, decision)
    assert (tmp_path / "nested" / "dir" / "decisions.jsonl").exists()


def test_logged_record_round_trips_decision_fields(tmp_path):
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.5)
    action = make_action("delete_file", {"path": "/x"})
    decision = run(policy.evaluate(action))
    logger.log(action, decision)

    record = json.loads((tmp_path / "decisions.jsonl").read_text().strip())
    assert record["decision"]["verdict"] == "ask_human"
    assert record["decision"]["hard_override_triggered"] is True
    assert record["decision"]["aggregate_score"] == decision.aggregate_score


# --- Policy integration ---

def test_policy_without_logger_does_not_touch_disk(tmp_path):
    policy = Policy(reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.5)
    run(policy.evaluate(make_action("read_file")))
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere


def test_policy_with_logger_logs_every_evaluate_call(tmp_path):
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    policy = Policy(
        reversibility=ReversibilitySignal(), weights={"reversibility": 1.0}, threshold=0.5, logger=logger
    )
    run(policy.evaluate(make_action("read_file")))
    run(policy.evaluate(make_action("write_file")))

    lines = (tmp_path / "decisions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_policy_logs_every_signal_even_when_hard_override_decides_the_verdict(tmp_path):
    from threshold.signals.blast_radius import BlastRadiusSignal
    from threshold.signals.self_consistency import SelfConsistencySignal

    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    policy = Policy(
        reversibility=ReversibilitySignal(),
        weights={"reversibility": 0.4, "blast_radius": 0.2, "self_consistency": 0.4},
        threshold=0.9,  # deliberately high, so only the hard override forces ask_human
        extra_signals=[BlastRadiusSignal(), SelfConsistencySignal()],
        logger=logger,
    )
    run(policy.evaluate(make_action("delete_file")))  # hard override, not threshold, decides this

    record = json.loads((tmp_path / "decisions.jsonl").read_text().strip())
    logged_signal_names = {s["name"] for s in record["decision"]["signals"]}
    assert logged_signal_names == {"reversibility", "blast_radius", "self_consistency"}
