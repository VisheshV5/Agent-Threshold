"""Tests for the thrash signal, written before the implementation.

Pins the API surface:
- ThrashSignal() with no constructor args -- pure function of
  action.trajectory + action.tool_name, no injectable fallback needed
- Counts CONSECUTIVE trailing failures of the SAME tool_name immediately
  before the proposed action -- walking backward from the end of the
  trajectory, stopping at the first step that doesn't match (different
  tool, succeeded=True, or succeeded=None)
- succeeded=None (unknown outcome) breaks the streak, doesn't count as
  a confirmed failure
"""

import asyncio

import pytest

from threshold.signals.base import Signal
from threshold.signals.thrash import ThrashSignal
from threshold.types import ProposedAction, Step


def make_action(tool_name: str, trajectory: list[Step] | None = None) -> ProposedAction:
    return ProposedAction(
        tool_name=tool_name,
        arguments={},
        agent_reasoning="test",
        trajectory=trajectory or [],
    )


def failed(tool_name: str) -> Step:
    return Step(tool_name=tool_name, arguments={}, succeeded=False)


def succeeded(tool_name: str) -> Step:
    return Step(tool_name=tool_name, arguments={}, succeeded=True)


def unknown(tool_name: str) -> Step:
    return Step(tool_name=tool_name, arguments={}, succeeded=None)


def run(coro):
    return asyncio.run(coro)


def test_no_trajectory_is_not_thrashing():
    signal = ThrashSignal()
    result = run(signal.score(make_action("read_file")))
    assert result.score == pytest.approx(0.0)


def test_trailing_failure_of_a_different_tool_does_not_count():
    signal = ThrashSignal()
    action = make_action("read_file", trajectory=[failed("write_file")])
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.0)


def test_one_consecutive_trailing_failure():
    signal = ThrashSignal()
    action = make_action("read_file", trajectory=[failed("read_file")])
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.3)


def test_two_consecutive_trailing_failures():
    signal = ThrashSignal()
    action = make_action("read_file", trajectory=[failed("read_file"), failed("read_file")])
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.6)


def test_three_or_more_consecutive_trailing_failures():
    signal = ThrashSignal()
    action = make_action(
        "read_file", trajectory=[failed("read_file"), failed("read_file"), failed("read_file")]
    )
    result = run(signal.score(action))
    assert result.score == pytest.approx(1.0)


def test_a_different_tool_in_between_breaks_the_streak():
    signal = ThrashSignal()
    action = make_action(
        "read_file",
        trajectory=[failed("read_file"), succeeded("write_file"), failed("read_file")],
    )
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.3)  # only the last failure counts, streak was broken


def test_unknown_outcome_breaks_the_streak():
    signal = ThrashSignal()
    action = make_action("read_file", trajectory=[failed("read_file"), unknown("read_file")])
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.0)  # most recent same-tool step has unknown outcome


def test_a_recent_success_means_not_currently_thrashing():
    signal = ThrashSignal()
    action = make_action(
        "read_file",
        trajectory=[failed("read_file"), failed("read_file"), succeeded("read_file")],
    )
    result = run(signal.score(action))
    assert result.score == pytest.approx(0.0)  # most recent attempt succeeded


def test_signal_is_a_signal_with_expected_name_and_cost():
    signal = ThrashSignal()
    result = run(signal.score(make_action("read_file", trajectory=[failed("read_file")])))
    assert isinstance(signal, Signal)
    assert result.name == "thrash"
    assert result.cost_ms >= 0
