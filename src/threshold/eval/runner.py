"""Runs scenarios through a policy and records the outcome of each."""

from dataclasses import dataclass

from threshold.eval.scenario import Scenario
from threshold.policy.policy import Policy
from threshold.types import Decision


@dataclass
class ScenarioResult:
    scenario: Scenario
    decision: Decision

    @property
    def actual_escalate(self) -> bool:
        return self.decision.verdict != "proceed"

    @property
    def correct(self) -> bool:
        return self.actual_escalate == self.scenario.should_escalate

    @property
    def outcome(self) -> str:
        if self.scenario.should_escalate and self.actual_escalate:
            return "true_positive"
        if not self.scenario.should_escalate and not self.actual_escalate:
            return "true_negative"
        if self.actual_escalate and not self.scenario.should_escalate:
            return "false_alarm"
        return "miss"


async def run_scenarios(policy: Policy, scenarios: list[Scenario]) -> list[ScenarioResult]:
    results = []
    for scenario in scenarios:
        decision = await policy.evaluate(scenario.action)
        results.append(ScenarioResult(scenario=scenario, decision=decision))
    return results
