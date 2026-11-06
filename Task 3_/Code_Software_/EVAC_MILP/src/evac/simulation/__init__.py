"""Plan inspection and simulation."""

from __future__ import annotations
from evac.domain import EvacuationPlan, PreparedScenario, SimulationResult
from evac.errors import ValidationError
from evac.simulation.engine import (
    inspect_plan as run_plan_inspection,
    simulate as run_simulation,
    validate_plan,
)
from evac.simulation.inspection import PlanInspection, PlanInspectionIssue


def inspect_plan(
    prepared_scenario: PreparedScenario, plan: EvacuationPlan
) -> PlanInspection:
    """Check plan structure and physical constraints without simulation."""
    if not isinstance(prepared_scenario, PreparedScenario) or not isinstance(
        plan, EvacuationPlan
    ):
        raise ValidationError(
            "inspect_plan requires a PreparedScenario and EvacuationPlan."
        )
    return run_plan_inspection(prepared_scenario, plan)


def simulate(
    prepared_scenario: PreparedScenario, plan: EvacuationPlan
) -> SimulationResult:
    """Simulate a plan using its matching prepared inputs."""
    if not isinstance(prepared_scenario, PreparedScenario) or not isinstance(
        plan, EvacuationPlan
    ):
        raise ValidationError(
            "simulate requires a PreparedScenario and EvacuationPlan."
        )
    return run_simulation(prepared_scenario, plan)


__all__ = [
    "PlanInspection",
    "PlanInspectionIssue",
    "inspect_plan",
    "simulate",
    "validate_plan",
]
