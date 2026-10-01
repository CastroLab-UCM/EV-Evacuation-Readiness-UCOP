"""Inspect the actual MILP size without starting a solver."""

from dataclasses import dataclass
from evac.scenario.preparation import prepare
from .run_case import RunCase
from evac.planners.milp.adapter import build_model, model_dimensions


@dataclass(frozen=True, slots=True)
class RunCaseInspection:
    variables: int
    linear_constraints: int
    objective_tiers: int
    vehicle_count: int
    candidate_path_count: int


def inspect_run_case(case: RunCase) -> RunCaseInspection:
    prepared = prepare(
        case.scenario,
        objective=case.objective,
        mcs_count=case.mcs_count,
        initial_soc=case.initial_soc,
    )
    _, context = build_model(prepared)
    return RunCaseInspection(
        **model_dimensions(context.problem),
        vehicle_count=len(prepared.vehicles),
        candidate_path_count=len(prepared.candidate_paths),
    )
