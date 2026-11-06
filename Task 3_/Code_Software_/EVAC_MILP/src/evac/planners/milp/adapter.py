"""MILP compilation, solving, and plan conversion."""

from dataclasses import asdict
from time import perf_counter
from evac.domain import PlannerResult, PlannerTermination, PreparedScenario
from evac.errors import ValidationError
from evac.planning.identity import model_identity
from evac.planning.selections import MILP
from evac.planners.milp.compiler import compile_prepared_scenario
from evac.planners.milp.conversion import canonical_plan_from_source
from evac.planners.milp.formulation import build_formulation
from evac.planners.milp.lowering import lower_exact_maxima
from evac.planners.milp.results import NormalizedTermination
from evac.planners.milp.solve import solve_problem


def build_model(prepared):
    projection = compile_prepared_scenario(prepared)
    return (projection, build_formulation(projection.pack))


def model_dimensions(problem):
    lowered = lower_exact_maxima(problem)
    return {
        "variables": lowered.variable_count,
        "linear_constraints": len(lowered.constraints.row_ids),
        "objective_tiers": len(lowered.objective_tiers),
    }


def plan(prepared_scenario: PreparedScenario, planner: MILP) -> PlannerResult:
    prepared = prepared_scenario
    selection = planner
    if not isinstance(prepared, PreparedScenario) or not isinstance(selection, MILP):
        raise ValidationError("plan requires a PreparedScenario and MILP selection.")
    start = perf_counter()
    projection, context = build_model(prepared)
    build_seconds = perf_counter() - start
    start = perf_counter()
    solved = solve_problem(context.problem, selection)
    solve_seconds = perf_counter() - start
    plan = None
    raw_values = None
    if solved.incumbent is not None:
        raw_values = tuple(solved.incumbent.raw_values)
        document, _ = context.materialize_source_solution(raw_values)
        plan = canonical_plan_from_source(projection, document)
    if solved.termination is NormalizedTermination.OPTIMAL:
        termination = PlannerTermination.OPTIMAL
    elif plan is not None:
        termination = PlannerTermination.FEASIBLE
    elif solved.termination is NormalizedTermination.INFEASIBLE:
        termination = PlannerTermination.INFEASIBLE
    else:
        termination = PlannerTermination.NO_INCUMBENT
    certificate = solved.certificates[0] if solved.certificates else None
    divisor = (
        len(prepared.vehicles)
        if prepared.evaluation.objectives[0].norm == "mean"
        else 1
    )
    value = (
        None
        if certificate is None or certificate.incumbent_value is None
        else certificate.incumbent_value / divisor
    )
    bound = (
        None
        if certificate is None or certificate.objective_bound is None
        else certificate.objective_bound / divisor
    )
    settings = {
        "time_limit_seconds": selection.time_limit_seconds,
        "threads": selection.threads,
        "mip_gap": selection.mip_gap,
        "mip_gap_absolute": selection.mip_gap_absolute,
        "seed": selection.seed,
        "parameters": dict(selection.parameters),
    }
    metadata = {
        "backend": "gurobi",
        "provider_status": solved.diagnostics.provider_status,
        "provider_version": solved.diagnostics.provider_version,
        "model_size": model_dimensions(context.problem),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "raw_variable_values": raw_values,
        "variable_blocks": [
            {"id": b.id, "offset": b.offset, "shape": b.shape}
            for b in context.problem.variables
        ],
        "certificates": [asdict(c) for c in solved.certificates],
        "solver_telemetry": dict(solved.diagnostics.telemetry),
    }
    return PlannerResult(
        "milp",
        model_identity(prepared, "milp", settings),
        termination,
        plan,
        bound,
        None if certificate is None else certificate.normalized_gap,
        solved.terminal_cause,
        metadata,
        formulation_objective=value,
    )
