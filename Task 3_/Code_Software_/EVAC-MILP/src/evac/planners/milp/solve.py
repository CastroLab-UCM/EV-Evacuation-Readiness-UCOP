"""Sequential optimization of MILP objective priorities."""

from __future__ import annotations
import time
from dataclasses import replace
from evac.planning.selections import MILP
from evac.planners.milp.gurobi import GurobiSession, probe_gurobi
from evac.planners.milp.lowering import lower_exact_maxima, objective_degradation
from evac.planners.milp.problem import MilpProblem
from evac.planners.milp.results import (
    SolverDiagnostics,
    Incumbent,
    NormalizedTermination,
    ObjectiveCertificate,
    SolveResult,
    WarmStart,
    normalized_mip_gap,
    warm_start_columns,
)


def solve_problem(
    problem: MilpProblem,
    planner: MILP,
) -> SolveResult:
    lowered = lower_exact_maxima(problem)
    deadline = (
        None
        if planner.time_limit_seconds is None
        else time.monotonic() + planner.time_limit_seconds
    )
    certificates: list[ObjectiveCertificate] = []
    accepted_values: tuple[float, ...] | None = None
    final_status: str | int | None = None
    final_cause: str | None = None
    tiers: list[dict[str, object]] = []
    final_telemetry: dict[str, object] = {"tiers": tiers}
    provider_version = probe_gurobi().dependency_version
    session = GurobiSession(lowered, planner=planner)
    next_start = None
    try:
        for tier_index, tier in enumerate(lowered.objective_tiers):
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if remaining == 0.0:
                termination = (
                    NormalizedTermination.FEASIBLE
                    if accepted_values is not None
                    else NormalizedTermination.NO_INCUMBENT
                )
                final_cause = "time_limit"
                return _result(
                    problem,
                    termination,
                    accepted_values,
                    certificates,
                    final_status,
                    final_telemetry,
                    provider_version,
                    final_cause,
                )
            start_for_tier = next_start
            outcome = session.solve_one(
                tier.expression,
                remaining_seconds=remaining,
                start=start_for_tier,
            )
            final_status = outcome.provider_status
            final_cause = outcome.terminal_cause
            start_columns = (
                0
                if start_for_tier is None
                else len(warm_start_columns(start_for_tier, lowered)[0])
            )
            tiers.append(
                {
                    "tier_id": tier.id,
                    "termination": outcome.termination.value,
                    "warm_start_columns": start_columns,
                    "warm_start_accepted": "unknown",
                    **outcome.telemetry,
                }
            )
            final_telemetry = {**dict(outcome.telemetry), "tiers": list(tiers)}
            if outcome.values is not None:
                lowered.validate_values(outcome.values)
                accepted_values = outcome.values
                next_start = WarmStart(
                    values={
                        block.id: outcome.values[
                            block.offset : block.offset + block.size
                        ]
                        for block in lowered.variables
                    }
                )
            certificate = ObjectiveCertificate(
                tier_id=tier.id,
                incumbent_value=outcome.objective_value,
                objective_bound=outcome.objective_bound,
                normalized_gap=normalized_mip_gap(
                    outcome.objective_value, outcome.objective_bound
                )
                if lowered.is_mip
                else None,
                termination=outcome.termination,
            )
            certificates.append(certificate)
            if outcome.termination is not NormalizedTermination.OPTIMAL:
                return _result(
                    problem,
                    outcome.termination,
                    accepted_values,
                    certificates,
                    final_status,
                    final_telemetry,
                    provider_version,
                    final_cause,
                )
            if outcome.objective_value is None or accepted_values is None:
                raise RuntimeError(
                    "An optimal MILP objective tier returned no incumbent certificate."
                )
            if tier_index + 1 < len(lowered.objective_tiers):
                allowed = objective_degradation(
                    outcome.objective_value,
                    absolute=tier.absolute_degradation,
                    relative=tier.relative_degradation,
                )
                session.add_objective_bound(
                    tier.expression,
                    upper=outcome.objective_value + allowed,
                    name=f"__objective_tier_bound__/{tier.id}",
                )
        return _result(
            problem,
            NormalizedTermination.OPTIMAL,
            accepted_values,
            certificates,
            final_status,
            final_telemetry,
            provider_version,
            final_cause,
        )
    finally:
        session.close()


def _result(
    problem: MilpProblem,
    termination: NormalizedTermination,
    values: tuple[float, ...] | None,
    certificates: list[ObjectiveCertificate],
    provider_status: str | int | None,
    telemetry: dict[str, object],
    provider_version: str | None,
    terminal_cause: str | None,
) -> SolveResult:
    if terminal_cause is None and termination is not NormalizedTermination.OPTIMAL:
        terminal_cause = termination.value
    if termination in {NormalizedTermination.OPTIMAL, NormalizedTermination.FEASIBLE}:
        if values is None:
            termination = NormalizedTermination.NO_INCUMBENT
    elif values is not None and termination in {
        NormalizedTermination.INTERRUPTED,
        NormalizedTermination.NO_INCUMBENT,
    }:
        termination = NormalizedTermination.FEASIBLE
    public_values = None if values is None else values[: problem.variable_count]
    if public_values is not None:
        expressions = {tier.id: tier.expression for tier in problem.objective_tiers}
        certificates = [
            replace(
                cert,
                incumbent_value=expressions[cert.tier_id].evaluate(public_values),
                normalized_gap=normalized_mip_gap(
                    expressions[cert.tier_id].evaluate(public_values),
                    cert.objective_bound,
                ),
            )
            for cert in certificates
        ]
    incumbent = (
        None
        if public_values is None
        else Incumbent(raw_values=public_values, variable_blocks=problem.variables)
    )
    return SolveResult(
        termination=termination,
        incumbent=incumbent,
        certificates=tuple(certificates),
        diagnostics=SolverDiagnostics(
            provider_status=provider_status,
            provider_version=provider_version,
            telemetry=telemetry,
        ),
        terminal_cause=terminal_cause,
    )
