"""Gurobi runtime inspection and MILP session lifecycle."""

from __future__ import annotations
import math
from importlib import metadata, util
from evac.errors import ValidationError
from evac.planning.selections import MILP
from evac.planners.milp.problem import AffineExpression, MilpProblem, VariableType
from evac.planners.milp.results import (
    GurobiProbe,
    NativeSolveOutcome,
    NormalizedTermination,
    WarmStart,
    warm_start_columns,
)

# These names translate application controls; Gurobi owns native validation.
_CONTROL_PARAMETERS = {
    "TimeLimit": "time_limit_seconds",
    "Threads": "threads",
    "MIPGap": "mip_gap",
    "MIPGapAbs": "mip_gap_absolute",
    "Seed": "seed",
}


def probe_gurobi(*, runtime: bool = False) -> GurobiProbe:
    """Inspect installation; explicitly check the license when runtime=True."""
    if util.find_spec("gurobipy") is None:
        return GurobiProbe(
            False, None, runtime, False if runtime else None,
            failure_kind="dependency", reason="The gurobipy package is not installed.",
        )
    version = metadata.version("gurobipy")
    if not runtime:
        return GurobiProbe(True, version, False, None)
    import gurobipy as gp

    environment = None
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
    except gp.GurobiError as exc:
        return GurobiProbe(
            True, version, True, False, failure_kind="license", reason=str(exc),
        )
    finally:
        if environment is not None:
            environment.dispose()
    return GurobiProbe(True, version, True, True)


class GurobiSession:
    def __init__(self, problem: MilpProblem, *, planner: MILP) -> None:
        import gurobipy as gp

        self._gp = gp
        self._problem_contract = problem
        self._planner = planner
        self._closed = False
        self._environment = gp.Env(empty=True)
        try:
            self._environment.setParam("OutputFlag", 0)
            self._environment.setParam("LogToConsole", 0)
            self._environment.start()
            self._model = gp.Model("EVac MILP", env=self._environment)
            self._variables: list[object] = []
            self._apply_parameters()
            self._load_problem()
        except Exception:
            model = getattr(self, "_model", None)
            if model is not None:
                model.dispose()
            self._environment.dispose()
            self._closed = True
            raise

    def _load_problem(self) -> None:
        gp = self._gp
        for block in self._problem_contract.variables:
            native_type = {
                VariableType.CONTINUOUS: gp.GRB.CONTINUOUS,
                VariableType.INTEGER: gp.GRB.INTEGER,
                VariableType.BINARY: gp.GRB.BINARY,
            }[block.variable_type]
            for index, (lower, upper) in enumerate(
                zip(block.lower, block.upper, strict=True)
            ):
                self._variables.append(
                    self._model.addVar(
                        lb=lower,
                        ub=upper,
                        vtype=native_type,
                        name=f"{block.id}[{index}]",
                    )
                )
        self._model.update()
        for (
            row_id,
            expression,
            lower,
            upper,
        ) in self._problem_contract.constraints.rows():
            native = self._expression(expression)
            if lower == upper:
                self._model.addLConstr(native == lower, name=row_id)
            elif math.isfinite(lower) and math.isfinite(upper):
                self._model.addRange(native, lower, upper, name=row_id)
            elif math.isfinite(lower):
                self._model.addLConstr(native >= lower, name=row_id)
            elif math.isfinite(upper):
                self._model.addLConstr(native <= upper, name=row_id)
        self._model.update()

    def _expression(self, expression: AffineExpression) -> object:
        return (
            self._gp.LinExpr(
                list(expression.coefficients),
                [self._variables[column] for column in expression.columns],
            )
            + expression.constant
        )

    def solve_one(
        self,
        objective: AffineExpression,
        *,
        remaining_seconds: float | None,
        start: WarmStart | None,
    ) -> NativeSolveOutcome:
        gp = self._gp
        self._model.setObjective(self._expression(objective), gp.GRB.MINIMIZE)
        if remaining_seconds is not None:
            self._model.setParam("TimeLimit", remaining_seconds)
        if start is not None:
            indices, values = warm_start_columns(start, self._problem_contract)
            for column, value in zip(indices, values, strict=True):
                self._variables[column].Start = value

        self._model.optimize()
        has_incumbent = int(self._model.SolCount) > 0
        values = (
            tuple((float(variable.X) for variable in self._variables))
            if has_incumbent
            else None
        )
        objective_value = objective.evaluate(values) if values is not None else None
        objective_bound = _finite_or_none(getattr(self._model, "ObjBound", None))
        return NativeSolveOutcome(
            termination=_gurobi_termination(
                int(self._model.Status), gp.GRB, has_incumbent=has_incumbent
            ),
            values=values,
            objective_value=objective_value,
            objective_bound=objective_bound,
            provider_status=int(self._model.Status),
            telemetry={
                "runtime_seconds": float(self._model.Runtime),
                "node_count": float(self._model.NodeCount),
                "native_variables": int(self._model.NumVars),
                "native_linear_constraints": int(self._model.NumConstrs),
            },
            terminal_cause=_gurobi_terminal_cause(int(self._model.Status), gp.GRB),
        )

    def _apply_parameters(self) -> None:
        for name, value in self._planner.parameters.items():
            info = self._model.getParamInfo(name)
            if info is not None:
                common = {key.casefold(): field for key, field in _CONTROL_PARAMETERS.items()}
                field = common.get(info[0].replace("_", "").casefold())
                if field is not None:
                    raise ValidationError(f"Set {info[0]} through MILP.{field}.")
            self._model.setParam(name, value)
        for native_name, field in _CONTROL_PARAMETERS.items():
            value = getattr(self._planner, field)
            if value is not None:
                self._model.setParam(native_name, value)

    def add_objective_bound(
        self, expression: AffineExpression, *, upper: float, name: str
    ) -> None:
        self._model.addLConstr(self._expression(expression) <= upper, name=name)

    def close(self) -> None:
        if not self._closed:
            self._model.dispose()
            self._environment.dispose()
            self._closed = True


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _gurobi_termination(
    status: int, grb: object, *, has_incumbent: bool
) -> NormalizedTermination:
    if status == grb.OPTIMAL:
        return NormalizedTermination.OPTIMAL
    if status == grb.INFEASIBLE:
        return NormalizedTermination.INFEASIBLE
    if status == grb.UNBOUNDED:
        return NormalizedTermination.UNBOUNDED
    if status == grb.INF_OR_UNBD:
        return NormalizedTermination.INFEASIBLE_OR_UNBOUNDED
    if status == grb.INTERRUPTED and (not has_incumbent):
        return NormalizedTermination.INTERRUPTED
    return (
        NormalizedTermination.FEASIBLE
        if has_incumbent
        else NormalizedTermination.NO_INCUMBENT
    )


def _gurobi_terminal_cause(status: int, grb: object) -> str | None:
    if status == grb.OPTIMAL:
        return None
    names = {
        grb.INFEASIBLE: "infeasible",
        grb.INF_OR_UNBD: "infeasible_or_unbounded",
        grb.UNBOUNDED: "unbounded",
        grb.TIME_LIMIT: "time_limit",
        grb.NODE_LIMIT: "node_limit",
        grb.ITERATION_LIMIT: "iteration_limit",
        grb.SOLUTION_LIMIT: "solution_limit",
        grb.INTERRUPTED: "interrupted",
        grb.NUMERIC: "numeric",
    }
    for optional_name, cause in (
        ("WORK_LIMIT", "work_limit"),
        ("MEM_LIMIT", "memory_limit"),
    ):
        optional_status = getattr(grb, optional_name, None)
        if optional_status is not None:
            names[optional_status] = cause
    return names.get(status, f"provider_status_{status}")
