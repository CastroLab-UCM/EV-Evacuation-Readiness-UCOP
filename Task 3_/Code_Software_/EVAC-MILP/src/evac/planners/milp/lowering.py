"""Shared exact transformations for the solver-neutral MILP problem."""

from __future__ import annotations
import math
from dataclasses import replace
from evac.errors import ValidationError
from evac.planners.milp.problem import (
    AffineExpression,
    MilpProblem,
    SparseConstraintMatrix,
    VariableBlock,
    VariableType,
    expression_bounds,
)


def lower_exact_maxima(problem: MilpProblem) -> MilpProblem:
    """Replace every exact maximum with a bounded exact MILP linearization."""
    if not problem.exact_maxima:
        return problem
    variables = list(problem.variables)
    rows = list(problem.constraints.rows())
    lower_bounds, upper_bounds = problem.bounds()
    next_offset = problem.variable_count
    for maximum in problem.exact_maxima:
        result_upper = upper_bounds[maximum.result_column]
        if not math.isfinite(result_upper):
            raise ValidationError(
                f"Exact maximum {maximum.id!r} requires a finite result upper bound."
            )
        selector = VariableBlock(
            id=f"__exact_max_selector__/{maximum.id}",
            shape=(len(maximum.operands),),
            variable_type=VariableType.BINARY,
            lower=(0.0,) * len(maximum.operands),
            upper=(1.0,) * len(maximum.operands),
            offset=next_offset,
        )
        next_offset += selector.size
        variables.append(selector)
        rows.append(
            (
                f"{maximum.id}/selector",
                AffineExpression.from_terms(
                    ((column, 1.0) for column in selector.columns)
                ),
                1.0,
                1.0,
            )
        )
        for operand_index, operand in enumerate(maximum.operands):
            operand_lower, operand_upper = expression_bounds(problem, operand)
            if not math.isfinite(operand_lower) or not math.isfinite(operand_upper):
                raise ValidationError(
                    f"Exact maximum {maximum.id!r} operand {operand_index} requires finite bounds."
                )
            if result_upper < operand_upper:
                raise ValidationError(
                    f"Exact maximum {maximum.id!r} result upper bound {result_upper} is below operand {operand_index} upper bound {operand_upper}."
                )
            rows.append(
                (
                    f"{maximum.id}/lower/{operand_index}",
                    AffineExpression.from_terms(
                        [(maximum.result_column, 1.0)]
                        + [
                            (column, -coefficient)
                            for column, coefficient in zip(
                                operand.columns, operand.coefficients, strict=True
                            )
                        ]
                    ),
                    operand.constant,
                    math.inf,
                )
            )
            big_m = result_upper - operand_lower
            rows.append(
                (
                    f"{maximum.id}/upper/{operand_index}",
                    AffineExpression.from_terms(
                        [
                            (maximum.result_column, 1.0),
                            (selector.offset + operand_index, big_m),
                        ]
                        + [
                            (column, -coefficient)
                            for column, coefficient in zip(
                                operand.columns, operand.coefficients, strict=True
                            )
                        ]
                    ),
                    -math.inf,
                    big_m + operand.constant,
                )
            )
    return replace(
        problem,
        variables=tuple(variables),
        constraints=SparseConstraintMatrix.from_rows(rows),
        exact_maxima=(),
    )


def objective_degradation(value: float, *, absolute: float, relative: float) -> float:
    """EVac's explicit objective degradation rule for sequential minimization."""
    if not all((math.isfinite(item) for item in (value, absolute, relative))):
        raise ValidationError(
            "Objective value and degradation tolerances must be finite."
        )
    if absolute < 0.0 or relative < 0.0:
        raise ValidationError("Objective degradation tolerances must be nonnegative.")
    return max(absolute, abs(value) * relative)
