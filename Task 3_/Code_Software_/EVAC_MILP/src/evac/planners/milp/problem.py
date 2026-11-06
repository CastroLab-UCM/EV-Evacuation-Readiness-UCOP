"""Solver-independent variables, constraints, and objectives."""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping
from evac.errors import InvariantError, ValidationError


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValidationError(
                f"Variable-block shape extents must be nonnegative; got {shape!r}."
            )
        size *= extent
    return size


def _finite_or_infinite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if math.isnan(result):
        raise ValidationError(f"{label} must not be NaN.")
    return result


def _finite(value: object, *, label: str) -> float:
    result = _finite_or_infinite(value, label=label)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite; got {result!r}.")
    return result


@dataclass(frozen=True, slots=True)
class VariableBlock:
    id: str
    shape: tuple[int, ...]
    variable_type: VariableType
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("VariableBlock.id must be non-empty.")
        if not isinstance(self.variable_type, VariableType):
            raise ValidationError("VariableBlock.variable_type must be typed.")
        if (
            isinstance(self.offset, bool)
            or not isinstance(self.offset, int)
            or self.offset < 0
        ):
            raise ValidationError("VariableBlock.offset must be a nonnegative integer.")
        size = _shape_size(self.shape)
        if len(self.lower) != size or len(self.upper) != size:
            raise ValidationError(
                f"VariableBlock {self.id!r} bounds must contain {size} values."
            )
        lower = tuple(
            (
                _finite_or_infinite(
                    value, label=f"VariableBlock {self.id!r} lower bound"
                )
                for value in self.lower
            )
        )
        upper = tuple(
            (
                _finite_or_infinite(
                    value, label=f"VariableBlock {self.id!r} upper bound"
                )
                for value in self.upper
            )
        )
        for index, (lb, ub) in enumerate(zip(lower, upper, strict=True)):
            if lb > ub:
                raise ValidationError(
                    f"VariableBlock {self.id!r} bound {index} has lower {lb} above upper {ub}."
                )
            if self.variable_type is VariableType.BINARY and (lb < 0.0 or ub > 1.0):
                raise ValidationError(
                    f"Binary VariableBlock {self.id!r} bound {index} must stay within [0, 1]."
                )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        return _shape_size(self.shape)

    @property
    def columns(self) -> range:
        return range(self.offset, self.offset + self.size)


@dataclass(frozen=True, slots=True)
class AffineExpression:
    columns: tuple[int, ...] = ()
    coefficients: tuple[float, ...] = ()
    constant: float = 0.0

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.coefficients):
            raise ValidationError(
                "AffineExpression columns and coefficients must have equal length."
            )
        previous = -1
        normalized: list[float] = []
        for column, coefficient in zip(self.columns, self.coefficients, strict=True):
            if isinstance(column, bool) or not isinstance(column, int) or column < 0:
                raise ValidationError(
                    "AffineExpression columns must be nonnegative integers."
                )
            if column <= previous:
                raise ValidationError(
                    "AffineExpression columns must be strictly increasing and duplicate-free."
                )
            previous = column
            normalized.append(
                _finite(coefficient, label="AffineExpression coefficient")
            )
        object.__setattr__(self, "coefficients", tuple(normalized))
        object.__setattr__(
            self, "constant", _finite(self.constant, label="AffineExpression constant")
        )

    @classmethod
    def from_terms(
        cls, terms: Iterable[tuple[int, float]], *, constant: float = 0.0
    ) -> AffineExpression:
        combined: dict[int, float] = {}
        for column, coefficient in terms:
            combined[column] = combined.get(column, 0.0) + float(coefficient)
        ordered = sorted(
            (
                (column, coefficient)
                for column, coefficient in combined.items()
                if coefficient != 0.0
            )
        )
        return cls(
            columns=tuple((column for column, _ in ordered)),
            coefficients=tuple((coefficient for _, coefficient in ordered)),
            constant=constant,
        )

    def evaluate(self, values: tuple[float, ...]) -> float:
        try:
            return self.constant + sum(
                (
                    coefficient * values[column]
                    for column, coefficient in zip(
                        self.columns, self.coefficients, strict=True
                    )
                )
            )
        except IndexError as exc:
            raise ValidationError(
                "AffineExpression references a missing solution column."
            ) from exc


@dataclass(frozen=True, slots=True)
class SparseConstraintMatrix:
    """Canonical CSR matrix with ranged row bounds."""

    row_ids: tuple[str, ...]
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        row_count = len(self.row_ids)
        if len(set(self.row_ids)) != row_count or any(
            (not item for item in self.row_ids)
        ):
            raise ValidationError(
                "SparseConstraintMatrix row_ids must be unique and non-empty."
            )
        if len(self.indptr) != row_count + 1 or not self.indptr or self.indptr[0] != 0:
            raise ValidationError(
                "SparseConstraintMatrix.indptr must describe every row from zero."
            )
        if self.indptr[-1] != len(self.indices) or len(self.indices) != len(
            self.coefficients
        ):
            raise ValidationError(
                "SparseConstraintMatrix CSR arrays have inconsistent lengths."
            )
        if len(self.lower) != row_count or len(self.upper) != row_count:
            raise ValidationError(
                "SparseConstraintMatrix row bounds have inconsistent lengths."
            )
        previous_pointer = 0
        for row_index, pointer in enumerate(self.indptr[1:]):
            if pointer < previous_pointer:
                raise ValidationError(
                    "SparseConstraintMatrix.indptr must be nondecreasing."
                )
            row_columns = self.indices[previous_pointer:pointer]
            previous_column = -1
            for column in row_columns:
                if (
                    isinstance(column, bool)
                    or not isinstance(column, int)
                    or column < 0
                ):
                    raise ValidationError(
                        "SparseConstraintMatrix column indices must be nonnegative."
                    )
                if column <= previous_column:
                    raise ValidationError(
                        f"SparseConstraintMatrix row {row_index} columns must be sorted and unique."
                    )
                previous_column = column
            previous_pointer = pointer
        coefficients = tuple(
            (
                _finite(value, label="SparseConstraintMatrix coefficient")
                for value in self.coefficients
            )
        )
        lower = tuple(
            (
                _finite_or_infinite(value, label="SparseConstraintMatrix lower bound")
                for value in self.lower
            )
        )
        upper = tuple(
            (
                _finite_or_infinite(value, label="SparseConstraintMatrix upper bound")
                for value in self.upper
            )
        )
        for row_id, lb, ub in zip(self.row_ids, lower, upper, strict=True):
            if lb > ub:
                raise ValidationError(
                    f"SparseConstraintMatrix row {row_id!r} has lower {lb} above upper {ub}."
                )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def empty(cls) -> SparseConstraintMatrix:
        return cls(
            row_ids=(), indptr=(0,), indices=(), coefficients=(), lower=(), upper=()
        )

    @classmethod
    def from_rows(
        cls, rows: Iterable[tuple[str, AffineExpression, float, float]]
    ) -> SparseConstraintMatrix:
        row_ids: list[str] = []
        indptr = [0]
        indices: list[int] = []
        coefficients: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for row_id, expression, row_lower, row_upper in rows:
            row_ids.append(row_id)
            indices.extend(expression.columns)
            coefficients.extend(expression.coefficients)
            lower.append(float(row_lower) - expression.constant)
            upper.append(float(row_upper) - expression.constant)
            indptr.append(len(indices))
        return cls(
            row_ids=tuple(row_ids),
            indptr=tuple(indptr),
            indices=tuple(indices),
            coefficients=tuple(coefficients),
            lower=tuple(lower),
            upper=tuple(upper),
        )

    def rows(self) -> Iterable[tuple[str, AffineExpression, float, float]]:
        for row, row_id in enumerate(self.row_ids):
            start = self.indptr[row]
            end = self.indptr[row + 1]
            yield (
                row_id,
                AffineExpression(
                    columns=self.indices[start:end],
                    coefficients=self.coefficients[start:end],
                ),
                self.lower[row],
                self.upper[row],
            )


@dataclass(frozen=True, slots=True)
class ExactMaximum:
    id: str
    result_column: int
    operands: tuple[AffineExpression, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("ExactMaximum.id must be non-empty.")
        if isinstance(self.result_column, bool) or not isinstance(
            self.result_column, int
        ):
            raise ValidationError("ExactMaximum.result_column must be an integer.")
        if self.result_column < 0:
            raise ValidationError("ExactMaximum.result_column must be nonnegative.")
        if not self.operands:
            raise ValidationError("ExactMaximum requires at least one operand.")


@dataclass(frozen=True, slots=True)
class WeightedObjective:
    id: str
    expression: AffineExpression
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("WeightedObjective.id must be non-empty.")
        object.__setattr__(
            self, "weight", _finite(self.weight, label="objective weight")
        )
        if self.weight == 0.0:
            raise ValidationError("WeightedObjective.weight must be nonzero.")


@dataclass(frozen=True, slots=True)
class ObjectiveTier:
    id: str
    priority: int
    objectives: tuple[WeightedObjective, ...]
    absolute_degradation: float = 0.0
    relative_degradation: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("ObjectiveTier.id must be non-empty.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValidationError("ObjectiveTier.priority must be an integer.")
        if not self.objectives:
            raise ValidationError(
                "ObjectiveTier requires at least one weighted objective."
            )
        if len({objective.id for objective in self.objectives}) != len(self.objectives):
            raise ValidationError("ObjectiveTier objective identifiers must be unique.")
        absolute = _finite(
            self.absolute_degradation, label="absolute objective degradation"
        )
        relative = _finite(
            self.relative_degradation, label="relative objective degradation"
        )
        if absolute < 0.0 or relative < 0.0:
            raise ValidationError(
                "Objective degradation tolerances must be nonnegative."
            )
        object.__setattr__(self, "absolute_degradation", absolute)
        object.__setattr__(self, "relative_degradation", relative)

    @property
    def expression(self) -> AffineExpression:
        terms: list[tuple[int, float]] = []
        constant = 0.0
        for objective in self.objectives:
            terms.extend(
                (
                    (column, objective.weight * coefficient)
                    for column, coefficient in zip(
                        objective.expression.columns,
                        objective.expression.coefficients,
                        strict=True,
                    )
                )
            )
            constant += objective.weight * objective.expression.constant
        return AffineExpression.from_terms(terms, constant=constant)


@dataclass(frozen=True, slots=True)
class PlanBinding:
    field: str
    block_id: str

    def __post_init__(self) -> None:
        if not self.field or not self.block_id:
            raise ValidationError("PlanBinding field and block_id must be non-empty.")


@dataclass(frozen=True, slots=True)
class MilpProblem:
    identity: str
    variables: tuple[VariableBlock, ...]
    constraints: SparseConstraintMatrix
    exact_maxima: tuple[ExactMaximum, ...]
    objective_tiers: tuple[ObjectiveTier, ...]
    plan_bindings: tuple[PlanBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValidationError("MilpProblem.identity must be non-empty.")
        block_ids = [block.id for block in self.variables]
        if len(set(block_ids)) != len(block_ids):
            raise ValidationError(
                "MilpProblem variable block identifiers must be unique."
            )
        expected_offset = 0
        for block in self.variables:
            if block.offset != expected_offset:
                raise ValidationError(
                    f"VariableBlock {block.id!r} starts at {block.offset}; expected {expected_offset}."
                )
            expected_offset += block.size
        for column in self.constraints.indices:
            if column >= expected_offset:
                raise ValidationError(
                    "MilpProblem constraint references a missing variable column."
                )
        for maximum in self.exact_maxima:
            if maximum.result_column >= expected_offset:
                raise ValidationError(
                    "MilpProblem exact maximum result column is missing."
                )
            for operand in maximum.operands:
                if operand.columns and operand.columns[-1] >= expected_offset:
                    raise ValidationError(
                        "MilpProblem exact maximum operand column is missing."
                    )
        priorities = [tier.priority for tier in self.objective_tiers]
        if priorities != sorted(priorities, reverse=True) or len(
            set(priorities)
        ) != len(priorities):
            raise ValidationError(
                "MilpProblem objective tiers must have unique priorities in descending order."
            )
        for tier in self.objective_tiers:
            expression = tier.expression
            if expression.columns and expression.columns[-1] >= expected_offset:
                raise ValidationError(
                    "MilpProblem objective references a missing variable column."
                )
        available_blocks = set(block_ids)
        binding_fields: set[str] = set()
        for binding in self.plan_bindings:
            if binding.block_id not in available_blocks:
                raise ValidationError(
                    f"PlanBinding {binding.field!r} references unknown block {binding.block_id!r}."
                )
            if binding.field in binding_fields:
                raise ValidationError(f"Duplicate PlanBinding field {binding.field!r}.")
            binding_fields.add(binding.field)

    @property
    def variable_count(self) -> int:
        return sum((block.size for block in self.variables))

    @property
    def is_mip(self) -> bool:
        return any(
            (
                block.variable_type in {VariableType.INTEGER, VariableType.BINARY}
                for block in self.variables
            )
        )

    @property
    def block_by_id(self) -> Mapping[str, VariableBlock]:
        return {block.id: block for block in self.variables}

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lower: list[float] = []
        upper: list[float] = []
        for block in self.variables:
            lower.extend(block.lower)
            upper.extend(block.upper)
        return (tuple(lower), tuple(upper))

    def validate_values(self, values: tuple[float, ...]) -> None:
        if len(values) != self.variable_count:
            raise ValidationError(
                f"MILP solution has {len(values)} values; expected {self.variable_count}."
            )
        if any((not math.isfinite(float(value)) for value in values)):
            raise ValidationError("MILP solution values must be finite.")


def expression_bounds(
    problem: MilpProblem, expression: AffineExpression
) -> tuple[float, float]:
    """Return exact interval-arithmetic bounds implied by variable bounds."""
    lower_bounds, upper_bounds = problem.bounds()
    lower = expression.constant
    upper = expression.constant
    for column, coefficient in zip(
        expression.columns, expression.coefficients, strict=True
    ):
        try:
            column_lower = lower_bounds[column]
            column_upper = upper_bounds[column]
        except IndexError as exc:
            raise InvariantError(
                "Expression bound calculation referenced a missing column."
            ) from exc
        if coefficient >= 0.0:
            lower += coefficient * column_lower
            upper += coefficient * column_upper
        else:
            lower += coefficient * column_upper
            upper += coefficient * column_lower
    return (lower, upper)
