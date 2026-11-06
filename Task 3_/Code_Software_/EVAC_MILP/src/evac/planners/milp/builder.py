"""Sparse linear expressions for MILP construction."""

from __future__ import annotations
import math
import numbers
import operator
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence
import numpy as np
from evac.errors import InvariantError, ValidationError
from evac.planners.milp.problem import (
    AffineExpression,
    ExactMaximum,
    MilpProblem,
    ObjectiveTier,
    PlanBinding,
    SparseConstraintMatrix,
    VariableBlock,
    VariableType,
    WeightedObjective,
)

_VTYPE_CODE = {
    VariableType.CONTINUOUS: "C",
    VariableType.INTEGER: "I",
    VariableType.BINARY: "B",
}


def _numeric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be numeric; got {type(value).__name__}.")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{label} must not be NaN.")
    return result


@dataclass(frozen=True, slots=True)
class LinearExpression:
    terms: tuple[tuple[int, float], ...] = ()
    constant_value: float = 0.0

    def __post_init__(self) -> None:
        normalized = AffineExpression.from_terms(
            self.terms, constant=self.constant_value
        )
        object.__setattr__(
            self,
            "terms",
            tuple(zip(normalized.columns, normalized.coefficients, strict=True)),
        )
        object.__setattr__(self, "constant_value", normalized.constant)

    @classmethod
    def constant(cls, value: float = 0.0) -> LinearExpression:
        return cls(constant_value=_numeric(value, label="linear-expression constant"))

    @classmethod
    def from_affine(cls, value: AffineExpression) -> LinearExpression:
        return cls(
            terms=tuple(zip(value.columns, value.coefficients, strict=True)),
            constant_value=value.constant,
        )

    def to_affine(self) -> AffineExpression:
        return AffineExpression(
            columns=tuple((column for column, _ in self.terms)),
            coefficients=tuple((coefficient for _, coefficient in self.terms)),
            constant=self.constant_value,
        )

    def __add__(self, other: object) -> object:
        if isinstance(other, ExpressionArray):
            return other.__radd__(self)
        right = as_linear_expression(other)
        return LinearExpression(
            terms=self.terms + right.terms,
            constant_value=self.constant_value + right.constant_value,
        )

    def __radd__(self, other: object) -> object:
        return self + other

    def __sub__(self, other: object) -> object:
        if isinstance(other, ExpressionArray):
            return other.__rsub__(self)
        return self + -as_linear_expression(other)

    def __rsub__(self, other: object) -> object:
        if isinstance(other, ExpressionArray):
            return other - self
        return as_linear_expression(other) - self

    def __mul__(self, other: object) -> LinearExpression:
        factor = _numeric(other, label="linear-expression multiplier")
        return LinearExpression(
            terms=tuple(
                ((column, factor * coefficient) for column, coefficient in self.terms)
            ),
            constant_value=factor * self.constant_value,
        )

    def __rmul__(self, other: object) -> LinearExpression:
        return self * other

    def __truediv__(self, other: object) -> LinearExpression:
        divisor = _numeric(other, label="linear-expression divisor")
        if divisor == 0.0:
            raise ZeroDivisionError("Cannot divide a linear expression by zero.")
        return self * (1.0 / divisor)

    def __neg__(self) -> LinearExpression:
        return self * -1.0

    def __le__(self, other: object) -> PendingConstraint:
        return PendingConstraint(self - other, lower=-math.inf, upper=0.0)

    def __ge__(self, other: object) -> PendingConstraint:
        return PendingConstraint(self - other, lower=0.0, upper=math.inf)

    def __eq__(self, other: object) -> PendingConstraint:
        return PendingConstraint(self - other, lower=0.0, upper=0.0)


@dataclass(slots=True, eq=False)
class ScalarVariable:
    column: int
    name: str
    variable_type: VariableType
    lower: float
    upper: float
    start: float | None = None
    solution_value: float | None = None

    @property
    def VarName(self) -> str:
        return self.name

    @property
    def VType(self) -> str:
        return _VTYPE_CODE[self.variable_type]

    @property
    def LB(self) -> float:
        return self.lower

    @LB.setter
    def LB(self, value: float) -> None:
        self.lower = _numeric(value, label=f"{self.name} lower bound")

    @property
    def UB(self) -> float:
        return self.upper

    @UB.setter
    def UB(self, value: float) -> None:
        self.upper = _numeric(value, label=f"{self.name} upper bound")

    @property
    def Start(self) -> float | None:
        return self.start

    @Start.setter
    def Start(self, value: float) -> None:
        self.start = _numeric(value, label=f"{self.name} warm start")

    @property
    def X(self) -> float:
        if self.solution_value is None:
            raise InvariantError(
                f"Variable {self.name!r} has no accepted solution value."
            )
        return self.solution_value

    def expression(self) -> LinearExpression:
        return LinearExpression(terms=((self.column, 1.0),))

    def __add__(self, other: object) -> object:
        return self.expression() + other

    def __radd__(self, other: object) -> object:
        return self.expression() + other

    def __sub__(self, other: object) -> object:
        return self.expression() - other

    def __rsub__(self, other: object) -> object:
        return as_linear_expression(other) - self.expression()

    def __mul__(self, other: object) -> LinearExpression:
        return self.expression() * other

    def __rmul__(self, other: object) -> LinearExpression:
        return self.expression() * other

    def __truediv__(self, other: object) -> LinearExpression:
        return self.expression() / other

    def __neg__(self) -> LinearExpression:
        return -self.expression()

    def __le__(self, other: object) -> PendingConstraint:
        return self.expression() <= other

    def __ge__(self, other: object) -> PendingConstraint:
        return self.expression() >= other

    def __eq__(self, other: object) -> PendingConstraint:
        return self.expression() == other


def as_linear_expression(value: object) -> LinearExpression:
    if isinstance(value, LinearExpression):
        return value
    if isinstance(value, ScalarVariable):
        return value.expression()
    if isinstance(value, numbers.Real) and (not isinstance(value, bool)):
        return LinearExpression.constant(float(value))
    raise TypeError(f"Expected a scalar linear expression; got {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class PendingConstraint:
    expression: LinearExpression
    lower: float
    upper: float

    def __bool__(self) -> bool:
        raise TypeError("MILP constraints cannot be used as booleans.")


@dataclass(frozen=True, slots=True)
class ConstraintArray:
    values: np.ndarray

    def flattened(self) -> Iterator[tuple[tuple[int, ...], PendingConstraint]]:
        for index in np.ndindex(self.values.shape):
            value = self.values[index]
            if not isinstance(value, PendingConstraint):
                raise TypeError("ConstraintArray contains a non-constraint value.")
            yield (index, value)


class ExpressionArray:
    __array_priority__ = 10000

    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=object)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def size(self) -> int:
        return self._values.size

    @property
    def T(self) -> ExpressionArray:
        return ExpressionArray(self._values.T)

    @property
    def LB(self) -> np.ndarray:
        return np.vectorize(lambda item: _as_scalar_variable(item).LB, otypes=[float])(
            self._values
        )

    @LB.setter
    def LB(self, values: object) -> None:
        broadcast = np.broadcast_to(np.asarray(values, dtype=float), self.shape)
        for index in np.ndindex(self.shape):
            _as_scalar_variable(self._values[index]).LB = float(broadcast[index])

    @property
    def UB(self) -> np.ndarray:
        return np.vectorize(lambda item: _as_scalar_variable(item).UB, otypes=[float])(
            self._values
        )

    @UB.setter
    def UB(self, values: object) -> None:
        broadcast = np.broadcast_to(np.asarray(values, dtype=float), self.shape)
        for index in np.ndindex(self.shape):
            _as_scalar_variable(self._values[index]).UB = float(broadcast[index])

    def tolist(self) -> list[object]:
        return self._values.tolist()

    def reshape(self, *shape: int) -> ExpressionArray:
        return ExpressionArray(self._values.reshape(*shape))

    def ravel(self) -> ExpressionArray:
        return ExpressionArray(self._values.ravel())

    def sum(self, axis: int | tuple[int, ...] | None = None) -> object:
        if axis is None:
            return _combine_linear_expressions(self._values.flat)
        if self._values.ndim == 0:
            raw_axes = (axis,) if isinstance(axis, numbers.Integral) else axis
            if len(raw_axes) > 1 or any((value not in (0, -1) for value in raw_axes)):
                raise ValueError(
                    f"axis {axis!r} is invalid for a scalar expression array."
                )
            return _combine_linear_expressions(self._values.flat)
        raw_axes = (axis,) if isinstance(axis, numbers.Integral) else axis
        normalized_axes: list[int] = []
        for raw_axis in raw_axes:
            try:
                normalized_axis = operator.index(raw_axis)
            except TypeError as exc:
                raise TypeError(
                    "expression-array reduction axes must be integers."
                ) from exc
            if normalized_axis < 0:
                normalized_axis += self._values.ndim
            if normalized_axis < 0 or normalized_axis >= self._values.ndim:
                raise ValueError(
                    f"axis {raw_axis} is out of bounds for expression array with dimension {self._values.ndim}."
                )
            if normalized_axis in normalized_axes:
                raise ValueError("duplicate value in expression-array reduction axis.")
            normalized_axes.append(normalized_axis)
        if not normalized_axes:
            return ExpressionArray(self._values.copy())
        retained_axes = tuple(
            (
                index
                for index in range(self._values.ndim)
                if index not in normalized_axes
            )
        )
        output_shape = tuple((self._values.shape[index] for index in retained_axes))
        if not output_shape:
            return _combine_linear_expressions(self._values.flat)
        output = np.empty(output_shape, dtype=object)
        for output_index in np.ndindex(output_shape):
            selector: list[int | slice] = [slice(None)] * self._values.ndim
            for retained_axis, coordinate in zip(
                retained_axes, output_index, strict=True
            ):
                selector[retained_axis] = coordinate
            output[output_index] = _combine_linear_expressions(
                self._values[tuple(selector)].flat
            )
        return ExpressionArray(output)

    def __getitem__(self, key: object) -> object:
        result = self._values[key]
        if isinstance(result, np.ndarray):
            return ExpressionArray(result)
        return result

    def _binary(self, other: object, operation: str) -> ExpressionArray:
        right = (
            other._values
            if isinstance(other, ExpressionArray)
            else np.asarray(other, dtype=object)
        )
        left_values, right_values = np.broadcast_arrays(self._values, right)
        output = np.empty(left_values.shape, dtype=object)
        for index in np.ndindex(output.shape):
            left_expr = as_linear_expression(left_values[index])
            right_value = right_values[index]
            if operation == "add":
                output[index] = left_expr + right_value
            elif operation == "sub":
                output[index] = left_expr - right_value
            elif operation == "rsub":
                output[index] = as_linear_expression(right_value) - left_expr
            elif operation == "mul":
                output[index] = left_expr * right_value
            elif operation == "div":
                output[index] = left_expr / right_value
            else:
                raise InvariantError(
                    f"Unknown expression-array operation {operation!r}."
                )
        return ExpressionArray(output)

    def __add__(self, other: object) -> ExpressionArray:
        return self._binary(other, "add")

    def __radd__(self, other: object) -> ExpressionArray:
        return self + other

    def __sub__(self, other: object) -> ExpressionArray:
        return self._binary(other, "sub")

    def __rsub__(self, other: object) -> ExpressionArray:
        return self._binary(other, "rsub")

    def __mul__(self, other: object) -> ExpressionArray:
        return self._binary(other, "mul")

    def __rmul__(self, other: object) -> ExpressionArray:
        return self * other

    def __truediv__(self, other: object) -> ExpressionArray:
        return self._binary(other, "div")

    def __neg__(self) -> ExpressionArray:
        return self * -1.0

    def _compare(self, other: object, relation: str) -> ConstraintArray:
        right = (
            other._values
            if isinstance(other, ExpressionArray)
            else np.asarray(other, dtype=object)
        )
        left_values, right_values = np.broadcast_arrays(self._values, right)
        output = np.empty(left_values.shape, dtype=object)
        for index in np.ndindex(output.shape):
            left_expr = as_linear_expression(left_values[index])
            if relation == "eq":
                output[index] = left_expr == right_values[index]
            elif relation == "le":
                output[index] = left_expr <= right_values[index]
            elif relation == "ge":
                output[index] = left_expr >= right_values[index]
            else:
                raise InvariantError(f"Unknown constraint relation {relation!r}.")
        return ConstraintArray(output)

    def __eq__(self, other: object) -> ConstraintArray:
        return self._compare(other, "eq")

    def __le__(self, other: object) -> ConstraintArray:
        return self._compare(other, "le")

    def __ge__(self, other: object) -> ConstraintArray:
        return self._compare(other, "ge")

    def __matmul__(self, other: object) -> object:
        right = (
            other._values
            if isinstance(other, ExpressionArray)
            else np.asarray(other, dtype=object)
        )
        return _wrap_expression_result(np.matmul(self._values, right))

    def __rmatmul__(self, other: object) -> object:
        if hasattr(other, "tocsr"):
            return _sparse_matrix_times_expressions(other, self)
        dense = np.asarray(other)
        if dense.ndim in {1, 2} and self._values.ndim in {1, 2}:
            return _dense_matrix_times_expressions(dense, self)
        return _wrap_expression_result(np.matmul(dense.astype(object), self._values))


class VariableArray(ExpressionArray):
    def __init__(
        self, *, block_id: str, values: np.ndarray, variable_type: VariableType
    ) -> None:
        super().__init__(values)
        self.block_id = block_id
        self.variable_type = variable_type

    @property
    def VType(self) -> np.ndarray:
        return np.full(self.shape, _VTYPE_CODE[self.variable_type], dtype=object)

    @property
    def LB(self) -> np.ndarray:
        return np.vectorize(lambda item: item.LB, otypes=[float])(self._values)

    @LB.setter
    def LB(self, values: object) -> None:
        broadcast = np.broadcast_to(np.asarray(values, dtype=float), self.shape)
        for index in np.ndindex(self.shape):
            self._values[index].LB = float(broadcast[index])

    @property
    def UB(self) -> np.ndarray:
        return np.vectorize(lambda item: item.UB, otypes=[float])(self._values)

    @UB.setter
    def UB(self, values: object) -> None:
        broadcast = np.broadcast_to(np.asarray(values, dtype=float), self.shape)
        for index in np.ndindex(self.shape):
            self._values[index].UB = float(broadcast[index])

    @property
    def Start(self) -> np.ndarray:
        return np.vectorize(lambda item: item.Start, otypes=[object])(self._values)

    @Start.setter
    def Start(self, values: object) -> None:
        broadcast = np.broadcast_to(np.asarray(values, dtype=float), self.shape)
        for index in np.ndindex(self.shape):
            self._values[index].Start = float(broadcast[index])

    @property
    def X(self) -> np.ndarray:
        return np.vectorize(lambda item: item.X, otypes=[float])(self._values)


def _wrap_expression_result(value: object) -> object:
    if isinstance(value, np.ndarray):
        return ExpressionArray(value)
    return value


def _combine_linear_expressions(values: Iterable[object]) -> LinearExpression:
    terms: list[tuple[int, float]] = []
    constant = 0.0
    for value in values:
        expression = as_linear_expression(value)
        terms.extend(expression.terms)
        constant += expression.constant_value
    return LinearExpression(terms=tuple(terms), constant_value=constant)


def _as_scalar_variable(value: object) -> ScalarVariable:
    if not isinstance(value, ScalarVariable):
        raise TypeError("Variable bound assignment requires a scalar variable view.")
    return value


def _sparse_matrix_times_expressions(
    matrix: object, expressions: ExpressionArray
) -> object:
    csr = matrix.tocsr()
    values = expressions._values
    if values.ndim == 0:
        raise ValueError(
            "Sparse matrix multiplication requires a variable vector or matrix."
        )
    if csr.shape[1] != values.shape[0]:
        raise ValueError(
            f"Sparse matrix width {csr.shape[1]} does not match expression leading dimension {values.shape[0]}."
        )
    trailing_shape = values.shape[1:]
    output = np.empty((csr.shape[0], *trailing_shape), dtype=object)
    for row in range(csr.shape[0]):
        trailing_indices = np.ndindex(trailing_shape) if trailing_shape else [()]
        for trailing_index in trailing_indices:
            terms: list[tuple[int, float]] = []
            constant = 0.0
            for position in range(csr.indptr[row], csr.indptr[row + 1]):
                source_index = (int(csr.indices[position]), *trailing_index)
                factor = float(csr.data[position])
                source = as_linear_expression(values[source_index])
                terms.extend(
                    (
                        (column, factor * coefficient)
                        for column, coefficient in source.terms
                    )
                )
                constant += factor * source.constant_value
            output[row, *trailing_index] = LinearExpression(
                terms=tuple(terms), constant_value=constant
            )
    return ExpressionArray(output)


def _dense_matrix_times_expressions(
    matrix: np.ndarray, expressions: ExpressionArray
) -> object:
    values = expressions._values
    matrix_width = matrix.shape[-1]
    expression_height = values.shape[0]
    if matrix_width != expression_height:
        raise ValueError(
            f"Dense matrix width {matrix_width} does not match expression leading dimension {expression_height}."
        )
    matrix_is_vector = matrix.ndim == 1
    expressions_are_vector = values.ndim == 1
    row_count = 1 if matrix_is_vector else matrix.shape[0]
    column_count = 1 if expressions_are_vector else values.shape[1]
    output = np.empty((row_count, column_count), dtype=object)
    for row in range(row_count):
        factors = matrix if matrix_is_vector else matrix[row, :]
        for column in range(column_count):
            source = values if expressions_are_vector else values[:, column]
            output[row, column] = _combine_weighted_linear_expressions(factors, source)
    if matrix_is_vector and expressions_are_vector:
        return output[0, 0]
    if matrix_is_vector:
        return ExpressionArray(output[0, :])
    if expressions_are_vector:
        return ExpressionArray(output[:, 0])
    return ExpressionArray(output)


def _combine_weighted_linear_expressions(
    factors: Iterable[object], values: Iterable[object]
) -> LinearExpression:
    terms: list[tuple[int, float]] = []
    constant = 0.0
    for factor, value in zip(factors, values, strict=True):
        numeric_factor = _numeric(factor, label="dense expression multiplier")
        expression = as_linear_expression(value)
        terms.extend(
            (
                (column, numeric_factor * coefficient)
                for column, coefficient in expression.terms
            )
        )
        constant += numeric_factor * expression.constant_value
    return LinearExpression(terms=tuple(terms), constant_value=constant)


class SparseMilpBuilder:
    """Mutable formulation builder whose only product is an immutable ``MilpProblem``."""

    def __init__(self, name: str = "EVac MILP") -> None:
        self.name = name
        self._variables: list[VariableArray | ScalarVariable] = []
        self._blocks: list[VariableBlock] = []
        self._rows: list[tuple[str, AffineExpression, float, float]] = []
        self._maxima: list[ExactMaximum] = []
        self._objective_records: list[
            tuple[str, int, float, float, float, AffineExpression]
        ] = []
        self._single_objective: AffineExpression | None = None
        self._parameter_values: dict[str, object] = {}

    @property
    def variable_count(self) -> int:
        return sum((block.size for block in self._blocks))

    def set_parameter(self, name: str, value: object) -> None:
        self._parameter_values[name] = value

    def bind_block_id(
        self, variables: VariableArray | ScalarVariable, block_id: str
    ) -> None:
        if not block_id:
            raise ValidationError("MILP block identity must be non-empty.")
        if any((block.id == block_id for block in self._blocks)):
            raise ValidationError(f"Duplicate MILP block identity {block_id!r}.")
        try:
            index = next(
                (
                    index
                    for index, declared in enumerate(self._variables)
                    if declared is variables
                )
            )
        except StopIteration as exc:
            raise InvariantError(
                "Cannot bind an undeclared MILP variable block."
            ) from exc
        self._blocks[index] = replace(self._blocks[index], id=block_id)
        if isinstance(variables, VariableArray):
            variables.block_id = block_id

    def add_variable_block(
        self,
        shape: int | Sequence[int],
        *,
        lower: object = 0.0,
        upper: object = math.inf,
        variable_type: VariableType = VariableType.CONTINUOUS,
        name: str,
        block_id: str | None = None,
    ) -> VariableArray:
        normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        lower_values = np.broadcast_to(
            np.asarray(lower, dtype=float), normalized_shape
        ).copy()
        upper_values = np.broadcast_to(
            np.asarray(upper, dtype=float), normalized_shape
        ).copy()
        identifier = block_id or name
        offset = self.variable_count
        block = VariableBlock(
            id=identifier,
            shape=normalized_shape,
            variable_type=variable_type,
            lower=tuple((float(value) for value in lower_values.reshape(-1))),
            upper=tuple((float(value) for value in upper_values.reshape(-1))),
            offset=offset,
        )
        values = np.empty(normalized_shape, dtype=object)
        for flat_index, index in enumerate(np.ndindex(normalized_shape)):
            suffix = ",".join((str(item) for item in index))
            values[index] = ScalarVariable(
                column=offset + flat_index,
                name=f"{name}[{suffix}]",
                variable_type=variable_type,
                lower=float(lower_values[index]),
                upper=float(upper_values[index]),
            )
        result = VariableArray(
            block_id=identifier, values=values, variable_type=variable_type
        )
        self._blocks.append(block)
        self._variables.append(result)
        return result

    def add_variable(
        self,
        *,
        lower: float = 0.0,
        upper: float = math.inf,
        variable_type: VariableType = VariableType.CONTINUOUS,
        name: str,
        block_id: str | None = None,
    ) -> ScalarVariable:
        identifier = block_id or name
        offset = self.variable_count
        block = VariableBlock(
            id=identifier,
            shape=(),
            variable_type=variable_type,
            lower=(float(lower),),
            upper=(float(upper),),
            offset=offset,
        )
        result = ScalarVariable(
            column=offset,
            name=name,
            variable_type=variable_type,
            lower=float(lower),
            upper=float(upper),
        )
        self._blocks.append(block)
        self._variables.append(result)
        return result

    def add_constraint(
        self, constraint: PendingConstraint | ConstraintArray, *, name: str
    ) -> None:
        if isinstance(constraint, ConstraintArray):
            for index, item in constraint.flattened():
                suffix = ",".join((str(value) for value in index))
                self._append_constraint(item, name=f"{name}[{suffix}]")
            return
        self._append_constraint(constraint, name=name)

    def add_constraints(
        self, constraints: Iterable[PendingConstraint], *, name: str
    ) -> None:
        for index, constraint in enumerate(constraints):
            self._append_constraint(constraint, name=f"{name}[{index}]")

    def _append_constraint(self, constraint: PendingConstraint, *, name: str) -> None:
        if not isinstance(constraint, PendingConstraint):
            raise TypeError(
                f"Expected a pending MILP constraint; got {type(constraint).__name__}."
            )
        expression = constraint.expression.to_affine()
        self._rows.append(
            (
                name,
                AffineExpression(
                    columns=expression.columns, coefficients=expression.coefficients
                ),
                constraint.lower - expression.constant,
                constraint.upper - expression.constant,
            )
        )

    def add_exact_maximum(
        self, result: ScalarVariable, operands: Iterable[object], *, name: str
    ) -> None:
        self._maxima.append(
            ExactMaximum(
                id=name,
                result_column=result.column,
                operands=tuple(
                    (as_linear_expression(value).to_affine() for value in operands)
                ),
            )
        )

    def set_objective(self, expression: object) -> None:
        self._single_objective = as_linear_expression(expression).to_affine()
        self._objective_records.clear()

    def add_objective(
        self,
        expression: object,
        *,
        index: int,
        priority: int,
        weight: float,
        absolute_degradation: float = 1e-06,
        relative_degradation: float = 0.0,
        name: str,
    ) -> None:
        if index != len(self._objective_records):
            raise ValidationError(
                "MILP objectives must be registered in contiguous index order."
            )
        self._single_objective = None
        self._objective_records.append(
            (
                name,
                int(priority),
                float(weight),
                float(absolute_degradation),
                float(relative_degradation),
                as_linear_expression(expression).to_affine(),
            )
        )

    def to_problem(
        self, *, identity: str, plan_bindings: Iterable[PlanBinding] = ()
    ) -> MilpProblem:
        blocks = self._current_blocks()
        tiers = self._objective_tiers()
        return MilpProblem(
            identity=identity,
            variables=blocks,
            constraints=SparseConstraintMatrix.from_rows(self._rows),
            exact_maxima=tuple(self._maxima),
            objective_tiers=tiers,
            plan_bindings=tuple(plan_bindings),
        )

    def _current_blocks(self) -> tuple[VariableBlock, ...]:
        blocks: list[VariableBlock] = []
        for declared, variables in zip(self._blocks, self._variables, strict=True):
            if isinstance(variables, ScalarVariable):
                lower = (variables.lower,)
                upper = (variables.upper,)
            else:
                flattened = variables._values.reshape(-1)
                lower = tuple((float(item.lower) for item in flattened))
                upper = tuple((float(item.upper) for item in flattened))
            blocks.append(
                VariableBlock(
                    id=declared.id,
                    shape=declared.shape,
                    variable_type=declared.variable_type,
                    lower=lower,
                    upper=upper,
                    offset=declared.offset,
                )
            )
        return tuple(blocks)

    def _objective_tiers(self) -> tuple[ObjectiveTier, ...]:
        if self._single_objective is not None:
            return (
                ObjectiveTier(
                    id="primary",
                    priority=1,
                    objectives=(
                        WeightedObjective(
                            id="primary", expression=self._single_objective
                        ),
                    ),
                ),
            )
        grouped: dict[int, list[tuple[str, float, float, float, AffineExpression]]] = {}
        for (
            name,
            priority,
            weight,
            absolute,
            relative,
            expression,
        ) in self._objective_records:
            grouped.setdefault(priority, []).append(
                (name, weight, absolute, relative, expression)
            )
        tiers: list[ObjectiveTier] = []
        for priority in sorted(grouped, reverse=True):
            records = grouped[priority]
            absolute_values = {record[2] for record in records}
            relative_values = {record[3] for record in records}
            if len(absolute_values) != 1 or len(relative_values) != 1:
                raise ValidationError(
                    f"Objectives sharing priority {priority} must share degradation tolerances."
                )
            tiers.append(
                ObjectiveTier(
                    id=f"priority-{priority}",
                    priority=priority,
                    objectives=tuple(
                        (
                            WeightedObjective(
                                id=name, expression=expression, weight=weight
                            )
                            for name, weight, _, _, expression in records
                        )
                    ),
                    absolute_degradation=next(iter(absolute_values)),
                    relative_degradation=next(iter(relative_values)),
                )
            )
        return tuple(tiers)

    def assign_solution(self, values: Sequence[float]) -> None:
        if len(values) != self.variable_count:
            raise ValidationError(
                f"Native solution has {len(values)} values; expected {self.variable_count}."
            )
        for variables in self._variables:
            if isinstance(variables, ScalarVariable):
                variables.solution_value = float(values[variables.column])
                continue
            for variable in variables._values.reshape(-1):
                variable.solution_value = float(values[variable.column])

    @property
    def NumVars(self) -> int:
        return self.variable_count

    @property
    def NumConstrs(self) -> int:
        return len(self._rows)

    @property
    def NumGenConstrs(self) -> int:
        return len(self._maxima)

    @property
    def NumIntVars(self) -> int:
        return sum(
            (
                block.size
                for block in self._current_blocks()
                if block.variable_type in {VariableType.INTEGER, VariableType.BINARY}
            )
        )

    @property
    def NumBinVars(self) -> int:
        return sum(
            (
                block.size
                for block in self._current_blocks()
                if block.variable_type is VariableType.BINARY
            )
        )

    @property
    def NumQConstrs(self) -> int:
        return 0

    @property
    def NumNZs(self) -> int:
        return sum((len(expression.columns) for _, expression, _, _ in self._rows))

    @property
    def IsMIP(self) -> int:
        return int(self.NumIntVars > 0)

    @property
    def IsMultiObj(self) -> int:
        return int(len(self._objective_records) > 1)

    def update(self) -> None:
        return None
