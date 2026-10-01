"""MILP objective certificates and accepted numerical solutions."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from evac.errors import ValidationError
from evac.planners.milp.problem import MilpProblem, VariableBlock


class NormalizedTermination(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    INTERRUPTED = "interrupted"
    NO_INCUMBENT = "no_incumbent"


@dataclass(frozen=True, slots=True)
class WarmStart:
    values: Mapping[str, tuple[float | None, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def warm_start_columns(
    start: WarmStart, problem: MilpProblem
) -> tuple[list[int], list[float]]:
    """Map a `WarmStart`'s per-block values to `(column, value)` pairs
    against `problem`'s blocks, skipping `None` entries; the single owner of
    the start-to-column mapping between objective tiers."""
    blocks = problem.block_by_id
    indices: list[int] = []
    values: list[float] = []
    for block_id, block_values in start.values.items():
        try:
            block = blocks[block_id]
        except KeyError as exc:
            raise ValueError(f"Unknown MILP warm-start block {block_id!r}.") from exc
        if len(block_values) != block.size:
            raise ValueError(
                f"MILP warm-start block {block_id!r} contains {len(block_values)} values; expected {block.size}."
            )
        for offset, value in enumerate(block_values):
            if value is not None:
                indices.append(block.offset + offset)
                values.append(float(value))
    return (indices, values)


@dataclass(frozen=True, slots=True)
class Incumbent:
    raw_values: tuple[float, ...]
    variable_blocks: tuple[VariableBlock, ...]

    def __post_init__(self) -> None:
        expected = sum((block.size for block in self.variable_blocks))
        if len(self.raw_values) != expected:
            raise ValidationError(
                f"Incumbent contains {len(self.raw_values)} values; expected {expected}."
            )
        if any((not math.isfinite(float(value)) for value in self.raw_values)):
            raise ValidationError("Incumbent raw values must be finite.")


@dataclass(frozen=True, slots=True)
class ObjectiveCertificate:
    tier_id: str
    incumbent_value: float | None
    objective_bound: float | None
    normalized_gap: float | None
    termination: NormalizedTermination

    def __post_init__(self) -> None:
        if not self.tier_id:
            raise ValidationError("ObjectiveCertificate.tier_id must be non-empty.")
        for name in ("incumbent_value", "objective_bound", "normalized_gap"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value))):
                raise ValidationError(
                    f"ObjectiveCertificate.{name} must be finite when provided."
                )
        if self.normalized_gap is not None and self.normalized_gap < 0.0:
            raise ValidationError(
                "ObjectiveCertificate.normalized_gap must be nonnegative."
            )


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    provider_status: str | int | None = None
    provider_version: str | None = None
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "telemetry", MappingProxyType(dict(self.telemetry)))


@dataclass(frozen=True, slots=True)
class SolveResult:
    termination: NormalizedTermination
    incumbent: Incumbent | None
    certificates: tuple[ObjectiveCertificate, ...]
    diagnostics: SolverDiagnostics
    terminal_cause: str | None = None

    def __post_init__(self) -> None:
        if self.termination in {
            NormalizedTermination.OPTIMAL,
            NormalizedTermination.FEASIBLE,
        }:
            if self.incumbent is None:
                raise ValidationError(
                    "Optimal or feasible SolveResult requires an incumbent."
                )
        elif self.incumbent is not None and self.termination in {
            NormalizedTermination.INFEASIBLE,
            NormalizedTermination.UNBOUNDED,
            NormalizedTermination.INFEASIBLE_OR_UNBOUNDED,
            NormalizedTermination.NO_INCUMBENT,
        }:
            raise ValidationError(
                f"{self.termination.value} SolveResult cannot contain an incumbent."
            )
        if self.terminal_cause is not None and (not self.terminal_cause):
            raise ValidationError(
                "SolveResult.terminal_cause must be non-empty when provided."
            )


@dataclass(frozen=True, slots=True)
class GurobiProbe:
    dependency_installed: bool
    dependency_version: str | None
    runtime_checked: bool
    runtime_usable: bool | None
    failure_kind: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class NativeSolveOutcome:
    termination: NormalizedTermination
    values: tuple[float, ...] | None
    objective_value: float | None
    objective_bound: float | None
    provider_status: str | int | None
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    terminal_cause: str | None = None

    def __post_init__(self) -> None:
        if self.values is not None and any(
            (not math.isfinite(float(value)) for value in self.values)
        ):
            raise ValidationError("Native MILP solution values must be finite.")
        for name in ("objective_value", "objective_bound"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value))):
                raise ValidationError(
                    f"NativeSolveOutcome.{name} must be finite when provided."
                )
        if self.termination in {
            NormalizedTermination.OPTIMAL,
            NormalizedTermination.FEASIBLE,
        }:
            if self.values is None:
                raise ValidationError(
                    "A native optimal or feasible outcome requires values."
                )
        if self.terminal_cause is not None and (not self.terminal_cause):
            raise ValidationError(
                "NativeSolveOutcome.terminal_cause must be non-empty when provided."
            )
        object.__setattr__(self, "telemetry", MappingProxyType(dict(self.telemetry)))


def normalized_mip_gap(
    incumbent_value: float | None, objective_bound: float | None
) -> float | None:
    """Return the shared minimization gap; zero incumbents have no relative gap."""
    if incumbent_value is None or objective_bound is None or incumbent_value == 0.0:
        return None
    if not math.isfinite(incumbent_value) or not math.isfinite(objective_bound):
        return None
    return abs(incumbent_value - objective_bound) / abs(incumbent_value)
