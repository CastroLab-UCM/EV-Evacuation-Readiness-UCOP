"""User controls for Gurobi MILP planning."""

from __future__ import annotations
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from evac.errors import ValidationError


@dataclass(frozen=True, slots=True, kw_only=True)
class MILP:
    """Plan with Gurobi; the time budget spans all objective tiers.

    Additional native Gurobi options are passed through ``parameters``.
    """

    time_limit_seconds: float | None = None
    threads: int | None = None
    mip_gap: float | None = None
    mip_gap_absolute: float | None = None
    seed: int | None = None
    parameters: Mapping[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("time_limit_seconds", "mip_gap", "mip_gap_absolute"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValidationError(f"MILP.{name} must be numeric.")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValidationError(f"MILP.{name} must be finite and nonnegative.")
            if name == "time_limit_seconds" and value == 0:
                raise ValidationError("MILP.time_limit_seconds must be positive.")
            object.__setattr__(self, name, value)
        for name in ("threads", "seed"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValidationError(f"MILP.{name} must be an integer.")
            if value < 0 or (name == "threads" and value == 0):
                raise ValidationError(f"MILP.{name} is outside its documented range.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("MILP.parameters must be a mapping.")
        if any(not isinstance(name, str) for name in self.parameters):
            raise ValidationError("MILP.parameters keys must be strings.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
