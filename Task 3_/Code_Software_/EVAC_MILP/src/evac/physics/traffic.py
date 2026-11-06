from __future__ import annotations
import bisect
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping
from evac.errors import ValidationError
from evac.physics.traffic_kernel import traffic_exit_time_kernel


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite; got {result!r}.")
    return result


@dataclass(frozen=True, slots=True, order=True)
class ProfilePoint:
    elapsed_seconds: float
    multiplier: float

    def __post_init__(self) -> None:
        elapsed = _finite(self.elapsed_seconds, label="profile elapsed_seconds")
        multiplier = _finite(self.multiplier, label="profile multiplier")
        if elapsed < 0.0:
            raise ValidationError("profile elapsed_seconds must be nonnegative.")
        if multiplier <= 0.0:
            raise ValidationError("profile multiplier must be positive.")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "multiplier", multiplier)


@dataclass(frozen=True, slots=True)
class ExogenousProfile:
    points: tuple[ProfilePoint, ...]
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _multipliers: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValidationError("exogenous_profile must contain at least one point.")
        if self.points[0].elapsed_seconds != 0.0:
            raise ValidationError("exogenous_profile must start at elapsed second 0.")
        times = tuple((point.elapsed_seconds for point in self.points))
        if any((right <= left for left, right in zip(times, times[1:]))):
            raise ValidationError(
                "exogenous_profile times must be strictly increasing."
            )
        object.__setattr__(self, "_times", times)
        object.__setattr__(
            self, "_multipliers", tuple((point.multiplier for point in self.points))
        )

    @classmethod
    def constant(cls, multiplier: float) -> ExogenousProfile:
        return cls((ProfilePoint(0.0, multiplier),))

    @classmethod
    def from_data(cls, value: object) -> ExogenousProfile:
        if not isinstance(value, Mapping):
            raise ValidationError("exogenous_profile must be a mapping.")
        kind = value.get("kind")
        if kind == "constant":
            if set(value) != {"kind", "multiplier"}:
                raise ValidationError(
                    "constant exogenous_profile requires exactly kind and multiplier."
                )
            return cls.constant(value["multiplier"])
        if kind == "piecewise_linear":
            if set(value) != {"kind", "points"}:
                raise ValidationError(
                    "piecewise_linear exogenous_profile requires exactly kind and points."
                )
            raw_points = value["points"]
            if not isinstance(raw_points, list):
                raise ValidationError("piecewise_linear points must be a list.")
            points: list[ProfilePoint] = []
            for index, raw_point in enumerate(raw_points):
                if not isinstance(raw_point, Mapping) or set(raw_point) != {
                    "elapsed_seconds",
                    "multiplier",
                }:
                    raise ValidationError(
                        f"profile point {index} must contain elapsed_seconds and multiplier."
                    )
                points.append(
                    ProfilePoint(raw_point["elapsed_seconds"], raw_point["multiplier"])
                )
            return cls(tuple(points))
        raise ValidationError(
            "exogenous_profile.kind must be 'constant' or 'piecewise_linear'."
        )

    def to_data(self) -> dict[str, Any]:
        if len(self.points) == 1:
            return {"kind": "constant", "multiplier": self.points[0].multiplier}
        return {
            "kind": "piecewise_linear",
            "points": [
                {
                    "elapsed_seconds": point.elapsed_seconds,
                    "multiplier": point.multiplier,
                }
                for point in self.points
            ],
        }

    def multiplier_at(self, elapsed_seconds: float) -> float:
        elapsed = _finite(elapsed_seconds, label="elapsed_seconds")
        if elapsed < 0.0:
            raise ValidationError("elapsed_seconds must be nonnegative.")
        index = bisect.bisect_right(self._times, elapsed) - 1
        if index >= len(self.points) - 1:
            return self.points[-1].multiplier
        left = self.points[index]
        right = self.points[index + 1]
        fraction = (elapsed - left.elapsed_seconds) / (
            right.elapsed_seconds - left.elapsed_seconds
        )
        return left.multiplier + fraction * (right.multiplier - left.multiplier)

    def exit_time(
        self, *, entry_time_seconds: float, free_flow_duration_seconds: float
    ) -> float:
        entry = _finite(entry_time_seconds, label="entry_time_seconds")
        free_flow = _finite(
            free_flow_duration_seconds, label="free_flow_duration_seconds"
        )
        if entry < 0.0:
            raise ValidationError("entry_time_seconds must be nonnegative.")
        if free_flow < 0.0:
            raise ValidationError("free_flow_duration_seconds must be nonnegative.")
        return traffic_exit_time_kernel(
            self._times, self._multipliers, entry, free_flow
        )


def is_unit_constant(profile: ExogenousProfile) -> bool:
    return profile == ExogenousProfile.constant(1.0)


@dataclass(frozen=True, slots=True)
class TrafficSpec:
    exogenous_profile: ExogenousProfile

    def exit_time(
        self, *, entry_time_seconds: float, free_flow_duration_seconds: float
    ) -> float:
        return self.exogenous_profile.exit_time(
            entry_time_seconds=entry_time_seconds,
            free_flow_duration_seconds=free_flow_duration_seconds,
        )
