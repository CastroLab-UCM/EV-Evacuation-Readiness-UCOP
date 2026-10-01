"""Physical units and conversions."""

from __future__ import annotations
import math
import numbers
from types import MappingProxyType
from typing import Any, Mapping
from evac.errors import ValidationError

SECONDS_PER_HOUR = 3600.0
CANONICAL_UNIT_SYSTEM: Mapping[str, str] = MappingProxyType(
    {
        "energy": "kWh",
        "power": "kW",
        "time": "s",
        "distance": "m",
        "vehicle_efficiency": "m/kWh",
    }
)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{label} must be a finite number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite; got {result!r}.")
    return result


def _energy_kwh_from_kw_seconds(power_kw: Any, duration_seconds: Any) -> Any:
    """Convert numeric arrays or MILP expressions using the canonical units."""
    return power_kw * duration_seconds / SECONDS_PER_HOUR


def energy_kwh_from_kw_seconds(power_kw: float, duration_seconds: float) -> float:
    power = _finite_number(power_kw, label="power_kw")
    duration = _finite_number(duration_seconds, label="duration_seconds")
    if power < 0.0:
        raise ValidationError(f"power_kw must be nonnegative; got {power!r}.")
    if duration < 0.0:
        raise ValidationError(
            f"duration_seconds must be nonnegative; got {duration!r}."
        )
    return _energy_kwh_from_kw_seconds(power, duration)


def mcs_per_port_power_kw(
    power_kw: float, port_count: int, power_is_unit_total: bool
) -> float:
    """Return port power in kW, dividing total unit power equally when needed."""
    power = _finite_number(power_kw, label="MCS power_kw")
    if power <= 0.0:
        raise ValidationError(f"MCS power_kw must be positive; got {power!r}.")
    if (
        isinstance(port_count, bool)
        or not isinstance(port_count, int)
        or port_count <= 0
    ):
        raise ValidationError("MCS port_count must be a positive integer.")
    if not isinstance(power_is_unit_total, bool):
        raise ValidationError("MCS power_is_unit_total must be boolean.")
    return power / port_count if power_is_unit_total else power


def mcs_unit_total_power_kw(
    power_kw: float, port_count: int, power_is_unit_total: bool
) -> float:
    """Return total MCS power in kW across all ports."""
    per_port = mcs_per_port_power_kw(power_kw, port_count, power_is_unit_total)
    return per_port * port_count


def validate_unit_system(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError("unit_system must be a mapping.")
    if dict(value) != dict(CANONICAL_UNIT_SYSTEM):
        raise ValidationError(
            f"unit_system must equal {dict(CANONICAL_UNIT_SYSTEM)!r}; got {dict(value)!r}."
        )
    return CANONICAL_UNIT_SYSTEM
