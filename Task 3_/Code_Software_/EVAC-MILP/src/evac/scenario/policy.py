"""Combine scenario policy overlays."""

from __future__ import annotations
import math
from typing import Any, Mapping
from evac.errors import ValidationError
from evac.scenario.authored import AuthoredScenarioResources

_NON_SEMANTIC_FIELDS = {
    "source",
    "authorship",
    "purpose",
    "status",
    "semantic_blockers",
}


def merged_semantic_overlay(resources: AuthoredScenarioResources) -> dict[str, Any]:
    """Compose ordered overlays, with later leaf values replacing earlier ones."""
    merged: dict[str, Any] = {}
    for overlay in resources.overlays:
        for key, value in overlay.values.items():
            if key in _NON_SEMANTIC_FIELDS:
                continue
            if (
                key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = _merge_mapping(merged[key], value)
            else:
                merged[key] = value
    return merged


def effective_mcs_site_limits(
    resources: AuthoredScenarioResources, overlay: Mapping[str, Any]
) -> dict[str, int]:
    """Resolve map defaults plus local overlay overrides into one effective mapping."""
    limits = {str(node.id.value): node.mcs_limit for node in resources.map.nodes}
    raw_overrides = overlay.get("mcs_site_limits")
    if raw_overrides is None:
        return limits
    if not isinstance(raw_overrides, Mapping) or any(
        (not isinstance(key, str) for key in raw_overrides)
    ):
        raise ValidationError("mcs_site_limits must be a string-keyed mapping.")
    for key, raw_limit in raw_overrides.items():
        if key not in limits:
            raise ValidationError(f"MCS limit references unknown node {key!r}.")
        limits[key] = _nonnegative_integer(raw_limit, label=f"MCS limit {key}")
    return limits


def normalized_semantic_overlay(resources: AuthoredScenarioResources) -> dict[str, Any]:
    """Return only effective, result-affecting overlay values for semantic identity."""
    merged = merged_semantic_overlay(resources)
    effective_limits = effective_mcs_site_limits(resources, merged)
    merged["mcs_site_limits"] = {
        key: value for key, value in effective_limits.items() if value > 0
    }
    return merged


def _merge_mapping(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be exactly integral.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise ValidationError(f"{label} must be exactly integral.")
    result = int(value)
    if result < 0:
        raise ValidationError("MCS limits must be nonnegative.")
    return result


__all__ = [
    "effective_mcs_site_limits",
    "merged_semantic_overlay",
    "normalized_semantic_overlay",
]
