"""Typed identifiers for scenario and planning objects."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar
from evac.errors import ValidationError

IdentifierValue = str | int


@dataclass(frozen=True, slots=True)
class DomainIdentifier:
    """A namespace-preserving identifier.

    String and integer values are intentionally not coerced. Their Python type
    is part of the serialized identifier contract.
    """

    value: IdentifierValue
    namespace: ClassVar[str] = "domain"

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (str, int)):
            raise ValidationError(
                f"{type(self).__name__}.value must be a string or integer; got {self.value!r}."
            )
        if isinstance(self.value, str) and (not self.value):
            raise ValidationError(f"{type(self).__name__}.value must not be empty.")

    def to_data(self) -> dict[str, IdentifierValue]:
        return {"namespace": self.namespace, "value": self.value}

    def __str__(self) -> str:
        return str(self.value)


class MapId(DomainIdentifier):
    namespace = "map"


class NodeId(DomainIdentifier):
    namespace = "node"


class EdgeId(DomainIdentifier):
    namespace = "edge"


class AssetTypeId(DomainIdentifier):
    namespace = "asset_type"


class VehicleId(DomainIdentifier):
    namespace = "vehicle"


class CohortId(DomainIdentifier):
    namespace = "cohort"


class PathId(DomainIdentifier):
    namespace = "path"


class ChargingActionId(DomainIdentifier):
    namespace = "charging_action"


class ChargingSiteId(DomainIdentifier):
    namespace = "charging_site"


class ChargingProviderId(DomainIdentifier):
    namespace = "charging_provider"


class MCSUnitId(DomainIdentifier):
    namespace = "mcs_unit"


class DeploymentId(DomainIdentifier):
    namespace = "deployment"


class ScenarioId(DomainIdentifier):
    namespace = "scenario"


_IDENTIFIER_TYPES: dict[str, type[DomainIdentifier]] = {
    cls.namespace: cls
    for cls in (
        MapId,
        NodeId,
        EdgeId,
        AssetTypeId,
        VehicleId,
        CohortId,
        PathId,
        ChargingActionId,
        ChargingSiteId,
        ChargingProviderId,
        MCSUnitId,
        DeploymentId,
        ScenarioId,
    )
}
IdentifierT = TypeVar("IdentifierT", bound=DomainIdentifier)


def identifier_from_data(
    value: Any, *, expected_type: type[IdentifierT] | None = None
) -> DomainIdentifier | IdentifierT:
    if not isinstance(value, dict):
        raise ValidationError("A serialized identifier must be a mapping.")
    if set(value) != {"namespace", "value"}:
        raise ValidationError(
            "A serialized identifier must contain exactly 'namespace' and 'value'."
        )
    namespace = value["namespace"]
    if not isinstance(namespace, str) or namespace not in _IDENTIFIER_TYPES:
        raise ValidationError(f"Unsupported identifier namespace: {namespace!r}.")
    identifier_type = _IDENTIFIER_TYPES[namespace]
    if expected_type is not None and identifier_type is not expected_type:
        raise ValidationError(
            f"Expected identifier namespace {expected_type.namespace!r}; got {namespace!r}."
        )
    return identifier_type(value["value"])


def require_distinct_values(
    identifiers: tuple[DomainIdentifier, ...], *, label: str
) -> None:
    """Reject duplicate typed identifiers without coercing their values."""
    seen: set[tuple[str, type[str] | type[int], IdentifierValue]] = set()
    for identifier in identifiers:
        key = (identifier.namespace, type(identifier.value), identifier.value)
        if key in seen:
            raise ValidationError(
                f"{label} contains duplicate identifier {identifier.to_data()!r}."
            )
        seen.add(key)
