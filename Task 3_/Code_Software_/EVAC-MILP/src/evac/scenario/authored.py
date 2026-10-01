"""Typed input resources for a scenario."""

from __future__ import annotations
import math
import numbers
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from evac.domain.identifiers import (
    AssetTypeId,
    CohortId,
    EdgeId,
    MCSUnitId,
    MapId,
    NodeId,
)
from evac.errors import ValidationError
from evac.physics import TrafficSpec


def finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite; got {result!r}.")
    return result


def positive(value: object, *, label: str) -> float:
    result = finite(value, label=label)
    if result <= 0.0:
        raise ValidationError(f"{label} must be positive; got {result!r}.")
    return result


def nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a nonnegative integer; got {value!r}.")
    return value


class MapDirectionality(str, Enum):
    BIDIRECTIONAL_PHYSICAL_LINKS = "bidirectional_physical_links"
    DIRECTED_EDGE_RECORDS = "directed_edge_records"
    NOT_ESTABLISHED = "not_established"


@dataclass(frozen=True, slots=True)
class AuthoredMapNode:
    id: NodeId
    coordinate: tuple[float, float]
    fixed_charger_ports: int = 0
    fixed_charger_power_kw: float | None = None
    mcs_limit: int = 0
    charger_type_label: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if len(self.coordinate) != 2:
            raise ValidationError("Map node coordinates require exactly two values.")
        object.__setattr__(
            self,
            "coordinate",
            tuple(
                (finite(item, label="map node coordinate") for item in self.coordinate)
            ),
        )
        nonnegative_integer(self.fixed_charger_ports, label="fixed charger ports")
        nonnegative_integer(self.mcs_limit, label="MCS limit")
        if self.fixed_charger_ports == 0:
            if self.fixed_charger_power_kw is not None:
                raise ValidationError(
                    "A node without fixed-charger ports cannot declare power."
                )
        elif self.fixed_charger_power_kw is None:
            raise ValidationError(
                "A node with fixed-charger ports must declare provider power."
            )
        else:
            object.__setattr__(
                self,
                "fixed_charger_power_kw",
                positive(self.fixed_charger_power_kw, label="fixed charger power_kw"),
            )
        for field_name in ("charger_type_label", "notes"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValidationError(
                    f"Map node {field_name} must be a non-empty string when set."
                )


@dataclass(frozen=True, slots=True)
class AuthoredRoadRecord:
    id: EdgeId
    source: NodeId
    target: NodeId
    distance_m: float
    free_flow_duration_seconds: float | None
    source_speed_m_per_second: float | None = None

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValidationError(f"Road {self.id} cannot be a self-loop.")
        object.__setattr__(
            self, "distance_m", positive(self.distance_m, label="road distance_m")
        )
        if self.free_flow_duration_seconds is not None:
            object.__setattr__(
                self,
                "free_flow_duration_seconds",
                positive(
                    self.free_flow_duration_seconds,
                    label="road free_flow_duration_seconds",
                ),
            )
        if self.source_speed_m_per_second is not None:
            object.__setattr__(
                self,
                "source_speed_m_per_second",
                positive(self.source_speed_m_per_second, label="road source speed"),
            )
        if (
            self.free_flow_duration_seconds is None
            and self.source_speed_m_per_second is None
        ):
            raise ValidationError(
                "A road requires free-flow duration or a positive authored source speed."
            )


@dataclass(frozen=True, slots=True)
class AuthoredMap:
    id: MapId
    display_name: str
    coordinate_system: str
    coordinate_units: str
    directionality: MapDirectionality
    nodes: tuple[AuthoredMapNode, ...]
    roads: tuple[AuthoredRoadRecord, ...]

    def __post_init__(self) -> None:
        if (
            not self.display_name
            or not self.coordinate_system
            or (not self.coordinate_units)
        ):
            raise ValidationError(
                "Map display name and coordinate declarations must be non-empty."
            )
        if not self.nodes or not self.roads:
            raise ValidationError("An authored Map requires nodes and road records.")
        node_keys = {(type(node.id.value), node.id.value) for node in self.nodes}
        if len(node_keys) != len(self.nodes):
            raise ValidationError("Authored Map contains duplicate node IDs.")
        road_keys = {(type(road.id.value), road.id.value) for road in self.roads}
        if len(road_keys) != len(self.roads):
            raise ValidationError("Authored Map contains duplicate road IDs.")
        for road in self.roads:
            if (type(road.source.value), road.source.value) not in node_keys:
                raise ValidationError(
                    f"Road {road.id} references unknown source {road.source}."
                )
            if (type(road.target.value), road.target.value) not in node_keys:
                raise ValidationError(
                    f"Road {road.id} references unknown target {road.target}."
                )

    @property
    def executable(self) -> bool:
        return (
            self.coordinate_system != "not_established"
            and self.coordinate_units != "not_established"
            and (self.directionality is not MapDirectionality.NOT_ESTABLISHED)
        )


@dataclass(frozen=True, slots=True)
class VehicleType:
    id: AssetTypeId
    battery_kwh: float
    efficiency_m_per_kwh: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "battery_kwh", positive(self.battery_kwh, label="battery_kwh")
        )
        object.__setattr__(
            self,
            "efficiency_m_per_kwh",
            positive(self.efficiency_m_per_kwh, label="efficiency_m_per_kwh"),
        )


@dataclass(frozen=True, slots=True)
class VehicleCatalog:
    types: tuple[VehicleType, ...]

    def __post_init__(self) -> None:
        if not self.types:
            raise ValidationError("Vehicle catalog requires at least one type.")
        keys = {(type(item.id.value), item.id.value) for item in self.types}
        if len(keys) != len(self.types):
            raise ValidationError("Vehicle catalog contains duplicate type IDs.")

    def by_id(self) -> Mapping[AssetTypeId, VehicleType]:
        return MappingProxyType({item.id: item for item in self.types})


@dataclass(frozen=True, slots=True)
class MobileChargerType:
    id: AssetTypeId
    battery_kwh: float
    power_kw: float
    port_count: int
    power_is_unit_total: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "battery_kwh", positive(self.battery_kwh, label="MCS battery_kwh")
        )
        object.__setattr__(
            self, "power_kw", positive(self.power_kw, label="MCS power_kw")
        )
        if (
            isinstance(self.port_count, bool)
            or not isinstance(self.port_count, int)
            or self.port_count <= 0
        ):
            raise ValidationError("MCS port_count must be a positive integer.")
        if not isinstance(self.power_is_unit_total, bool):
            raise ValidationError("MCS power_is_unit_total must be boolean.")


@dataclass(frozen=True, slots=True)
class MobileChargerCatalog:
    types: tuple[MobileChargerType, ...]

    def __post_init__(self) -> None:
        if not self.types:
            raise ValidationError("Mobile-charger catalog requires at least one type.")
        keys = {(type(item.id.value), item.id.value) for item in self.types}
        if len(keys) != len(self.types):
            raise ValidationError("Mobile-charger catalog contains duplicate type IDs.")


@dataclass(frozen=True, slots=True)
class AuthoredDemandCohort:
    id: CohortId
    vehicle_type_id: AssetTypeId
    speed_m_per_second: float
    initial_soc: float
    origin: NodeId
    destination: NodeId
    size: int
    source_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "speed_m_per_second",
            positive(self.speed_m_per_second, label="demand speed_m_per_second"),
        )
        soc = finite(self.initial_soc, label="demand initial_soc")
        if not 0.0 <= soc <= 1.0:
            raise ValidationError("Demand initial_soc must be within [0, 1].")
        object.__setattr__(self, "initial_soc", soc)
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise ValidationError("Demand size must be a positive integer.")
        if self.origin == self.destination:
            raise ValidationError("Demand origin and destination must differ.")


@dataclass(frozen=True, slots=True)
class AuthoredSupplyUnit:
    id: MCSUnitId
    mcs_type_id: AssetTypeId
    initial_soc: float

    def __post_init__(self) -> None:
        soc = finite(self.initial_soc, label="supply initial_soc")
        if not 0.0 <= soc <= 1.0:
            raise ValidationError("Supply initial_soc must be within [0, 1].")
        object.__setattr__(self, "initial_soc", soc)


@dataclass(frozen=True, slots=True)
class AuthoredObjective:
    name: str
    aggregation: str | None = None
    norm: str | None = None
    unit: str | None = None
    weight: float | None = None
    priority: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Authored objective name must be non-empty.")
        if self.weight is not None:
            object.__setattr__(
                self, "weight", finite(self.weight, label="objective weight")
            )
        if self.priority is not None and (
            isinstance(self.priority, bool) or not isinstance(self.priority, int)
        ):
            raise ValidationError("Objective priority must be an integer.")


@dataclass(frozen=True, slots=True)
class EvaluationDeclaration:
    objectives: tuple[AuthoredObjective, ...]

    def __post_init__(self):
        if len(self.objectives) != 1:
            raise ValidationError(
                "Evaluation declaration requires one primary objective."
            )


@dataclass(frozen=True, slots=True)
class ScenarioOverlay:
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class AuthoredScenarioResources:
    map: AuthoredMap
    vehicles: VehicleCatalog
    mobile_chargers: MobileChargerCatalog
    demand: tuple[AuthoredDemandCohort, ...]
    supply: tuple[AuthoredSupplyUnit, ...]
    traffic: TrafficSpec
    evaluation: EvaluationDeclaration
    overlays: tuple[ScenarioOverlay, ...]
