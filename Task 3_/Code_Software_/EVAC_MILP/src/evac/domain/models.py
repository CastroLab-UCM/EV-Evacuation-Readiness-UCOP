from __future__ import annotations
import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from evac.domain.identifiers import (
    AssetTypeId,
    ChargingActionId,
    ChargingProviderId,
    ChargingSiteId,
    CohortId,
    DeploymentId,
    EdgeId,
    MCSUnitId,
    NodeId,
    PathId,
    ScenarioId,
    VehicleId,
)
from evac.errors import ValidationError
from evac.physics import CANONICAL_UNIT_SYSTEM, TrafficSpec


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{label} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite; got {result!r}.")
    return result


def _nonnegative(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        raise ValidationError(f"{label} must be nonnegative; got {result!r}.")
    return result


def _positive(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise ValidationError(f"{label} must be positive; got {result!r}.")
    return result


def _probability(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValidationError(f"{label} must be within [0, 1]; got {result!r}.")
    return result


def _freeze_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping.")
    if any((not isinstance(key, str) or not key for key in value)):
        raise ValidationError(f"{label} keys must be non-empty strings.")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class SemanticIdentity:
    scope: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, str) or not self.scope:
            raise ValidationError("SemanticIdentity.scope must be non-empty.")
        if not isinstance(self.value, str) or not self.value:
            raise ValidationError("SemanticIdentity.value must be non-empty.")


@dataclass(frozen=True, slots=True)
class ChargingProviderSpec:
    id: ChargingProviderId
    site_id: ChargingSiteId
    kind: str
    port_count: int
    power_kw: float
    efficiency: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValidationError("ChargingProviderSpec.kind must be non-empty.")
        if isinstance(self.port_count, bool) or not isinstance(self.port_count, int):
            raise ValidationError("ChargingProviderSpec.port_count must be an integer.")
        if self.port_count < 0:
            raise ValidationError(
                "ChargingProviderSpec.port_count must be nonnegative."
            )
        object.__setattr__(
            self, "power_kw", _positive(self.power_kw, label="provider power_kw")
        )
        efficiency = _probability(self.efficiency, label="provider efficiency")
        if efficiency == 0.0:
            raise ValidationError("provider efficiency must be greater than zero.")
        object.__setattr__(self, "efficiency", efficiency)


@dataclass(frozen=True, slots=True)
class PhysicalNode:
    id: NodeId
    coordinate: tuple[float, float]
    charging_providers: tuple[ChargingProviderSpec, ...] = ()
    mcs_limit: int = 0

    def __post_init__(self) -> None:
        if len(self.coordinate) != 2:
            raise ValidationError(
                "PhysicalNode.coordinate must contain exactly two values."
            )
        object.__setattr__(
            self,
            "coordinate",
            tuple((_finite(item, label="node coordinate") for item in self.coordinate)),
        )
        if isinstance(self.mcs_limit, bool) or not isinstance(
            self.mcs_limit, int
        ):
            raise ValidationError("PhysicalNode.mcs_limit must be an integer.")
        if self.mcs_limit < 0:
            raise ValidationError("PhysicalNode.mcs_limit must be nonnegative.")


@dataclass(frozen=True, slots=True)
class PhysicalEdge:
    id: EdgeId
    source: NodeId
    target: NodeId
    distance_m: float
    free_flow_duration_seconds: float
    speed_limit_m_per_second: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "distance_m", _positive(self.distance_m, label="edge distance_m")
        )
        object.__setattr__(
            self,
            "free_flow_duration_seconds",
            _positive(
                self.free_flow_duration_seconds, label="edge free_flow_duration_seconds"
            ),
        )
        for field_name in ("speed_limit_m_per_second",):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _positive(value, label=f"edge {field_name}")
                )


@dataclass(frozen=True, slots=True)
class PhysicalNetwork:
    nodes: tuple[PhysicalNode, ...]
    edges: tuple[PhysicalEdge, ...]
    coordinate_system: str
    coordinate_units: str

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValidationError("PhysicalNetwork requires at least one node.")
        if not isinstance(self.coordinate_system, str) or not self.coordinate_system:
            raise ValidationError(
                "PhysicalNetwork.coordinate_system must be non-empty."
            )
        if not isinstance(self.coordinate_units, str) or not self.coordinate_units:
            raise ValidationError("PhysicalNetwork.coordinate_units must be non-empty.")
        node_keys = {(type(node.id.value), node.id.value) for node in self.nodes}
        if len(node_keys) != len(self.nodes):
            raise ValidationError(
                "PhysicalNetwork contains duplicate Node identifiers."
            )
        edge_keys = {(type(edge.id.value), edge.id.value) for edge in self.edges}
        if len(edge_keys) != len(self.edges):
            raise ValidationError(
                "PhysicalNetwork contains duplicate Edge identifiers."
            )
        for edge in self.edges:
            if (type(edge.source.value), edge.source.value) not in node_keys:
                raise ValidationError(
                    f"Edge {edge.id} references unknown source {edge.source}."
                )
            if (type(edge.target.value), edge.target.value) not in node_keys:
                raise ValidationError(
                    f"Edge {edge.id} references unknown target {edge.target}."
                )


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    id: VehicleId
    cohort_id: CohortId
    vehicle_type_id: AssetTypeId
    origin: NodeId
    destination: NodeId
    speed_m_per_second: float
    battery_kwh: float
    efficiency_m_per_kwh: float
    initial_soc: float
    minimum_soc: float
    maximum_soc: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "speed_m_per_second",
            _positive(self.speed_m_per_second, label="vehicle speed_m_per_second"),
        )
        object.__setattr__(
            self,
            "battery_kwh",
            _positive(self.battery_kwh, label="vehicle battery_kwh"),
        )
        object.__setattr__(
            self,
            "efficiency_m_per_kwh",
            _positive(self.efficiency_m_per_kwh, label="vehicle efficiency_m_per_kwh"),
        )
        object.__setattr__(
            self, "initial_soc", _probability(self.initial_soc, label="vehicle SOC")
        )
        minimum = _probability(self.minimum_soc, label="vehicle minimum SOC")
        maximum = _probability(self.maximum_soc, label="vehicle maximum SOC")
        if minimum > maximum:
            raise ValidationError("vehicle minimum SOC must not exceed maximum SOC.")
        if not minimum <= self.initial_soc <= maximum:
            raise ValidationError(
                "vehicle initial SOC must be inside its declared SOC bounds."
            )
        object.__setattr__(self, "minimum_soc", minimum)
        object.__setattr__(self, "maximum_soc", maximum)


@dataclass(frozen=True, slots=True)
class MCSUnitSpec:
    id: MCSUnitId
    asset_type_id: AssetTypeId
    battery_kwh: float
    power_kw: float
    port_count: int
    initial_soc: float
    minimum_soc: float
    maximum_soc: float
    charging_efficiency: float
    discharge_efficiency: float
    power_is_unit_total: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "battery_kwh", _positive(self.battery_kwh, label="MCS battery_kwh")
        )
        object.__setattr__(
            self, "power_kw", _positive(self.power_kw, label="MCS power_kw")
        )
        if isinstance(self.port_count, bool) or not isinstance(self.port_count, int):
            raise ValidationError("MCS port_count must be an integer.")
        if self.port_count <= 0:
            raise ValidationError("MCS port_count must be positive.")
        object.__setattr__(
            self, "initial_soc", _probability(self.initial_soc, label="MCS SOC")
        )
        minimum = _probability(self.minimum_soc, label="MCS minimum SOC")
        maximum = _probability(self.maximum_soc, label="MCS maximum SOC")
        if minimum > maximum:
            raise ValidationError("MCS minimum SOC must not exceed maximum SOC.")
        if not minimum <= self.initial_soc <= maximum:
            raise ValidationError(
                "MCS initial SOC must be inside its declared SOC bounds."
            )
        charging = _probability(
            self.charging_efficiency, label="MCS charging efficiency"
        )
        discharge = _probability(
            self.discharge_efficiency, label="MCS discharge efficiency"
        )
        if charging == 0.0:
            raise ValidationError("MCS charging efficiency must be greater than zero.")
        if discharge == 0.0:
            raise ValidationError("MCS discharge efficiency must be greater than zero.")
        if not isinstance(self.power_is_unit_total, bool):
            raise ValidationError("MCS power_is_unit_total must be boolean.")
        object.__setattr__(self, "minimum_soc", minimum)
        object.__setattr__(self, "maximum_soc", maximum)
        object.__setattr__(self, "charging_efficiency", charging)
        object.__setattr__(self, "discharge_efficiency", discharge)


@dataclass(frozen=True, slots=True)
class DepartureWindow:
    index: int
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValidationError(
                "DepartureWindow.index must be a nonnegative integer."
            )
        start = _nonnegative(self.start_seconds, label="departure window start")
        end = _positive(self.end_seconds, label="departure window end")
        if end <= start:
            raise ValidationError(
                "DepartureWindow.end_seconds must be after start_seconds."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class ChargingAction:
    id: ChargingActionId
    site_id: ChargingSiteId
    requested_energy_kwh: float
    eligible_provider_kinds: tuple[str, ...]
    required_post_charge_soc: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_energy_kwh",
            _positive(self.requested_energy_kwh, label="charging requested_energy_kwh"),
        )
        if not self.eligible_provider_kinds or any(
            (
                not isinstance(item, str) or not item
                for item in self.eligible_provider_kinds
            )
        ):
            raise ValidationError("ChargingAction requires non-empty provider kinds.")
        if len(set(self.eligible_provider_kinds)) != len(self.eligible_provider_kinds):
            raise ValidationError("ChargingAction provider kinds must be unique.")
        object.__setattr__(
            self,
            "required_post_charge_soc",
            _probability(
                self.required_post_charge_soc, label="required post-charge SOC"
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidatePath:
    id: PathId
    origin: NodeId
    destination: NodeId
    edge_ids: tuple[EdgeId, ...]
    charging_actions: tuple[ChargingAction, ...] = ()

    def __post_init__(self) -> None:
        if not self.edge_ids and self.origin != self.destination:
            raise ValidationError(
                "A nontrivial CandidatePath requires at least one edge."
            )


@dataclass(frozen=True, slots=True)
class DemandGroup:
    id: CohortId
    member_ids: tuple[VehicleId, ...]
    representative_id: VehicleId
    candidate_path_ids: tuple[PathId, ...]

    def __post_init__(self) -> None:
        if not self.member_ids:
            raise ValidationError("DemandGroup requires at least one vehicle.")
        if self.representative_id not in self.member_ids:
            raise ValidationError("DemandGroup representative must be a member.")
        if not self.candidate_path_ids:
            raise ValidationError("DemandGroup requires at least one certified path.")


@dataclass(frozen=True, slots=True)
class MCSDeploymentDomain:
    eligible_sites: tuple[ChargingSiteId, ...]
    site_limits: tuple[tuple[ChargingSiteId, int], ...]

    def __post_init__(self) -> None:
        eligible = set(self.eligible_sites)
        if len(eligible) != len(self.eligible_sites):
            raise ValidationError(
                "MCSDeploymentDomain contains duplicate eligible sites."
            )
        limit_sites = [site for site, _ in self.site_limits]
        if len(set(limit_sites)) != len(limit_sites):
            raise ValidationError(
                "MCSDeploymentDomain contains duplicate site limits."
            )
        if set(limit_sites) != eligible:
            raise ValidationError(
                "MCSDeploymentDomain requires exactly one limit per eligible site."
            )
        for site, limit in self.site_limits:
            if site not in eligible:
                raise ValidationError(
                    "MCS deployment limit references an ineligible site."
                )
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValidationError(
                    "MCS deployment limits must be nonnegative integers."
                )


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    name: str
    norm: str
    weight: float
    priority: int

    def __post_init__(self) -> None:
        if self.name != "completion_time":
            raise ValidationError(
                "The canonical objective name must be 'completion_time'."
            )
        if self.norm not in {"mean", "max"}:
            raise ValidationError("completion_time reducer must be 'mean' or 'max'.")
        weight = _finite(self.weight, label="objective weight")
        if weight != 1.0:
            raise ValidationError(
                "The primary completion_time objective must be unweighted."
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValidationError("ObjectiveSpec.priority must be an integer.")
        if self.priority != 1:
            raise ValidationError(
                "The primary completion_time objective must have priority 1."
            )
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    objectives: tuple[ObjectiveSpec, ...]

    def __post_init__(self) -> None:
        if len(self.objectives) != 1 or not isinstance(
            self.objectives[0], ObjectiveSpec
        ):
            raise ValidationError(
                "EvaluationSpec requires exactly one canonical completion_time objective."
            )


@dataclass(frozen=True, slots=True)
class PreparationPolicy:
    profile_id: str
    grouping_seed: int
    requested_cluster_count: int
    geographical_path_limit: int
    charging_path_limit: int
    charging_stop_limit: int
    charging_increment_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValidationError("PreparationPolicy.profile_id must be non-empty.")
        for name in (
            "grouping_seed",
            "requested_cluster_count",
            "geographical_path_limit",
            "charging_path_limit",
            "charging_stop_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"PreparationPolicy.{name} must be an integer.")
        if self.grouping_seed < 0:
            raise ValidationError(
                "PreparationPolicy.grouping_seed must be nonnegative."
            )
        if self.requested_cluster_count <= 0:
            raise ValidationError(
                "PreparationPolicy.requested_cluster_count must be positive."
            )
        if self.geographical_path_limit <= 0 or self.charging_path_limit <= 0:
            raise ValidationError("PreparationPolicy path limits must be positive.")
        if self.charging_stop_limit < 0:
            raise ValidationError(
                "PreparationPolicy.charging_stop_limit must be nonnegative."
            )
        object.__setattr__(
            self,
            "charging_increment_seconds",
            _positive(
                self.charging_increment_seconds, label="charging increment seconds"
            ),
        )


@dataclass(frozen=True, slots=True)
class NumericalPolicy:
    event_time_tolerance_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_time_tolerance_seconds",
            _nonnegative(
                self.event_time_tolerance_seconds, label="event_time_tolerance_seconds"
            ),
        )


class ChargingQueuePolicy(str, Enum):
    FORBIDDEN = "forbidden"
    FIFO = "fifo"


class ProviderDispatchPolicy(str, Enum):
    STABLE_PROVIDER_ID_V1 = "stable_provider_id_v1"


class ProviderBindingMode(str, Enum):
    PROVIDER_KIND = "provider_kind"
    PROVIDER_ID = "provider_id"


@dataclass(frozen=True, slots=True)
class ChargingPolicy:
    queueing: ChargingQueuePolicy
    provider_dispatch: ProviderDispatchPolicy
    queue_capacity_vehicles: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.queueing, ChargingQueuePolicy):
            raise ValidationError("ChargingPolicy.queueing must be typed.")
        if not isinstance(self.provider_dispatch, ProviderDispatchPolicy):
            raise ValidationError("ChargingPolicy.provider_dispatch must be typed.")
        capacity = self.queue_capacity_vehicles
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
        ):
            raise ValidationError(
                "ChargingPolicy.queue_capacity_vehicles must be a nonnegative integer or None."
            )
        if self.queueing is ChargingQueuePolicy.FORBIDDEN and capacity is not None:
            raise ValidationError(
                "A forbidden charging queue cannot declare waiting capacity."
            )


@dataclass(frozen=True, slots=True)
class PreparedScenario:
    scenario_id: ScenarioId
    identity: SemanticIdentity
    unit_system: Mapping[str, str]
    network: PhysicalNetwork
    vehicles: tuple[VehicleSpec, ...]
    mcs_units: tuple[MCSUnitSpec, ...]
    departure_windows: tuple[DepartureWindow, ...]
    mcs_deployment_domain: MCSDeploymentDomain
    demand_groups: tuple[DemandGroup, ...]
    candidate_paths: tuple[CandidatePath, ...]
    traffic: TrafficSpec
    evaluation: EvaluationSpec
    preparation_policy: PreparationPolicy
    numerical_policy: NumericalPolicy
    charging_policy: ChargingPolicy
    planning_horizon_seconds: float

    def __post_init__(self) -> None:
        if dict(self.unit_system) != dict(CANONICAL_UNIT_SYSTEM):
            raise ValidationError(
                "PreparedScenario must use the canonical unit system."
            )
        object.__setattr__(
            self, "unit_system", MappingProxyType(dict(self.unit_system))
        )
        vehicle_ids = {(type(item.id.value), item.id.value) for item in self.vehicles}
        if len(vehicle_ids) != len(self.vehicles):
            raise ValidationError("PreparedScenario contains duplicate Vehicle IDs.")
        path_ids = {
            (type(item.id.value), item.id.value) for item in self.candidate_paths
        }
        if len(path_ids) != len(self.candidate_paths):
            raise ValidationError("PreparedScenario contains duplicate Path IDs.")
        if not self.vehicles:
            raise ValidationError("PreparedScenario requires at least one vehicle.")
        if not self.departure_windows:
            raise ValidationError(
                "PreparedScenario requires at least one departure window."
            )
        for expected, label in (
            (self.preparation_policy, PreparationPolicy),
            (self.numerical_policy, NumericalPolicy),
            (self.charging_policy, ChargingPolicy),
        ):
            if not isinstance(expected, label):
                raise ValidationError(
                    f"PreparedScenario requires a typed {label.__name__}."
                )
        horizon = _finite(self.planning_horizon_seconds, label="planning_horizon_seconds")
        if horizon <= 0:
            raise ValidationError("planning_horizon_seconds must be positive.")
        object.__setattr__(self, "planning_horizon_seconds", horizon)
        _validate_prepared_scenario_references(self)


def _validate_prepared_scenario_references(prepared: PreparedScenario) -> None:
    nodes = {item.id: item for item in prepared.network.nodes}
    edges = {item.id: item for item in prepared.network.edges}
    vehicles = {item.id: item for item in prepared.vehicles}
    paths = {item.id: item for item in prepared.candidate_paths}
    mcs_ids = {item.id for item in prepared.mcs_units}
    if len(mcs_ids) != len(prepared.mcs_units):
        raise ValidationError("PreparedScenario contains duplicate MCS unit IDs.")
    provider_ids: set[ChargingProviderId] = set()
    fixed_kinds_by_site: dict[ChargingSiteId, set[str]] = {}
    for node in prepared.network.nodes:
        site_id = ChargingSiteId(node.id.value)
        fixed_kinds_by_site[site_id] = set()
        for provider in node.charging_providers:
            if provider.id in provider_ids:
                raise ValidationError(
                    "PreparedScenario contains duplicate charging provider IDs."
                )
            provider_ids.add(provider.id)
            if provider.site_id != site_id:
                raise ValidationError(
                    f"Charging provider {provider.id.value!r} is attached to node {node.id.value!r} but declares site {provider.site_id.value!r}."
                )
            if provider.port_count > 0:
                fixed_kinds_by_site[site_id].add(provider.kind)
    for vehicle in prepared.vehicles:
        if vehicle.origin not in nodes:
            raise ValidationError(
                f"Vehicle {vehicle.id.value!r} references unknown origin {vehicle.origin.value!r}."
            )
        if vehicle.destination not in nodes:
            raise ValidationError(
                f"Vehicle {vehicle.id.value!r} references unknown destination {vehicle.destination.value!r}."
            )
    eligible_sites = set(prepared.mcs_deployment_domain.eligible_sites)
    unknown_sites = eligible_sites - set(fixed_kinds_by_site)
    if unknown_sites:
        values = sorted((repr(item.value) for item in unknown_sites))
        raise ValidationError(
            f"MCS deployment domain references unknown sites: {values!r}."
        )
    effective_node_limits = {
        ChargingSiteId(node.id.value): node.mcs_limit
        for node in prepared.network.nodes
        if node.mcs_limit > 0
    }
    if dict(prepared.mcs_deployment_domain.site_limits) != effective_node_limits:
        raise ValidationError(
            "PreparedScenario network and MCSDeploymentDomain must expose the same effective MCS limits."
        )
    action_semantics: dict[ChargingActionId, tuple[object, ...]] = {}
    for path in prepared.candidate_paths:
        if path.origin not in nodes or path.destination not in nodes:
            raise ValidationError(
                f"Candidate path {path.id.value!r} references an unknown endpoint."
            )
        try:
            path_edges = tuple((edges[edge_id] for edge_id in path.edge_ids))
        except KeyError as exc:
            raise ValidationError(
                f"Candidate path {path.id.value!r} references an unknown edge."
            ) from exc
        route_nodes = [path.origin]
        for edge in path_edges:
            if edge.source != route_nodes[-1]:
                raise ValidationError(
                    f"Candidate path {path.id.value!r} is not contiguous."
                )
            route_nodes.append(edge.target)
        if route_nodes[-1] != path.destination:
            raise ValidationError(
                f"Candidate path {path.id.value!r} has inconsistent endpoints."
            )
        chargeable_sites = {ChargingSiteId(item.value) for item in route_nodes[:-1]}
        action_sites: set[ChargingSiteId] = set()
        for action in path.charging_actions:
            if action.site_id not in chargeable_sites:
                raise ValidationError(
                    f"Charging action {action.id.value!r} is not located on candidate path {path.id.value!r}."
                )
            if action.site_id in action_sites:
                raise ValidationError(
                    f"Candidate path {path.id.value!r} contains repeated charging actions at site {action.site_id.value!r}."
                )
            action_sites.add(action.site_id)
            available_kinds = set(fixed_kinds_by_site[action.site_id])
            if action.site_id in eligible_sites and prepared.mcs_units:
                available_kinds.add("mcs")
            unavailable = set(action.eligible_provider_kinds) - available_kinds
            if unavailable:
                raise ValidationError(
                    f"Charging action {action.id.value!r} references unavailable provider kinds {sorted(unavailable)!r} at site {action.site_id.value!r}."
                )
            semantics = (
                action.site_id,
                action.requested_energy_kwh,
                action.eligible_provider_kinds,
                action.required_post_charge_soc,
            )
            previous = action_semantics.setdefault(action.id, semantics)
            if previous != semantics:
                raise ValidationError(
                    f"Charging action ID {action.id.value!r} has conflicting semantics."
                )
    group_ids = {item.id for item in prepared.demand_groups}
    if len(group_ids) != len(prepared.demand_groups):
        raise ValidationError("PreparedScenario contains duplicate demand group IDs.")
    group_for_vehicle: dict[VehicleId, DemandGroup] = {}
    for group in prepared.demand_groups:
        for vehicle_id in group.member_ids:
            if vehicle_id not in vehicles:
                raise ValidationError(
                    f"Demand group {group.id.value!r} references unknown vehicle {vehicle_id.value!r}."
                )
            if vehicle_id in group_for_vehicle:
                raise ValidationError("PreparedScenario demand groups overlap.")
            group_for_vehicle[vehicle_id] = group
        for path_id in group.candidate_path_ids:
            if path_id not in paths:
                raise ValidationError(
                    f"Demand group {group.id.value!r} references unknown candidate path {path_id.value!r}."
                )
            path = paths[path_id]
            for vehicle_id in group.member_ids:
                vehicle = vehicles[vehicle_id]
                if (path.origin, path.destination) != (
                    vehicle.origin,
                    vehicle.destination,
                ):
                    raise ValidationError(
                        f"Candidate path {path.id.value!r} does not match vehicle {vehicle.id.value!r} origin and destination."
                    )
    if set(group_for_vehicle) != set(vehicles):
        raise ValidationError(
            "PreparedScenario demand groups must cover every vehicle exactly once."
        )


@dataclass(frozen=True, slots=True)
class PlannedChargingAction:
    action_id: ChargingActionId
    provider_id: ChargingProviderId | None = None
    provider_kind: str | None = None
    binding_mode: ProviderBindingMode | None = None

    def __post_init__(self) -> None:
        binding_mode = self.binding_mode
        if binding_mode is None:
            if self.provider_id is not None and self.provider_kind is None:
                binding_mode = ProviderBindingMode.PROVIDER_ID
            elif self.provider_id is None and self.provider_kind is not None:
                binding_mode = ProviderBindingMode.PROVIDER_KIND
            else:
                raise ValidationError(
                    "PlannedChargingAction requires an explicit binding mode when its provider fields do not identify exactly one provider binding."
                )
            object.__setattr__(self, "binding_mode", binding_mode)
        if not isinstance(binding_mode, ProviderBindingMode):
            raise ValidationError("PlannedChargingAction.binding_mode must be typed.")
        expected_fields = {
            ProviderBindingMode.PROVIDER_KIND: (False, True),
            ProviderBindingMode.PROVIDER_ID: (True, False),
        }
        if (
            self.provider_id is not None,
            self.provider_kind is not None,
        ) != expected_fields[binding_mode]:
            raise ValidationError(
                "PlannedChargingAction provider fields do not match its binding mode."
            )
        if self.provider_kind is not None and (
            not isinstance(self.provider_kind, str) or not self.provider_kind
        ):
            raise ValidationError(
                "PlannedChargingAction.provider_kind must be a non-empty string when set."
            )


@dataclass(frozen=True, slots=True)
class VehiclePlan:
    vehicle_id: VehicleId
    path_id: PathId
    departure_seconds: float
    charging_actions: tuple[PlannedChargingAction, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "departure_seconds",
            _nonnegative(self.departure_seconds, label="vehicle departure_seconds"),
        )


@dataclass(frozen=True, slots=True)
class MCSDeployment:
    id: DeploymentId
    mcs_unit_id: MCSUnitId
    site_id: ChargingSiteId


@dataclass(frozen=True, slots=True)
class EvacuationPlan:
    identity: SemanticIdentity
    preparation_identity: SemanticIdentity
    vehicle_plans: tuple[VehiclePlan, ...]
    mcs_deployments: tuple[MCSDeployment, ...]

    def __post_init__(self) -> None:
        vehicle_ids = {
            (type(item.vehicle_id.value), item.vehicle_id.value)
            for item in self.vehicle_plans
        }
        if len(vehicle_ids) != len(self.vehicle_plans):
            raise ValidationError("EvacuationPlan contains duplicate vehicle plans.")
        deployment_ids = {
            (type(item.id.value), item.id.value) for item in self.mcs_deployments
        }
        if len(deployment_ids) != len(self.mcs_deployments):
            raise ValidationError("EvacuationPlan contains duplicate deployment IDs.")


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    time_seconds: float
    kind: str
    subject_namespace: str
    subject_value: str | int
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_seconds",
            _nonnegative(self.time_seconds, label="timeline time_seconds"),
        )
        if not isinstance(self.kind, str) or not self.kind:
            raise ValidationError("TimelineEvent.kind must be non-empty.")
        if not isinstance(self.subject_namespace, str) or not self.subject_namespace:
            raise ValidationError("TimelineEvent.subject_namespace must be non-empty.")
        if isinstance(self.subject_value, bool) or not isinstance(
            self.subject_value, (str, int)
        ):
            raise ValidationError(
                "TimelineEvent.subject_value must be a string or integer."
            )
        object.__setattr__(
            self, "values", _freeze_mapping(self.values, label="timeline values")
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    identity: SemanticIdentity
    preparation_identity: SemanticIdentity
    plan_identity: SemanticIdentity
    vehicle_events: tuple[TimelineEvent, ...]
    edge_events: tuple[TimelineEvent, ...]
    charging_site_events: tuple[TimelineEvent, ...]
    mcs_events: tuple[TimelineEvent, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    vehicle_terminal_reasons: tuple[tuple[VehicleId, str], ...] = ()
    terminal_reason: str = "complete"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, label="simulation metadata"),
        )
        if not isinstance(self.terminal_reason, str) or not self.terminal_reason:
            raise ValidationError("SimulationResult.terminal_reason must be non-empty.")
        seen: set[tuple[type[str] | type[int], str | int]] = set()
        for vehicle_id, reason in self.vehicle_terminal_reasons:
            key = (type(vehicle_id.value), vehicle_id.value)
            if key in seen or not isinstance(reason, str) or (not reason):
                raise ValidationError(
                    "SimulationResult terminal reasons must be unique and non-empty."
                )
            seen.add(key)


@dataclass(frozen=True, slots=True)
class EvaluationViolation:
    kind: str
    subject_namespace: str | None
    subject_value: str | int | None
    magnitude: float
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValidationError("EvaluationViolation.kind must be non-empty.")
        object.__setattr__(
            self, "magnitude", _nonnegative(self.magnitude, label="violation magnitude")
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    identity: SemanticIdentity
    simulation_identity: SemanticIdentity
    objective_name: str
    scientific_objective: float | None
    feasible: bool
    violations: tuple[EvaluationViolation, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective_name not in {"completion_time/mean", "completion_time/max"}:
            raise ValidationError("EvaluationResult objective_name must be canonical.")
        if self.scientific_objective is not None:
            object.__setattr__(
                self,
                "scientific_objective",
                _finite(self.scientific_objective, label="scientific_objective"),
            )
        if not isinstance(self.feasible, bool):
            raise ValidationError("EvaluationResult.feasible must be boolean.")
        object.__setattr__(
            self, "metrics", _freeze_mapping(self.metrics, label="evaluation metrics")
        )

    @property
    def objective_available(self) -> bool:
        return self.scientific_objective is not None


class PlannerTermination(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NO_INCUMBENT = "no_incumbent"


@dataclass(frozen=True, slots=True)
class PlannerResult:
    planner_kind: str
    model_identity: SemanticIdentity
    termination: PlannerTermination
    plan: EvacuationPlan | None
    objective_bound: float | None
    objective_gap: float | None
    terminal_cause: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    formulation_objective: float | None = None


@dataclass(frozen=True, slots=True)
class ArtifactLocations:
    root: Path | None
    items: Mapping[str, Path]

    def __post_init__(self) -> None:
        if self.root is not None:
            object.__setattr__(self, "root", Path(self.root).resolve())
        normalized: dict[str, Path] = {}
        for name, path in self.items.items():
            if not isinstance(name, str) or not name:
                raise ValidationError(
                    "Artifact location names must be non-empty strings."
                )
            normalized[name] = Path(path)
        object.__setattr__(self, "items", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class RunResult:
    planner: PlannerResult
    simulation: SimulationResult | None
    evaluation: EvaluationResult | None
    artifacts: ArtifactLocations = field(
        default_factory=lambda: ArtifactLocations(None, {})
    )
    timings: Mapping[str, float] = field(default_factory=dict)

    @property
    def plan(self) -> EvacuationPlan | None:
        return self.planner.plan
