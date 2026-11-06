from __future__ import annotations
from dataclasses import dataclass, replace
import heapq
from itertools import combinations, count, product
import json
import math
import random
from typing import Any, Mapping, Sequence
import networkx as nx
import numpy as np
from evac.domain import (
    CandidatePath,
    ChargingAction,
    ChargingActionId,
    ChargingPolicy,
    ChargingQueuePolicy,
    ChargingProviderId,
    ChargingProviderSpec,
    ChargingSiteId,
    CohortId,
    DemandGroup,
    DepartureWindow,
    EdgeId,
    EvaluationSpec,
    MCSUnitSpec,
    NodeId,
    NumericalPolicy,
    ObjectiveSpec,
    PathId,
    PhysicalEdge,
    PhysicalNetwork,
    PhysicalNode,
    PreparationPolicy,
    PreparedScenario,
    ProviderDispatchPolicy,
    SemanticIdentity,
    MCSDeploymentDomain,
    VehicleId,
    VehicleSpec,
)
from evac.errors import InvariantError, ValidationError
from evac.domain.ordering import identifier_key
from evac.physics import CANONICAL_UNIT_SYSTEM, mcs_per_port_power_kw
from evac.scenario.authored import AuthoredScenarioResources, MapDirectionality
from evac.scenario.codecs import load_scenario_resources
from evac.scenario.models import Scenario
from evac.scenario.policy import (
    effective_mcs_site_limits,
    merged_semantic_overlay,
)
from evac.scenario.snapshot import ResolvedScenarioSnapshot, resolve_scenario_snapshot


@dataclass(frozen=True, slots=True)
class PreparationInspection:
    blockers: tuple[str, ...]
    authored_node_count: int
    authored_road_count: int
    expanded_vehicle_count: int
    mobile_charger_count: int

    @property
    def ready(self) -> bool:
        return not self.blockers


def inspect_preparation(scenario: Scenario) -> PreparationInspection:
    if not isinstance(scenario, Scenario):
        raise ValidationError("inspect_preparation requires a typed Scenario.")
    resources = load_scenario_resources(scenario)
    blockers: list[str] = [] if resources.map.executable else ["MAP-1"]
    return PreparationInspection(
        blockers=tuple(blockers),
        authored_node_count=len(resources.map.nodes),
        authored_road_count=len(resources.map.roads),
        expanded_vehicle_count=sum((item.size for item in resources.demand)),
        mobile_charger_count=len(resources.supply),
    )


def prepare(
    scenario: Scenario,
    *,
    objective: str | None = None,
    mcs_count: int | None = None,
    initial_soc: float | None = None,
) -> PreparedScenario:
    """Prepare inputs, optionally choosing mean/max, fleet size, or uniform initial SOC.

    mcs_count selects units from the supplied inventory; it must be between zero
    and the inventory size. initial_soc is a fraction in [0, 1] applied to every
    EV cohort. Omitted settings retain the authored data.
    """
    snapshot = resolve_scenario_snapshot(scenario)
    resources = snapshot.resources
    if mcs_count is not None:
        if (
            isinstance(mcs_count, bool)
            or not isinstance(mcs_count, int)
            or (not 0 <= mcs_count <= len(resources.supply))
        ):
            raise ValidationError(
                f"mcs_count must be an integer between 0 and {len(resources.supply)}."
            )
        resources = replace(resources, supply=resources.supply[:mcs_count])
    if initial_soc is not None:
        resources = replace(
            resources,
            demand=tuple(
                (replace(item, initial_soc=initial_soc) for item in resources.demand)
            ),
        )
    from evac.scenario.identity import scenario_identity_from_resources

    snapshot = replace(
        snapshot,
        resources=resources,
        identity=scenario_identity_from_resources(scenario, resources),
    )
    return prepare_snapshot(snapshot, objective=objective)


def prepare_snapshot(
    snapshot: ResolvedScenarioSnapshot, *, objective: str | None = None
) -> PreparedScenario:
    if not isinstance(snapshot, ResolvedScenarioSnapshot):
        raise ValidationError("prepare_snapshot requires a ResolvedScenarioSnapshot.")
    scenario = snapshot.scenario
    resources = snapshot.resources
    if not resources.map.executable:
        raise ValidationError(
            f"Map {resources.map.id.value!r} is declared source-only and cannot be prepared."
        )
    overlay = merged_semantic_overlay(resources)
    preparation_policy = _preparation_policy(overlay)
    numerical_policy = _numerical_policy(overlay)
    charging_policy = _charging_policy(overlay)
    vehicle_bounds, mcs_bounds = _soc_bounds(overlay)
    charging_efficiency, discharge_efficiency = _charging_efficiencies(overlay)
    mcs_site_limits = effective_mcs_site_limits(resources, overlay)
    network = _compile_network(
        resources, overlay, charging_efficiency, mcs_site_limits
    )
    vehicles = _expand_vehicles(resources, vehicle_bounds)
    mcs_units = _compile_mcs_units(
        resources, mcs_bounds, charging_efficiency, discharge_efficiency
    )
    windows = _departure_windows(overlay)
    deployment_domain = _mcs_deployment_domain(network)
    evaluation = (
        _evaluation_spec(resources)
        if objective is None
        else EvaluationSpec((ObjectiveSpec("completion_time", objective, 1.0, 1),))
    )
    initial_groups = _initial_groups(vehicles, preparation_policy)
    od_paths: dict[tuple[NodeId, NodeId], tuple[CandidatePath, ...]] = {}
    for origin, destination in sorted(
        {(item.origin, item.destination) for item in vehicles},
        key=lambda pair: (identifier_key(pair[0]), identifier_key(pair[1])),
    ):
        od_members = tuple(
            (
                item
                for item in vehicles
                if item.origin == origin and item.destination == destination
            )
        )
        od_paths[origin, destination] = _candidate_paths(
            network, od_members, mcs_units, deployment_domain, preparation_policy
        )
    certified_groups: list[
        tuple[tuple[VehicleSpec, ...], tuple[CandidatePath, ...]]
    ] = []
    for members in initial_groups:
        paths = od_paths[members[0].origin, members[0].destination]
        _certify_or_split(network, members, paths, preparation_policy, certified_groups)
    retained_by_id: dict[PathId, CandidatePath] = {}
    for _, paths in certified_groups:
        for path in paths:
            retained_by_id[path.id] = path
    retained_paths = tuple(
        sorted(
            retained_by_id.values(), key=lambda item: _canonical_path_key(network, item)
        )
    )
    path_rank = {item.id: index for index, item in enumerate(retained_paths)}
    groups: list[DemandGroup] = []
    for index, (members, paths) in enumerate(
        sorted(
            certified_groups,
            key=lambda item: tuple((identifier_key(v.id) for v in item[0])),
        )
    ):
        representative = _representative(members)
        groups.append(
            DemandGroup(
                CohortId(
                    f"{members[0].origin.value}->{members[0].destination.value}:group-{index}"
                ),
                tuple((item.id for item in members)),
                representative.id,
                tuple(
                    (item.id for item in sorted(paths, key=lambda p: path_rank[p.id]))
                ),
            )
        )
    planning_horizon_seconds = _execution_horizon_seconds(
        network,
        vehicles,
        tuple(groups),
        retained_paths,
        windows,
        resources.traffic,
        mcs_units,
    )
    identity_payload = {
        "contract": "evac/domain-semantics",
        "scenario_identity": snapshot.identity.value,
        "network": _network_identity_data(network),
        "vehicles": [_vehicle_identity_data(item) for item in vehicles],
        "mcs_units": [_mcs_identity_data(item) for item in mcs_units],
        "windows": [
            (item.index, item.start_seconds, item.end_seconds) for item in windows
        ],
        "groups": [
            (
                item.id.to_data(),
                [member.to_data() for member in item.member_ids],
                item.representative_id.to_data(),
                [path.to_data() for path in item.candidate_path_ids],
            )
            for item in groups
        ],
        "paths": [_path_identity_data(item) for item in retained_paths],
        "traffic": {
            "exogenous_profile": resources.traffic.exogenous_profile.to_data()
        },
        "evaluation": {
            "objectives": [
                (item.name, item.norm, item.weight, item.priority)
                for item in evaluation.objectives
            ]
        },
        "preparation_policy": _slots_data(preparation_policy),
        "numerical_policy": _slots_data(numerical_policy),
        "charging_policy": _slots_data(charging_policy),
        "planning_horizon_seconds": planning_horizon_seconds,
    }
    identity = SemanticIdentity(
        "preparation",
        json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )
    return PreparedScenario(
        scenario.id,
        identity,
        CANONICAL_UNIT_SYSTEM,
        network,
        vehicles,
        mcs_units,
        windows,
        deployment_domain,
        tuple(groups),
        retained_paths,
        resources.traffic,
        evaluation,
        preparation_policy,
        numerical_policy,
        charging_policy,
        planning_horizon_seconds,
    )


def _slots_data(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__slots__}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        (not isinstance(key, str) for key in value)
    ):
        raise ValidationError(f"{label} must be a string-keyed mapping.")
    return value


def _exact_integer(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be exactly integral.")
    if not math.isfinite(float(value)) or float(value) != math.floor(float(value)):
        raise ValidationError(f"{label} must be exactly integral.")
    result = int(value)
    if positive and result <= 0:
        raise ValidationError(f"{label} must be positive.")
    return result


def _preparation_policy(overlay: Mapping[str, Any]) -> PreparationPolicy:
    grouping = _mapping(overlay.get("grouping"), label="grouping")
    routes = _mapping(overlay.get("route_selection"), label="route_selection")
    grouping_fields = {"requested_cluster_count", "seed"}
    route_fields = {
        "geographical_path_limit",
        "charging_path_limit",
        "charging_stop_limit",
        "charging_increment_seconds",
    }
    if set(grouping) != grouping_fields:
        raise ValidationError(f"grouping requires exactly {sorted(grouping_fields)!r}.")
    if set(routes) != route_fields:
        raise ValidationError(
            f"route_selection requires exactly {sorted(route_fields)!r}."
        )
    return PreparationPolicy(
        "behavioral_zscore_v1+yen_charging_v2",
        _exact_integer(grouping["seed"], label="grouping.seed"),
        _exact_integer(
            grouping["requested_cluster_count"],
            label="grouping.requested_cluster_count",
            positive=True,
        ),
        _exact_integer(
            routes["geographical_path_limit"],
            label="route_selection.geographical_path_limit",
            positive=True,
        ),
        _exact_integer(
            routes["charging_path_limit"],
            label="route_selection.charging_path_limit",
            positive=True,
        ),
        _exact_integer(
            routes["charging_stop_limit"], label="route_selection.charging_stop_limit"
        ),
        float(routes["charging_increment_seconds"]),
    )


def _numerical_policy(overlay: Mapping[str, Any]) -> NumericalPolicy:
    policy = _mapping(overlay.get("numerical_policy"), label="numerical_policy")
    if set(policy) != {"event_time_tolerance_seconds"}:
        raise ValidationError(
            "numerical_policy requires exactly event_time_tolerance_seconds."
        )
    return NumericalPolicy(policy["event_time_tolerance_seconds"])


def _charging_policy(overlay: Mapping[str, Any]) -> ChargingPolicy:
    values = _mapping(overlay.get("charging_contention"), label="charging_contention")
    required = {"queueing", "provider_dispatch"}
    if not required <= set(values) or set(values) - (
        required | {"queue_capacity_vehicles"}
    ):
        raise ValidationError(
            "charging_contention requires queueing and provider_dispatch and permits queue_capacity_vehicles."
        )
    try:
        queueing = ChargingQueuePolicy(values["queueing"])
        dispatch = ProviderDispatchPolicy(values["provider_dispatch"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("Unsupported charging-contention policy.") from exc
    capacity = values.get("queue_capacity_vehicles")
    if capacity is not None:
        capacity = _exact_integer(
            capacity, label="charging_contention.queue_capacity_vehicles"
        )
    return ChargingPolicy(queueing, dispatch, capacity)


def _soc_bounds(
    overlay: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    bounds = _mapping(overlay.get("soc_bounds"), label="soc_bounds")
    vehicle = _mapping(bounds.get("vehicle"), label="vehicle SOC bounds")
    mcs = _mapping(bounds.get("mobile_charger"), label="MCS SOC bounds")
    for value, label in ((vehicle, "vehicle"), (mcs, "mobile_charger")):
        if set(value) != {"min", "max"}:
            raise ValidationError(f"{label} SOC bounds require min and max.")
    return (
        (float(vehicle["min"]), float(vehicle["max"])),
        (float(mcs["min"]), float(mcs["max"])),
    )


def _charging_efficiencies(overlay: Mapping[str, Any]) -> tuple[float, float]:
    values = _mapping(overlay.get("charging_physics"), label="charging_physics")
    required = {"charging_efficiency", "mcs_discharge_efficiency"}
    if set(values) != required:
        raise ValidationError(
            f"charging_physics requires exactly {sorted(required)!r}."
        )
    charging = float(values["charging_efficiency"])
    discharge = float(values["mcs_discharge_efficiency"])
    if not 0.0 < charging <= 1.0 or not 0.0 < discharge <= 1.0:
        raise ValidationError("Charging efficiencies must be within (0, 1].")
    return (charging, discharge)


def _compile_network(
    resources: AuthoredScenarioResources,
    overlay: Mapping[str, Any],
    charging_efficiency: float,
    mcs_site_limits: Mapping[str, int],
) -> PhysicalNetwork:
    authored = resources.map
    if authored.coordinate_system == "EPSG:4326":
        if authored.coordinate_units != "degree":
            raise ValidationError("EPSG:4326 coordinates must use degrees.")
        for node in authored.nodes:
            longitude, latitude = node.coordinate
            if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
                raise ValidationError(
                    "EPSG:4326 coordinate lies outside longitude/latitude bounds."
                )
    elif authored.id.value == "trellis" and (
        authored.coordinate_system != "local_schematic"
        or authored.coordinate_units != "arbitrary"
    ):
        raise ValidationError("Trellis must use local_schematic/arbitrary coordinates.")
    elif authored.id.value == "mariposa" and (
        authored.coordinate_system != "source_local_cartesian"
        or authored.coordinate_units != "m"
    ):
        raise ValidationError("Mariposa must use source_local_cartesian/m coordinates.")
    overrides = _mapping(
        overlay.get("fixed_charger_site_overrides", {}),
        label="fixed_charger_site_overrides",
    )
    nodes: list[PhysicalNode] = []
    for node in sorted(authored.nodes, key=lambda item: identifier_key(item.id)):
        site = ChargingSiteId(node.id.value)
        ports = node.fixed_charger_ports
        if str(node.id.value) in overrides:
            item = _mapping(
                overrides[str(node.id.value)], label="fixed charger override"
            )
            if set(item) != {"ports"}:
                raise ValidationError("A fixed-charger override may set only ports.")
            ports = _exact_integer(item["ports"], label="fixed charger ports")
        providers: tuple[ChargingProviderSpec, ...] = ()
        if ports:
            providers = (
                ChargingProviderSpec(
                    ChargingProviderId(f"fcs:{node.id.value}"),
                    site,
                    "fcs",
                    ports,
                    node.fixed_charger_power_kw,
                    charging_efficiency,
                ),
            )
        nodes.append(
            PhysicalNode(
                node.id,
                node.coordinate,
                providers,
                mcs_site_limits[str(node.id.value)],
            )
        )
    edges: list[PhysicalEdge] = []
    for road in sorted(authored.roads, key=lambda item: identifier_key(item.id)):
        duration = road.free_flow_duration_seconds
        if duration is None:
            if road.source_speed_m_per_second is None:
                raise InvariantError(
                    "A road has no free-flow duration or source speed."
                )
            duration = road.distance_m / road.source_speed_m_per_second
        implied_speed = road.distance_m / duration
        if (
            road.source_speed_m_per_second is not None
            and implied_speed != road.source_speed_m_per_second
        ):
            raise ValidationError(
                f"Road {road.id.value!r} has conflicting authored duration and speed."
            )
        directions: tuple[tuple[str, NodeId, NodeId], ...]
        if authored.directionality is MapDirectionality.BIDIRECTIONAL_PHYSICAL_LINKS:
            directions = (
                ("forward", road.source, road.target),
                ("reverse", road.target, road.source),
            )
        elif authored.directionality is MapDirectionality.DIRECTED_EDGE_RECORDS:
            directions = (("directed", road.source, road.target),)
        else:
            raise ValidationError("Map directionality is not executable.")
        for discriminator, source, target in directions:
            edge_id = (
                road.id
                if discriminator == "directed"
                else EdgeId(f"{road.id.value}::{discriminator}")
            )
            edges.append(
                PhysicalEdge(
                    edge_id,
                    source,
                    target,
                    road.distance_m,
                    duration,
                    road.source_speed_m_per_second or implied_speed,
                )
            )
    return PhysicalNetwork(
        tuple(nodes),
        tuple(sorted(edges, key=lambda item: identifier_key(item.id))),
        authored.coordinate_system,
        authored.coordinate_units,
    )


def _expanded_route_graph(network: PhysicalNetwork) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in network.nodes:
        graph.add_node(
            ("node", type(node.id.value).__name__, node.id.value), domain=node.id
        )
    for edge in network.edges:
        source = ("node", type(edge.source.value).__name__, edge.source.value)
        target = ("node", type(edge.target.value).__name__, edge.target.value)
        private = ("edge", type(edge.id.value).__name__, edge.id.value)
        graph.add_edge(source, private, weight=edge.distance_m, edge_id=edge.id)
        graph.add_edge(private, target, weight=0.0, edge_id=None)
    return graph


def _route_edges_from_expanded(
    nodes: Sequence[Any], graph: nx.DiGraph
) -> tuple[EdgeId, ...]:
    return tuple(
        (
            edge_id
            for left, right in zip(nodes, nodes[1:])
            if (edge_id := graph[left][right]["edge_id"]) is not None
        )
    )


def _expand_vehicles(
    resources: AuthoredScenarioResources, bounds: tuple[float, float]
) -> tuple[VehicleSpec, ...]:
    types = resources.vehicles.by_id()
    result: list[VehicleSpec] = []
    for cohort in sorted(resources.demand, key=lambda item: identifier_key(item.id)):
        asset = types[cohort.vehicle_type_id]
        for index in range(cohort.size):
            result.append(
                VehicleSpec(
                    VehicleId(f"{cohort.id.value}:{index}"),
                    cohort.id,
                    cohort.vehicle_type_id,
                    cohort.origin,
                    cohort.destination,
                    cohort.speed_m_per_second,
                    asset.battery_kwh,
                    asset.efficiency_m_per_kwh,
                    cohort.initial_soc,
                    bounds[0],
                    bounds[1],
                )
            )
    return tuple(result)


def _compile_mcs_units(
    resources: AuthoredScenarioResources,
    bounds: tuple[float, float],
    charging_efficiency: float,
    discharge_efficiency: float,
) -> tuple[MCSUnitSpec, ...]:
    types = {item.id: item for item in resources.mobile_chargers.types}
    return tuple(
        (
            MCSUnitSpec(
                unit.id,
                unit.mcs_type_id,
                types[unit.mcs_type_id].battery_kwh,
                types[unit.mcs_type_id].power_kw,
                types[unit.mcs_type_id].port_count,
                unit.initial_soc,
                bounds[0],
                bounds[1],
                charging_efficiency,
                discharge_efficiency,
                types[unit.mcs_type_id].power_is_unit_total,
            )
            for unit in sorted(
                resources.supply, key=lambda item: identifier_key(item.id)
            )
        )
    )


def _departure_windows(overlay: Mapping[str, Any]) -> tuple[DepartureWindow, ...]:
    raw = _mapping(overlay.get("departure_windows"), label="departure_windows")
    if set(raw) != {"count", "size_seconds"}:
        raise ValidationError(
            "departure_windows requires exactly count and size_seconds."
        )
    count = _exact_integer(raw["count"], label="departure_windows.count", positive=True)
    size = float(raw["size_seconds"])
    if not math.isfinite(size) or size <= 0.0:
        raise ValidationError("departure_windows.size_seconds must be positive.")
    return tuple((DepartureWindow(i, i * size, (i + 1) * size) for i in range(count)))


def _mcs_deployment_domain(network: PhysicalNetwork) -> MCSDeploymentDomain:
    values = [
        (ChargingSiteId(node.id.value), node.mcs_limit)
        for node in network.nodes
        if node.mcs_limit > 0
    ]
    values.sort(key=lambda item: identifier_key(item[0]))
    return MCSDeploymentDomain(tuple((site for site, _ in values)), tuple(values))


def _execution_horizon_seconds(
    network: PhysicalNetwork,
    vehicles: tuple[VehicleSpec, ...],
    groups: tuple[DemandGroup, ...],
    paths: tuple[CandidatePath, ...],
    windows: tuple[DepartureWindow, ...],
    traffic: Any,
    mcs_units: tuple[MCSUnitSpec, ...],
) -> float:
    dt = windows[0].end_seconds - windows[0].start_seconds
    latest_departure = windows[-1].end_seconds
    edge_by_id = {item.id: item for item in network.edges}
    path_by_id = {item.id: item for item in paths}
    vehicle_by_id = {item.id: item for item in vehicles}
    latest_completion = latest_departure
    for group in groups:
        members = tuple((vehicle_by_id[item] for item in group.member_ids))
        for path_id in group.candidate_path_ids:
            path = path_by_id[path_id]
            actions = {item.site_id: item for item in path.charging_actions}
            if len(actions) != len(path.charging_actions):
                raise InvariantError(
                    "Execution horizon requires at most one charging action per path site."
                )
            route_nodes = _route_nodes(network, path.edge_ids)
            current = latest_departure
            for node, edge_id in zip(route_nodes, path.edge_ids):
                action = actions.get(ChargingSiteId(node.value))
                if action is not None:
                    rates = _available_action_rates_kw(network, mcs_units, action)
                    current += 3600.0 * action.requested_energy_kwh / min(rates)
                edge = edge_by_id[edge_id]
                road_speed = edge.distance_m / edge.free_flow_duration_seconds
                robust_duration = max(
                    (
                        edge.distance_m / min(member.speed_m_per_second, road_speed)
                        for member in members
                    )
                )
                current = traffic.exit_time(
                    entry_time_seconds=current,
                    free_flow_duration_seconds=robust_duration,
                )
            latest_completion = max(latest_completion, current)
    return max(dt, math.ceil(latest_completion / dt) * dt)


def _available_action_rates_kw(
    network: PhysicalNetwork, mcs_units: tuple[MCSUnitSpec, ...], action: ChargingAction
) -> tuple[float, ...]:
    rates: list[float] = []
    if "fcs" in action.eligible_provider_kinds:
        rates.extend(
            (
                provider.power_kw * provider.efficiency
                for node in network.nodes
                for provider in node.charging_providers
                if provider.site_id == action.site_id and provider.kind == "fcs"
            )
        )
    if "mcs" in action.eligible_provider_kinds:
        rates.extend(
            (
                mcs_per_port_power_kw(
                    unit.power_kw, unit.port_count, unit.power_is_unit_total
                )
                * unit.charging_efficiency
                for unit in mcs_units
            )
        )
    if not rates or any((value <= 0.0 for value in rates)):
        raise InvariantError(
            f"Charging action {action.id.value!r} has no positive eligible service rate."
        )
    return tuple(rates)


def _evaluation_spec(resources: AuthoredScenarioResources) -> EvaluationSpec:
    authored = resources.evaluation.objectives[0]
    return EvaluationSpec(
        (ObjectiveSpec("completion_time", str(authored.aggregation), 1.0, 1),)
    )


def _feature_vector(vehicle: VehicleSpec) -> tuple[float, float, float, float]:
    return (
        vehicle.speed_m_per_second,
        vehicle.battery_kwh
        * (vehicle.initial_soc - vehicle.minimum_soc)
        * vehicle.efficiency_m_per_kwh,
        1.0 / vehicle.efficiency_m_per_kwh,
        vehicle.battery_kwh * (vehicle.maximum_soc - vehicle.initial_soc),
    )


def _standardized(members: Sequence[VehicleSpec]) -> np.ndarray:
    values = np.asarray([_feature_vector(item) for item in members], dtype=np.float64)
    mean = values.mean(axis=0)
    standard = values.std(axis=0, ddof=0)
    return np.divide(
        values - mean, standard, out=np.zeros_like(values), where=standard != 0.0
    )


def _kmeans_partition(
    members: tuple[VehicleSpec, ...], count: int, seed: int
) -> tuple[tuple[VehicleSpec, ...], ...]:
    ordered = tuple(sorted(members, key=lambda item: identifier_key(item.id)))
    values = _standardized(ordered)
    unique = {tuple(row) for row in values.tolist()}
    k = min(count, len(unique))
    if k <= 1:
        return (ordered,)
    rng = random.Random(seed)
    first = rng.randrange(len(ordered))
    centers = [values[first].copy()]
    while len(centers) < k:
        distances = np.asarray(
            [
                min((float(np.sum((row - center) ** 2)) for center in centers))
                for row in values
            ]
        )
        candidates = np.flatnonzero(distances == distances.max()).tolist()
        centers.append(values[candidates[rng.randrange(len(candidates))]].copy())
    matrix = np.vstack(centers)
    labels: np.ndarray | None = None
    visited_assignments: set[tuple[int, ...]] = set()
    while True:
        distances = ((values[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        clusters = [values[next_labels == index] for index in range(k)]
        if any((cluster.size == 0 for cluster in clusters)):
            raise InvariantError("Deterministic K-means produced an empty cluster.")
        next_centers = np.vstack([cluster.mean(axis=0) for cluster in clusters])
        if labels is not None and np.array_equal(labels, next_labels):
            labels = next_labels
            matrix = next_centers
            break
        signature = tuple((int(item) for item in next_labels))
        if signature in visited_assignments:
            raise InvariantError("Deterministic K-means entered an assignment cycle.")
        visited_assignments.add(signature)
        labels, matrix = (next_labels, next_centers)
    if labels is None:
        raise InvariantError("Deterministic K-means did not produce an assignment.")
    groups = [
        tuple((item for item, label in zip(ordered, labels) if label == index))
        for index in range(k)
    ]
    groups = [item for item in groups if item]
    return tuple(sorted(groups, key=lambda item: identifier_key(item[0].id)))


def _initial_groups(
    vehicles: tuple[VehicleSpec, ...], policy: PreparationPolicy
) -> tuple[tuple[VehicleSpec, ...], ...]:
    result: list[tuple[VehicleSpec, ...]] = []
    keys = sorted(
        {(item.origin, item.destination) for item in vehicles},
        key=lambda x: (identifier_key(x[0]), identifier_key(x[1])),
    )
    for key in keys:
        members = tuple(
            (item for item in vehicles if (item.origin, item.destination) == key)
        )
        result.extend(
            _kmeans_partition(
                members, policy.requested_cluster_count, policy.grouping_seed
            )
        )
    return tuple(result)


def _representative(members: Sequence[VehicleSpec]) -> VehicleSpec:
    ordered = tuple(sorted(members, key=lambda item: identifier_key(item.id)))
    values = _standardized(ordered)
    centroid = values.mean(axis=0)
    distances = ((values - centroid) ** 2).sum(axis=1)
    best = float(distances.min())
    return ordered[
        next((i for i, value in enumerate(distances) if float(value) == best))
    ]


def _geographical_routes(
    network: PhysicalNetwork, origin: NodeId, destination: NodeId, limit: int
) -> tuple[tuple[EdgeId, ...], ...]:
    routes = _deterministic_yen_routes(network, origin, destination, limit)
    if not routes:
        raise ValidationError(
            f"Demand OD {origin.value!r}->{destination.value!r} is unreachable."
        )
    return routes


def _deterministic_yen_routes(
    network: PhysicalNetwork, origin: NodeId, destination: NodeId, limit: int
) -> tuple[tuple[EdgeId, ...], ...]:
    edge_by_id = {item.id: item for item in network.edges}
    first = _shortest_route(
        network, origin, destination, banned_nodes=frozenset(), banned_edges=frozenset()
    )
    if first is None:
        return ()
    accepted = [first]
    accepted_set = {first}
    candidates: list[
        tuple[float, tuple[tuple[str, str], ...], int, tuple[EdgeId, ...]]
    ] = []
    candidate_set: set[tuple[EdgeId, ...]] = set()
    candidate_sequence = count()
    while len(accepted) < limit:
        previous = accepted[-1]
        previous_nodes = _route_nodes(network, previous)
        for spur_index in range(len(previous)):
            root_edges = previous[:spur_index]
            root_nodes = previous_nodes[: spur_index + 1]
            banned_edges = {
                route[spur_index]
                for route in accepted
                if len(route) > spur_index and route[:spur_index] == root_edges
            }
            spur = _shortest_route(
                network,
                root_nodes[-1],
                destination,
                banned_nodes=frozenset(root_nodes[:-1]),
                banned_edges=frozenset(banned_edges),
            )
            if spur is None:
                continue
            candidate = root_edges + spur
            if candidate in accepted_set or candidate in candidate_set:
                continue
            candidate_set.add(candidate)
            heapq.heappush(
                candidates,
                (
                    sum((edge_by_id[item].distance_m for item in candidate)),
                    tuple((identifier_key(item) for item in candidate)),
                    next(candidate_sequence),
                    candidate,
                ),
            )
        if not candidates:
            break
        _, _, _, selected = heapq.heappop(candidates)
        candidate_set.remove(selected)
        accepted.append(selected)
        accepted_set.add(selected)
    return tuple(accepted)


def _shortest_route(
    network: PhysicalNetwork,
    origin: NodeId,
    destination: NodeId,
    *,
    banned_nodes: frozenset[NodeId],
    banned_edges: frozenset[EdgeId],
) -> tuple[EdgeId, ...] | None:
    if origin in banned_nodes or destination in banned_nodes:
        return None
    if origin == destination:
        return ()
    outgoing: dict[NodeId, list[PhysicalEdge]] = {}
    for edge in network.edges:
        if (
            edge.id in banned_edges
            or edge.source in banned_nodes
            or edge.target in banned_nodes
        ):
            continue
        outgoing.setdefault(edge.source, []).append(edge)
    for edges in outgoing.values():
        edges.sort(key=lambda item: identifier_key(item.id))
    queue: list[
        tuple[
            float,
            tuple[tuple[str, str], ...],
            tuple[str, str],
            int,
            NodeId,
            tuple[EdgeId, ...],
        ]
    ]
    queue_sequence = count()
    queue = [(0.0, (), identifier_key(origin), next(queue_sequence), origin, ())]
    best: dict[NodeId, tuple[float, tuple[tuple[str, str], ...]]] = {origin: (0.0, ())}
    while queue:
        distance, signature, _, _, node, edge_ids = heapq.heappop(queue)
        if best.get(node) != (distance, signature):
            continue
        if node == destination:
            return edge_ids
        for edge in outgoing.get(node, ()):
            next_distance = distance + edge.distance_m
            next_signature = signature + (identifier_key(edge.id),)
            current = best.get(edge.target)
            if current is not None and current <= (next_distance, next_signature):
                continue
            best[edge.target] = (next_distance, next_signature)
            heapq.heappush(
                queue,
                (
                    next_distance,
                    next_signature,
                    identifier_key(edge.target),
                    next(queue_sequence),
                    edge.target,
                    edge_ids + (edge.id,),
                ),
            )
    return None


def _route_nodes(
    network: PhysicalNetwork, edge_ids: tuple[EdgeId, ...]
) -> tuple[NodeId, ...]:
    return _route_nodes_from_edges({item.id: item for item in network.edges}, edge_ids)


def _route_nodes_from_edges(
    by_id: Mapping[EdgeId, PhysicalEdge], edge_ids: tuple[EdgeId, ...]
) -> tuple[NodeId, ...]:
    nodes = [by_id[edge_ids[0]].source]
    for edge_id in edge_ids:
        edge = by_id[edge_id]
        if edge.source != nodes[-1]:
            raise InvariantError("Candidate route is not contiguous.")
        nodes.append(edge.target)
    return tuple(nodes)


def _action_levels(
    network: PhysicalNetwork,
    site_node: NodeId,
    members: Sequence[VehicleSpec],
    mcs_units: Sequence[MCSUnitSpec],
    deployment_domain: MCSDeploymentDomain,
    increment_seconds: float,
) -> tuple[ChargingAction, ...]:
    node = next((item for item in network.nodes if item.id == site_node))
    provider_rates: dict[str, set[tuple[float, float]]] = {}
    for provider in node.charging_providers:
        provider_rates.setdefault(provider.kind, set()).add(
            (provider.power_kw, provider.efficiency)
        )
    site = ChargingSiteId(site_node.value)
    if site in deployment_domain.eligible_sites:
        for unit in mcs_units:
            provider_rates.setdefault("mcs", set()).add(
                (
                    mcs_per_port_power_kw(
                        unit.power_kw, unit.port_count, unit.power_is_unit_total
                    ),
                    unit.charging_efficiency,
                )
            )
    if not provider_rates:
        return ()
    maximum_headroom = max(
        (item.battery_kwh * (item.maximum_soc - item.initial_soc) for item in members)
    )
    provider_kinds_by_energy: dict[float, set[str]] = {}
    for kind, rates in sorted(provider_rates.items()):
        if len(rates) != 1:
            raise ValidationError(
                f"Provider kind {kind!r} at site {site.value!r} is heterogeneous; the restricted shared profile requires concrete binding support."
            )
        power_kw, efficiency = next(iter(rates))
        increment = power_kw * increment_seconds * efficiency / 3600.0
        levels = math.floor(maximum_headroom / increment)
        for multiple in range(1, levels + 1):
            energy = increment * multiple
            provider_kinds_by_energy.setdefault(energy, set()).add(kind)
    postcondition = min((item.minimum_soc for item in members))
    actions: list[ChargingAction] = []
    for energy, provider_kinds in sorted(provider_kinds_by_energy.items()):
        eligible_provider_kinds = tuple(sorted(provider_kinds))
        signature = json.dumps(
            (
                (type(site.value).__name__, site.value),
                energy,
                eligible_provider_kinds,
                postcondition,
            ),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        actions.append(
            ChargingAction(
                ChargingActionId(signature),
                site,
                energy,
                eligible_provider_kinds,
                postcondition,
            )
        )
    return tuple(actions)


def _path_feasibility_trajectories(
    network: PhysicalNetwork, paths: tuple[CandidatePath, ...]
) -> dict[PathId, tuple[tuple[ChargingAction | None, PhysicalEdge], ...]]:
    edges = {item.id: item for item in network.edges}
    trajectories: dict[
        PathId, tuple[tuple[ChargingAction | None, PhysicalEdge], ...]
    ] = {}
    for path in paths:
        actions = {NodeId(item.site_id.value): item for item in path.charging_actions}
        nodes = _route_nodes_from_edges(edges, path.edge_ids)
        trajectories[path.id] = tuple(
            (
                (actions.get(node), edges[edge_id])
                for node, edge_id in zip(nodes[:-1], path.edge_ids, strict=True)
            )
        )
    return trajectories


def _path_feasible(
    trajectory: tuple[tuple[ChargingAction | None, PhysicalEdge], ...],
    vehicle: VehicleSpec,
) -> bool:
    soc = vehicle.initial_soc
    for action, edge in trajectory:
        if action is not None:
            soc += action.requested_energy_kwh / vehicle.battery_kwh
            if soc > vehicle.maximum_soc or soc < action.required_post_charge_soc:
                return False
        soc -= edge.distance_m / vehicle.efficiency_m_per_kwh / vehicle.battery_kwh
        if soc < vehicle.minimum_soc:
            return False
    return soc <= vehicle.maximum_soc


def _canonical_path_key(
    network: PhysicalNetwork, path: CandidatePath
) -> tuple[Any, ...]:
    return (
        identifier_key(path.origin),
        identifier_key(path.destination),
        _route_distance(network, path),
        tuple((identifier_key(item) for item in path.edge_ids)),
        len(path.charging_actions),
        tuple(
            (
                (
                    identifier_key(item.site_id),
                    item.requested_energy_kwh,
                    item.eligible_provider_kinds,
                    item.required_post_charge_soc,
                )
                for item in path.charging_actions
            )
        ),
    )


def _candidate_paths(
    network: PhysicalNetwork,
    members: tuple[VehicleSpec, ...],
    mcs_units: tuple[MCSUnitSpec, ...],
    deployment_domain: MCSDeploymentDomain,
    policy: PreparationPolicy,
) -> tuple[CandidatePath, ...]:
    candidates: dict[tuple[Any, ...], CandidatePath] = {}
    for edge_ids in _geographical_routes(
        network,
        members[0].origin,
        members[0].destination,
        policy.geographical_path_limit,
    ):
        nodes = _route_nodes(network, edge_ids)
        site_actions = [
            (
                node,
                _action_levels(
                    network,
                    node,
                    members,
                    mcs_units,
                    deployment_domain,
                    policy.charging_increment_seconds,
                ),
            )
            for node in nodes[:-1]
        ]
        site_actions = [(node, actions) for node, actions in site_actions if actions]
        variants: list[tuple[ChargingAction, ...]] = [()]
        for stop_count in range(
            1, min(policy.charging_stop_limit, len(site_actions)) + 1
        ):
            for selected in combinations(site_actions, stop_count):
                variants.extend(
                    (
                        tuple(items)
                        for items in product(*(actions for _, actions in selected))
                    )
                )
        for actions in variants:
            signature = (
                tuple(((type(item.value).__name__, item.value) for item in edge_ids)),
                tuple(
                    (
                        (
                            type(item.site_id.value).__name__,
                            item.site_id.value,
                            item.requested_energy_kwh,
                            item.eligible_provider_kinds,
                            item.required_post_charge_soc,
                        )
                        for item in actions
                    )
                ),
            )
            path = CandidatePath(
                PathId(
                    json.dumps(signature, separators=(",", ":"), ensure_ascii=False)
                ),
                members[0].origin,
                members[0].destination,
                edge_ids,
                actions,
            )
            candidates.setdefault(signature, path)
    ordered = sorted(
        candidates.values(), key=lambda item: _canonical_path_key(network, item)
    )
    return tuple(ordered)


def _route_distance(network: PhysicalNetwork, path: CandidatePath) -> float:
    edges = {item.id: item for item in network.edges}
    return sum((edges[item].distance_m for item in path.edge_ids))


def _certify_or_split(
    network: PhysicalNetwork,
    members: tuple[VehicleSpec, ...],
    paths: tuple[CandidatePath, ...],
    policy: PreparationPolicy,
    output: list[tuple[tuple[VehicleSpec, ...], tuple[CandidatePath, ...]]],
    trajectories: Mapping[
        PathId, tuple[tuple[ChargingAction | None, PhysicalEdge], ...]
    ]
    | None = None,
) -> None:
    if trajectories is None:
        trajectories = _path_feasibility_trajectories(network, paths)
    common_items: list[CandidatePath] = []
    for path in paths:
        if all((_path_feasible(trajectories[path.id], vehicle) for vehicle in members)):
            common_items.append(path)
            if len(common_items) == policy.charging_path_limit:
                break
    common = tuple(common_items)
    if common:
        output.append(
            (tuple(sorted(members, key=lambda item: identifier_key(item.id))), common)
        )
        return
    if len(members) == 1:
        vehicle = members[0]
        raise ValidationError(
            f"Vehicle {vehicle.id.value!r} has no feasible candidate path."
        )
    for child in _kmeans_partition(members, 2, policy.grouping_seed):
        if len(child) == len(members):
            ordered = tuple(sorted(members, key=lambda item: identifier_key(item.id)))
            midpoint = len(ordered) // 2
            children = (ordered[:midpoint], ordered[midpoint:])
            for fallback in children:
                _certify_or_split(
                    network, fallback, paths, policy, output, trajectories
                )
            return
        _certify_or_split(network, child, paths, policy, output, trajectories)


def _network_identity_data(network: PhysicalNetwork) -> dict[str, Any]:
    return {
        "coordinate_system": network.coordinate_system,
        "coordinate_units": network.coordinate_units,
        "nodes": [
            (
                item.id.to_data(),
                item.coordinate,
                item.mcs_limit,
                [
                    (
                        provider.id.to_data(),
                        provider.site_id.to_data(),
                        provider.kind,
                        provider.port_count,
                        provider.power_kw,
                        provider.efficiency,
                    )
                    for provider in item.charging_providers
                ],
            )
            for item in network.nodes
        ],
        "edges": [
            (
                item.id.to_data(),
                item.source.to_data(),
                item.target.to_data(),
                item.distance_m,
                item.free_flow_duration_seconds,
                item.speed_limit_m_per_second,
            )
            for item in network.edges
        ],
    }


def _vehicle_identity_data(item: VehicleSpec) -> tuple[Any, ...]:
    return tuple(
        (
            getattr(item, name).to_data()
            if hasattr(getattr(item, name), "to_data")
            else getattr(item, name)
            for name in item.__slots__
        )
    )


def _mcs_identity_data(item: MCSUnitSpec) -> tuple[Any, ...]:
    return tuple(
        (
            getattr(item, name).to_data()
            if hasattr(getattr(item, name), "to_data")
            else getattr(item, name)
            for name in item.__slots__
        )
    )


def _path_identity_data(item: CandidatePath) -> dict[str, Any]:
    return {
        "id": item.id.to_data(),
        "origin": item.origin.to_data(),
        "destination": item.destination.to_data(),
        "edges": [edge.to_data() for edge in item.edge_ids],
        "actions": [
            (
                action.id.to_data(),
                action.site_id.to_data(),
                action.requested_energy_kwh,
                action.eligible_provider_kinds,
                action.required_post_charge_soc,
            )
            for action in item.charging_actions
        ],
    }
