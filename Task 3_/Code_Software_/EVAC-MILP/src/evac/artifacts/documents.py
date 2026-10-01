from __future__ import annotations
import math
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast
from evac.artifacts.yaml_io import dump_yaml, load_yaml
from evac.domain import (
    ArtifactLocations,
    CandidatePath,
    ChargingAction,
    ChargingActionId,
    ChargingProviderId,
    ChargingProviderSpec,
    ChargingPolicy,
    ChargingQueuePolicy,
    ChargingSiteId,
    CohortId,
    DemandGroup,
    DepartureWindow,
    DeploymentId,
    EdgeId,
    EvaluationResult,
    EvaluationSpec,
    EvaluationViolation,
    EvacuationPlan,
    MCSDeployment,
    MCSUnitId,
    MCSUnitSpec,
    NodeId,
    ObjectiveSpec,
    NumericalPolicy,
    PathId,
    PhysicalEdge,
    PhysicalNetwork,
    PhysicalNode,
    PlannedChargingAction,
    PlannerResult,
    PlannerTermination,
    PreparationPolicy,
    PreparedScenario,
    ProviderBindingMode,
    ProviderDispatchPolicy,
    RunResult,
    SemanticIdentity,
    SimulationResult,
    MCSDeploymentDomain,
    TimelineEvent,
    VehicleId,
    VehiclePlan,
    VehicleSpec,
    identifier_from_data,
)
from evac.domain.identifiers import AssetTypeId, DomainIdentifier, ScenarioId
from evac.errors import ValidationError
from evac.physics import ExogenousProfile, TrafficSpec

PLAN_SCHEMA = "evac/plan/v1"
PREPARED_SCENARIO_SCHEMA = "evac/prepared-scenario/v1"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        (not isinstance(key, str) for key in value)
    ):
        raise ValidationError(f"{label} must be a string-keyed mapping.")
    return value


def _fields(value: Mapping[str, Any], *, fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValidationError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}."
        )


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list.")
    return value


IdentifierT = TypeVar("IdentifierT", bound=DomainIdentifier)


def _identifier(value: object, expected: type[IdentifierT]) -> IdentifierT:
    return cast(IdentifierT, identifier_from_data(value, expected_type=expected))


def _identity_to_data(value: SemanticIdentity) -> dict[str, str]:
    return {"scope": value.scope, "value": value.value}


def _identity_from_data(value: object, *, label: str) -> SemanticIdentity:
    data = _mapping(value, label=label)
    _fields(data, fields={"scope", "value"}, label=label)
    return SemanticIdentity(data["scope"], data["value"])


def _plain(value: Any, *, label: str) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{label} contains a non-finite number.")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, label=label) for item in value]
    if isinstance(value, Mapping):
        if any((not isinstance(key, str) for key in value)):
            raise ValidationError(f"{label} mapping keys must be strings.")
        return {key: _plain(item, label=label) for key, item in value.items()}
    raise ValidationError(f"{label} contains unsupported value {value!r}.")


def _plain_mapping(value: object, *, label: str) -> dict[str, Any]:
    return cast(dict[str, Any], _plain(dict(_mapping(value, label=label)), label=label))


def _plan_to_data(plan: EvacuationPlan) -> dict[str, Any]:
    return {
        "identity": _identity_to_data(plan.identity),
        "preparation_identity": _identity_to_data(plan.preparation_identity),
        "vehicle_plans": [
            {
                "vehicle_id": item.vehicle_id.to_data(),
                "path_id": item.path_id.to_data(),
                "departure_seconds": item.departure_seconds,
                "charging_actions": [
                    {
                        "action_id": action.action_id.to_data(),
                        "binding_mode": action.binding_mode.value,
                        "provider_id": None
                        if action.provider_id is None
                        else action.provider_id.to_data(),
                        "provider_kind": action.provider_kind,
                    }
                    for action in item.charging_actions
                ],
            }
            for item in plan.vehicle_plans
        ],
        "mcs_deployments": [
            {
                "id": item.id.to_data(),
                "mcs_unit_id": item.mcs_unit_id.to_data(),
                "site_id": item.site_id.to_data(),
            }
            for item in plan.mcs_deployments
        ],
    }


def _plan_from_data(value: object) -> EvacuationPlan:
    data = _mapping(value, label="EvacuationPlan")
    fields = {"identity", "preparation_identity", "vehicle_plans", "mcs_deployments"}
    _fields(data, fields=fields, label="EvacuationPlan")
    vehicle_plans: list[VehiclePlan] = []
    for index, raw in enumerate(_list(data["vehicle_plans"], label="vehicle_plans")):
        item = _mapping(raw, label=f"vehicle_plan {index}")
        item_fields = {"vehicle_id", "path_id", "departure_seconds", "charging_actions"}
        _fields(item, fields=item_fields, label=f"vehicle_plan {index}")
        actions: list[PlannedChargingAction] = []
        for action_index, raw_action in enumerate(
            _list(
                item["charging_actions"], label=f"vehicle_plan {index} charging_actions"
            )
        ):
            action = _mapping(
                raw_action, label=f"planned charging action {action_index}"
            )
            current_action_fields = {
                "action_id",
                "binding_mode",
                "provider_id",
                "provider_kind",
            }
            _fields(
                action, fields=current_action_fields, label="planned charging action"
            )
            provider_kind = action.get("provider_kind")
            actions.append(
                PlannedChargingAction(
                    _identifier(action["action_id"], ChargingActionId),
                    None
                    if action["provider_id"] is None
                    else _identifier(action["provider_id"], ChargingProviderId),
                    provider_kind,
                    ProviderBindingMode(action["binding_mode"]),
                )
            )
        vehicle_plans.append(
            VehiclePlan(
                _identifier(item["vehicle_id"], VehicleId),
                _identifier(item["path_id"], PathId),
                item["departure_seconds"],
                tuple(actions),
            )
        )
    deployments: list[MCSDeployment] = []
    for index, raw in enumerate(
        _list(data["mcs_deployments"], label="mcs_deployments")
    ):
        item = _mapping(raw, label=f"mcs_deployment {index}")
        item_fields = {"id", "mcs_unit_id", "site_id"}
        _fields(item, fields=item_fields, label=f"mcs_deployment {index}")
        deployments.append(
            MCSDeployment(
                _identifier(item["id"], DeploymentId),
                _identifier(item["mcs_unit_id"], MCSUnitId),
                _identifier(item["site_id"], ChargingSiteId),
            )
        )
    return EvacuationPlan(
        _identity_from_data(data["identity"], label="plan identity"),
        _identity_from_data(data["preparation_identity"], label="preparation identity"),
        tuple(vehicle_plans),
        tuple(deployments),
    )


def save_plan(plan: EvacuationPlan, path: str | Path) -> None:
    if not isinstance(plan, EvacuationPlan):
        raise ValidationError("save_plan requires an EvacuationPlan.")
    dump_yaml(
        {"schema": PLAN_SCHEMA, "plan": _plan_to_data(plan)}, Path(path).resolve()
    )


def load_plan(path: str | Path) -> EvacuationPlan:
    data = _mapping(load_yaml(Path(path).resolve()), label="Plan document")
    _fields(data, fields={"schema", "plan"}, label="Plan document")
    if data["schema"] != PLAN_SCHEMA:
        raise ValidationError(f"Unsupported Plan schema {data['schema']!r}.")
    return _plan_from_data(data["plan"])


def _traffic_to_data(value):
    return {"exogenous_profile": value.exogenous_profile.to_data()}


def _traffic_from_data(value):
    data = _mapping(value, label="TrafficSpec")
    _fields(data, fields={"exogenous_profile"}, label="TrafficSpec")
    return TrafficSpec(ExogenousProfile.from_data(data["exogenous_profile"]))


def _evaluation_spec_to_data(value: EvaluationSpec) -> dict[str, Any]:
    return {
        "objectives": [
            {
                "name": item.name,
                "norm": item.norm,
                "weight": item.weight,
                "priority": item.priority,
            }
            for item in value.objectives
        ]
    }


def _evaluation_spec_from_data(value: object) -> EvaluationSpec:
    data = _mapping(value, label="EvaluationSpec")
    _fields(data, fields={"objectives"}, label="EvaluationSpec")
    objectives: list[ObjectiveSpec] = []
    for raw in _list(data["objectives"], label="objectives"):
        item = _mapping(raw, label="ObjectiveSpec")
        _fields(
            item, fields={"name", "norm", "weight", "priority"}, label="ObjectiveSpec"
        )
        objectives.append(
            ObjectiveSpec(item["name"], item["norm"], item["weight"], item["priority"])
        )
    return EvaluationSpec(tuple(objectives))


def _prepared_to_data(value: PreparedScenario) -> dict[str, Any]:
    return {
        "scenario_id": value.scenario_id.to_data(),
        "identity": _identity_to_data(value.identity),
        "unit_system": dict(value.unit_system),
        "network": {
            "coordinate_system": value.network.coordinate_system,
            "coordinate_units": value.network.coordinate_units,
            "nodes": [
                {
                    "id": node.id.to_data(),
                    "coordinate": list(node.coordinate),
                    "mcs_limit": node.mcs_limit,
                    "charging_providers": [
                        {
                            "id": provider.id.to_data(),
                            "site_id": provider.site_id.to_data(),
                            "kind": provider.kind,
                            "port_count": provider.port_count,
                            "power_kw": provider.power_kw,
                            "efficiency": provider.efficiency,
                        }
                        for provider in node.charging_providers
                    ],
                }
                for node in value.network.nodes
            ],
            "edges": [
                {
                    "id": edge.id.to_data(),
                    "source": edge.source.to_data(),
                    "target": edge.target.to_data(),
                    "distance_m": edge.distance_m,
                    "free_flow_duration_seconds": edge.free_flow_duration_seconds,
                    "speed_limit_m_per_second": edge.speed_limit_m_per_second,
                }
                for edge in value.network.edges
            ],
        },
        "vehicles": [
            {
                "id": item.id.to_data(),
                "cohort_id": item.cohort_id.to_data(),
                "vehicle_type_id": item.vehicle_type_id.to_data(),
                "origin": item.origin.to_data(),
                "destination": item.destination.to_data(),
                "speed_m_per_second": item.speed_m_per_second,
                "battery_kwh": item.battery_kwh,
                "efficiency_m_per_kwh": item.efficiency_m_per_kwh,
                "initial_soc": item.initial_soc,
                "minimum_soc": item.minimum_soc,
                "maximum_soc": item.maximum_soc,
            }
            for item in value.vehicles
        ],
        "mcs_units": [
            {
                "id": item.id.to_data(),
                "asset_type_id": item.asset_type_id.to_data(),
                "battery_kwh": item.battery_kwh,
                "power_kw": item.power_kw,
                "port_count": item.port_count,
                "initial_soc": item.initial_soc,
                "minimum_soc": item.minimum_soc,
                "maximum_soc": item.maximum_soc,
                "charging_efficiency": item.charging_efficiency,
                "discharge_efficiency": item.discharge_efficiency,
                "power_is_unit_total": item.power_is_unit_total,
            }
            for item in value.mcs_units
        ],
        "departure_windows": [
            {
                "index": item.index,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
            }
            for item in value.departure_windows
        ],
        "mcs_deployment_domain": {
            "eligible_sites": [
                item.to_data() for item in value.mcs_deployment_domain.eligible_sites
            ],
            "site_limits": [
                {"site_id": site.to_data(), "limit": limit}
                for site, limit in value.mcs_deployment_domain.site_limits
            ],
        },
        "demand_groups": [
            {
                "id": item.id.to_data(),
                "member_ids": [member.to_data() for member in item.member_ids],
                "representative_id": item.representative_id.to_data(),
                "candidate_path_ids": [
                    path.to_data() for path in item.candidate_path_ids
                ],
            }
            for item in value.demand_groups
        ],
        "candidate_paths": [
            {
                "id": item.id.to_data(),
                "origin": item.origin.to_data(),
                "destination": item.destination.to_data(),
                "edge_ids": [edge.to_data() for edge in item.edge_ids],
                "charging_actions": [
                    {
                        "id": action.id.to_data(),
                        "site_id": action.site_id.to_data(),
                        "requested_energy_kwh": action.requested_energy_kwh,
                        "eligible_provider_kinds": list(action.eligible_provider_kinds),
                        "required_post_charge_soc": action.required_post_charge_soc,
                    }
                    for action in item.charging_actions
                ],
            }
            for item in value.candidate_paths
        ],
        "traffic": _traffic_to_data(value.traffic),
        "evaluation": _evaluation_spec_to_data(value.evaluation),
        "preparation_policy": {
            "profile_id": value.preparation_policy.profile_id,
            "grouping_seed": value.preparation_policy.grouping_seed,
            "requested_cluster_count": value.preparation_policy.requested_cluster_count,
            "geographical_path_limit": value.preparation_policy.geographical_path_limit,
            "charging_path_limit": value.preparation_policy.charging_path_limit,
            "charging_stop_limit": value.preparation_policy.charging_stop_limit,
            "charging_increment_seconds": value.preparation_policy.charging_increment_seconds,
        },
        "numerical_policy": {
            "event_time_tolerance_seconds": value.numerical_policy.event_time_tolerance_seconds
        },
        "charging_policy": {
            "queueing": value.charging_policy.queueing.value,
            "provider_dispatch": value.charging_policy.provider_dispatch.value,
            "queue_capacity_vehicles": value.charging_policy.queue_capacity_vehicles,
        },
        "planning_horizon_seconds": value.planning_horizon_seconds,
    }


def _prepared_from_data(value: object) -> PreparedScenario:
    data = _mapping(value, label="PreparedScenario")
    fields = {
        "scenario_id",
        "identity",
        "unit_system",
        "network",
        "vehicles",
        "mcs_units",
        "departure_windows",
        "mcs_deployment_domain",
        "demand_groups",
        "candidate_paths",
        "traffic",
        "evaluation",
        "preparation_policy",
        "numerical_policy",
        "charging_policy",
        "planning_horizon_seconds",
    }
    _fields(data, fields=fields, label="PreparedScenario")
    network_data = _mapping(data["network"], label="PhysicalNetwork")
    _fields(
        network_data,
        fields={"coordinate_system", "coordinate_units", "nodes", "edges"},
        label="PhysicalNetwork",
    )
    nodes: list[PhysicalNode] = []
    for raw in _list(network_data["nodes"], label="physical nodes"):
        item = _mapping(raw, label="PhysicalNode")
        _fields(
            item,
            fields={"id", "coordinate", "mcs_limit", "charging_providers"},
            label="PhysicalNode",
        )
        providers: list[ChargingProviderSpec] = []
        for raw_provider in _list(
            item["charging_providers"], label="charging providers"
        ):
            provider = _mapping(raw_provider, label="ChargingProviderSpec")
            provider_fields = {
                "id",
                "site_id",
                "kind",
                "port_count",
                "power_kw",
                "efficiency",
            }
            _fields(provider, fields=provider_fields, label="ChargingProviderSpec")
            providers.append(
                ChargingProviderSpec(
                    _identifier(provider["id"], ChargingProviderId),
                    _identifier(provider["site_id"], ChargingSiteId),
                    provider["kind"],
                    provider["port_count"],
                    provider["power_kw"],
                    provider["efficiency"],
                )
            )
        coordinate = _list(item["coordinate"], label="node coordinate")
        nodes.append(
            PhysicalNode(
                _identifier(item["id"], NodeId),
                tuple(coordinate),
                tuple(providers),
                item["mcs_limit"],
            )
        )
    edges: list[PhysicalEdge] = []
    for raw in _list(network_data["edges"], label="physical edges"):
        item = _mapping(raw, label="PhysicalEdge")
        edge_fields = {
            "id",
            "source",
            "target",
            "distance_m",
            "free_flow_duration_seconds",
            "speed_limit_m_per_second",
        }
        _fields(item, fields=edge_fields, label="PhysicalEdge")
        edges.append(
            PhysicalEdge(
                _identifier(item["id"], EdgeId),
                _identifier(item["source"], NodeId),
                _identifier(item["target"], NodeId),
                item["distance_m"],
                item["free_flow_duration_seconds"],
                item["speed_limit_m_per_second"],
            )
        )
    vehicles: list[VehicleSpec] = []
    for raw in _list(data["vehicles"], label="vehicles"):
        item = _mapping(raw, label="VehicleSpec")
        vehicle_fields = {
            "id",
            "cohort_id",
            "vehicle_type_id",
            "origin",
            "destination",
            "speed_m_per_second",
            "battery_kwh",
            "efficiency_m_per_kwh",
            "initial_soc",
            "minimum_soc",
            "maximum_soc",
        }
        _fields(item, fields=vehicle_fields, label="VehicleSpec")
        vehicles.append(
            VehicleSpec(
                _identifier(item["id"], VehicleId),
                _identifier(item["cohort_id"], CohortId),
                _identifier(item["vehicle_type_id"], AssetTypeId),
                _identifier(item["origin"], NodeId),
                _identifier(item["destination"], NodeId),
                item["speed_m_per_second"],
                item["battery_kwh"],
                item["efficiency_m_per_kwh"],
                item["initial_soc"],
                item["minimum_soc"],
                item["maximum_soc"],
            )
        )
    mcs_units: list[MCSUnitSpec] = []
    for raw in _list(data["mcs_units"], label="MCS units"):
        item = _mapping(raw, label="MCSUnitSpec")
        unit_fields = {
            "id",
            "asset_type_id",
            "battery_kwh",
            "power_kw",
            "port_count",
            "initial_soc",
            "minimum_soc",
            "maximum_soc",
            "charging_efficiency",
            "discharge_efficiency",
            "power_is_unit_total",
        }
        _fields(item, fields=unit_fields, label="MCSUnitSpec")
        mcs_units.append(
            MCSUnitSpec(
                _identifier(item["id"], MCSUnitId),
                _identifier(item["asset_type_id"], AssetTypeId),
                item["battery_kwh"],
                item["power_kw"],
                item["port_count"],
                item["initial_soc"],
                item["minimum_soc"],
                item["maximum_soc"],
                item["charging_efficiency"],
                item["discharge_efficiency"],
                item["power_is_unit_total"],
            )
        )
    windows: list[DepartureWindow] = []
    for raw in _list(data["departure_windows"], label="departure windows"):
        item = _mapping(raw, label="DepartureWindow")
        _fields(
            item,
            fields={"index", "start_seconds", "end_seconds"},
            label="DepartureWindow",
        )
        windows.append(
            DepartureWindow(item["index"], item["start_seconds"], item["end_seconds"])
        )
    deployment_data = _mapping(
        data["mcs_deployment_domain"], label="MCSDeploymentDomain"
    )
    _fields(
        deployment_data,
        fields={"eligible_sites", "site_limits"},
        label="MCSDeploymentDomain",
    )
    eligible = tuple(
        (
            _identifier(item, ChargingSiteId)
            for item in _list(deployment_data["eligible_sites"], label="eligible sites")
        )
    )
    limits: list[tuple[ChargingSiteId, int]] = []
    for raw in _list(deployment_data["site_limits"], label="site limits"):
        item = _mapping(raw, label="site limit")
        _fields(item, fields={"site_id", "limit"}, label="site limit")
        limits.append((_identifier(item["site_id"], ChargingSiteId), item["limit"]))
    groups: list[DemandGroup] = []
    for raw in _list(data["demand_groups"], label="demand groups"):
        item = _mapping(raw, label="DemandGroup")
        group_fields = {"id", "member_ids", "representative_id", "candidate_path_ids"}
        _fields(item, fields=group_fields, label="DemandGroup")
        groups.append(
            DemandGroup(
                _identifier(item["id"], CohortId),
                tuple(
                    (
                        _identifier(member, VehicleId)
                        for member in _list(item["member_ids"], label="member IDs")
                    )
                ),
                _identifier(item["representative_id"], VehicleId),
                tuple(
                    (
                        _identifier(path, PathId)
                        for path in _list(item["candidate_path_ids"], label="path IDs")
                    )
                ),
            )
        )
    paths: list[CandidatePath] = []
    for raw in _list(data["candidate_paths"], label="candidate paths"):
        item = _mapping(raw, label="CandidatePath")
        path_fields = {"id", "origin", "destination", "edge_ids", "charging_actions"}
        _fields(item, fields=path_fields, label="CandidatePath")
        actions: list[ChargingAction] = []
        for raw_action in _list(item["charging_actions"], label="charging actions"):
            action = _mapping(raw_action, label="ChargingAction")
            action_fields = {
                "id",
                "site_id",
                "requested_energy_kwh",
                "eligible_provider_kinds",
                "required_post_charge_soc",
            }
            _fields(action, fields=action_fields, label="ChargingAction")
            actions.append(
                ChargingAction(
                    _identifier(action["id"], ChargingActionId),
                    _identifier(action["site_id"], ChargingSiteId),
                    action["requested_energy_kwh"],
                    tuple(
                        _list(
                            action["eligible_provider_kinds"],
                            label="eligible provider kinds",
                        )
                    ),
                    action["required_post_charge_soc"],
                )
            )
        paths.append(
            CandidatePath(
                _identifier(item["id"], PathId),
                _identifier(item["origin"], NodeId),
                _identifier(item["destination"], NodeId),
                tuple(
                    (
                        _identifier(edge, EdgeId)
                        for edge in _list(item["edge_ids"], label="edge IDs")
                    )
                ),
                tuple(actions),
            )
        )
    preparation_data = _mapping(data["preparation_policy"], label="PreparationPolicy")
    preparation_fields = {
        "profile_id",
        "grouping_seed",
        "requested_cluster_count",
        "geographical_path_limit",
        "charging_path_limit",
        "charging_stop_limit",
        "charging_increment_seconds",
    }
    _fields(preparation_data, fields=preparation_fields, label="PreparationPolicy")
    numerical_data = _mapping(data["numerical_policy"], label="NumericalPolicy")
    _fields(
        numerical_data, fields={"event_time_tolerance_seconds"}, label="NumericalPolicy"
    )
    charging_data = _mapping(data["charging_policy"], label="ChargingPolicy")
    _fields(
        charging_data,
        fields={"queueing", "provider_dispatch", "queue_capacity_vehicles"},
        label="ChargingPolicy",
    )
    return PreparedScenario(
        _identifier(data["scenario_id"], ScenarioId),
        _identity_from_data(data["identity"], label="preparation identity"),
        _plain_mapping(data["unit_system"], label="unit system"),
        PhysicalNetwork(
            tuple(nodes),
            tuple(edges),
            network_data["coordinate_system"],
            network_data["coordinate_units"],
        ),
        tuple(vehicles),
        tuple(mcs_units),
        tuple(windows),
        MCSDeploymentDomain(eligible, tuple(limits)),
        tuple(groups),
        tuple(paths),
        _traffic_from_data(data["traffic"]),
        _evaluation_spec_from_data(data["evaluation"]),
        PreparationPolicy(**dict(preparation_data)),
        NumericalPolicy(numerical_data["event_time_tolerance_seconds"]),
        ChargingPolicy(
            ChargingQueuePolicy(charging_data["queueing"]),
            ProviderDispatchPolicy(charging_data["provider_dispatch"]),
            charging_data["queue_capacity_vehicles"],
        ),
        data["planning_horizon_seconds"],
    )


def save_prepared_scenario(value: PreparedScenario, path: str | Path) -> None:
    if not isinstance(value, PreparedScenario):
        raise ValidationError("save_prepared_scenario requires a PreparedScenario.")
    dump_yaml(
        {
            "schema": PREPARED_SCENARIO_SCHEMA,
            "prepared_scenario": _prepared_to_data(value),
        },
        Path(path).resolve(),
    )


def load_prepared_scenario(path: str | Path) -> PreparedScenario:
    data = _mapping(load_yaml(Path(path).resolve()), label="PreparedScenario document")
    _fields(
        data, fields={"schema", "prepared_scenario"}, label="PreparedScenario document"
    )
    if data["schema"] != PREPARED_SCENARIO_SCHEMA:
        raise ValidationError(
            f"Unsupported PreparedScenario schema {data['schema']!r}."
        )
    return _prepared_from_data(data["prepared_scenario"])


def _timeline_to_data(value: TimelineEvent) -> dict[str, Any]:
    return {
        "time_seconds": value.time_seconds,
        "kind": value.kind,
        "subject_namespace": value.subject_namespace,
        "subject_value": value.subject_value,
        "values": _plain(value.values, label="timeline values"),
    }


def _timeline_from_data(value: object) -> TimelineEvent:
    data = _mapping(value, label="TimelineEvent")
    fields = {"time_seconds", "kind", "subject_namespace", "subject_value", "values"}
    _fields(data, fields=fields, label="TimelineEvent")
    return TimelineEvent(
        data["time_seconds"],
        data["kind"],
        data["subject_namespace"],
        data["subject_value"],
        _plain_mapping(data["values"], label="timeline values"),
    )


def _simulation_to_data(value: SimulationResult) -> dict[str, Any]:
    return {
        "identity": _identity_to_data(value.identity),
        "preparation_identity": _identity_to_data(value.preparation_identity),
        "plan_identity": _identity_to_data(value.plan_identity),
        "vehicle_events": [_timeline_to_data(item) for item in value.vehicle_events],
        "edge_events": [_timeline_to_data(item) for item in value.edge_events],
        "charging_site_events": [
            _timeline_to_data(item) for item in value.charging_site_events
        ],
        "mcs_events": [_timeline_to_data(item) for item in value.mcs_events],
        "metadata": _plain(value.metadata, label="simulation metadata"),
        "vehicle_terminal_reasons": [
            {"vehicle_id": vehicle_id.to_data(), "reason": reason}
            for vehicle_id, reason in value.vehicle_terminal_reasons
        ],
        "terminal_reason": value.terminal_reason,
    }


def _simulation_from_data(value: object) -> SimulationResult:
    data = _mapping(value, label="SimulationResult")
    fields = {
        "identity",
        "preparation_identity",
        "plan_identity",
        "vehicle_events",
        "edge_events",
        "charging_site_events",
        "mcs_events",
        "metadata",
        "vehicle_terminal_reasons",
        "terminal_reason",
    }
    _fields(data, fields=fields, label="SimulationResult")

    def events(name: str) -> tuple[TimelineEvent, ...]:
        return tuple(
            (_timeline_from_data(item) for item in _list(data[name], label=name))
        )

    terminal_reasons: list[tuple[VehicleId, str]] = []
    for raw in _list(
        data["vehicle_terminal_reasons"], label="vehicle terminal reasons"
    ):
        item = _mapping(raw, label="vehicle terminal reason")
        _fields(item, fields={"vehicle_id", "reason"}, label="vehicle terminal reason")
        terminal_reasons.append(
            (_identifier(item["vehicle_id"], VehicleId), item["reason"])
        )
    return SimulationResult(
        _identity_from_data(data["identity"], label="simulation identity"),
        _identity_from_data(data["preparation_identity"], label="preparation identity"),
        _identity_from_data(data["plan_identity"], label="plan identity"),
        events("vehicle_events"),
        events("edge_events"),
        events("charging_site_events"),
        events("mcs_events"),
        _plain_mapping(data["metadata"], label="simulation metadata"),
        tuple(terminal_reasons),
        data["terminal_reason"],
    )


def _evaluation_to_data(value: EvaluationResult) -> dict[str, Any]:
    return {
        "identity": _identity_to_data(value.identity),
        "simulation_identity": _identity_to_data(value.simulation_identity),
        "objective_name": value.objective_name,
        "scientific_objective": value.scientific_objective,
        "feasible": value.feasible,
        "violations": [
            {
                "kind": item.kind,
                "subject_namespace": item.subject_namespace,
                "subject_value": item.subject_value,
                "magnitude": item.magnitude,
                "detail": item.detail,
            }
            for item in value.violations
        ],
        "metrics": _plain(value.metrics, label="evaluation metrics"),
    }


def _evaluation_from_data(value: object) -> EvaluationResult:
    data = _mapping(value, label="EvaluationResult")
    fields = {
        "identity",
        "simulation_identity",
        "objective_name",
        "scientific_objective",
        "feasible",
        "violations",
        "metrics",
    }
    _fields(data, fields=fields, label="EvaluationResult")
    violations: list[EvaluationViolation] = []
    for raw in _list(data["violations"], label="violations"):
        item = _mapping(raw, label="EvaluationViolation")
        violation_fields = {
            "kind",
            "subject_namespace",
            "subject_value",
            "magnitude",
            "detail",
        }
        _fields(item, fields=violation_fields, label="EvaluationViolation")
        violations.append(
            EvaluationViolation(
                item["kind"],
                item["subject_namespace"],
                item["subject_value"],
                item["magnitude"],
                item["detail"],
            )
        )
    return EvaluationResult(
        _identity_from_data(data["identity"], label="evaluation identity"),
        _identity_from_data(data["simulation_identity"], label="simulation identity"),
        data["objective_name"],
        data["scientific_objective"],
        data["feasible"],
        tuple(violations),
        _plain_mapping(data["metrics"], label="evaluation metrics"),
    )


def _planner_to_data(value):
    return {
        "planner_kind": value.planner_kind,
        "model_identity": _identity_to_data(value.model_identity),
        "termination": value.termination.value,
        "plan": None if value.plan is None else _plan_to_data(value.plan),
        "objective_bound": value.objective_bound,
        "objective_gap": value.objective_gap,
        "terminal_cause": value.terminal_cause,
        "metadata": _plain_mapping(value.metadata, label="planner metadata"),
        "formulation_objective": value.formulation_objective,
    }


def _planner_from_data(data):
    return PlannerResult(
        data["planner_kind"],
        _identity_from_data(data["model_identity"], label="model identity"),
        PlannerTermination(data["termination"]),
        None if data["plan"] is None else _plan_from_data(data["plan"]),
        data["objective_bound"],
        data["objective_gap"],
        data["terminal_cause"],
        data["metadata"],
        formulation_objective=data["formulation_objective"],
    )


def save_result(result, path):
    dump_yaml(
        {
            "schema": "evac/result/v1",
            "planner": _planner_to_data(result.planner),
            "simulation": None
            if result.simulation is None
            else _simulation_to_data(result.simulation),
            "evaluation": None
            if result.evaluation is None
            else _evaluation_to_data(result.evaluation),
            "timings": dict(result.timings),
        },
        Path(path),
    )


def load_result(path):
    data = load_yaml(path)
    if data["schema"] != "evac/result/v1":
        raise ValidationError("Unsupported result schema.")
    planner = _planner_from_data(data["planner"])
    return RunResult(
        planner,
        None
        if data["simulation"] is None
        else _simulation_from_data(data["simulation"]),
        None
        if data["evaluation"] is None
        else _evaluation_from_data(data["evaluation"]),
        ArtifactLocations(None, {}),
        data["timings"],
    )
