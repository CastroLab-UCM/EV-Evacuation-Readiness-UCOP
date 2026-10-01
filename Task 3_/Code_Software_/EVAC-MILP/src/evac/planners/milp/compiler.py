from __future__ import annotations
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping
import networkx as nx
import numpy as np
import pandas as pd
from evac.domain import (
    CandidatePath,
    ChargingSiteId,
    DemandGroup,
    EdgeId,
    MCSUnitId,
    NodeId,
    ObjectiveSpec,
    PreparedScenario,
    VehicleId,
)
from evac.domain.ordering import identifier_key
from evac.errors import UnsupportedCapabilityError, ValidationError
from evac.physics import mcs_per_port_power_kw
from evac.physics.units import CANONICAL_UNIT_SYSTEM
from evac.planners.milp._source.model_fingerprint import (
    GROUP_CERTIFICATION_SCHEMA,
    NORMALIZED_INPUT_SCHEMA,
    PATH_LIBRARY_SCHEMA,
    fingerprint_component,
    group_certification_payload,
    normalized_input_payload,
    path_library_payload,
)

_TOLERANCE = 1e-09


@dataclass(frozen=True, slots=True)
class MILPProjection:
    prepared: PreparedScenario
    pack: Mapping[str, Any]
    group_by_source_id: Mapping[int, DemandGroup]
    path_by_source_index: Mapping[tuple[int, int], CandidatePath]
    vehicle_by_source_id: Mapping[int, VehicleId]
    mcs_by_source_id: Mapping[int, MCSUnitId]
    node_by_source_label: Mapping[str, NodeId]


def compile_prepared_scenario(
    prepared: PreparedScenario, *, objectives: tuple[ObjectiveSpec, ...] | None = None
) -> MILPProjection:
    if not isinstance(prepared, PreparedScenario):
        raise ValidationError("MILP compilation requires a PreparedScenario.")
    scenario_window_count = len(prepared.departure_windows)
    window_count = scenario_window_count
    dt = _window_size(prepared)
    node_to_label, label_to_node = _node_labels(prepared)
    edge_by_id, base_graph = _base_graph(prepared, node_to_label)
    recharge = _recharge_contract(prepared)
    augmented = base_graph.copy()
    vehicles = sorted(prepared.vehicles, key=lambda item: identifier_key(item.id))
    vehicle_source_id = {item.id: index for index, item in enumerate(vehicles)}
    vehicle_by_source_id = {value: key for key, value in vehicle_source_id.items()}
    groups = tuple(prepared.demand_groups)
    group_by_source_id = {index: group for index, group in enumerate(groups)}
    group_for_vehicle = {
        member_id: source_id
        for source_id, group in group_by_source_id.items()
        for member_id in group.member_ids
    }
    representative_ids: set[VehicleId] = set()
    for source_id, group in group_by_source_id.items():
        representative_ids.add(group.representative_id)
    demand_rows = [
        {
            "id": vehicle_source_id[vehicle.id],
            "group_id": group_for_vehicle[vehicle.id],
            "representative": vehicle.id in representative_ids,
            "origin": node_to_label[vehicle.origin],
            "destination": node_to_label[vehicle.destination],
            "speed": vehicle.speed_m_per_second,
            "battery": vehicle.battery_kwh,
            "efficiency": vehicle.efficiency_m_per_kwh,
            "soc": vehicle.initial_soc,
        }
        for vehicle in vehicles
    ]
    demand = pd.DataFrame(demand_rows)
    mcs_units = sorted(prepared.mcs_units, key=lambda item: identifier_key(item.id))
    mcs_source_id = {item.id: index for index, item in enumerate(mcs_units)}
    mcs_by_source_id = {value: key for key, value in mcs_source_id.items()}
    supply = pd.DataFrame(
        [
            {
                "id": mcs_source_id[unit.id],
                "battery": unit.battery_kwh,
                "port": unit.port_count,
                "power": _mcs_per_port_power(unit),
                "soc": unit.initial_soc,
            }
            for unit in mcs_units
        ],
        columns=("id", "battery", "port", "power", "soc"),
    )
    paths_by_id = {item.id: item for item in prepared.candidate_paths}
    vehicles_by_id = {item.id: item for item in prepared.vehicles}
    library: dict[int, list[pd.DataFrame]] = {}
    path_by_source_index: dict[tuple[int, int], CandidatePath] = {}
    grouping_records: list[dict[str, Any]] = []
    for source_group_id, group in group_by_source_id.items():
        members = tuple((vehicles_by_id[item] for item in group.member_ids))
        retained: list[pd.DataFrame] = []
        for path_index, path_id in enumerate(group.candidate_path_ids):
            path = paths_by_id[path_id]
            retained.append(
                _path_frame(
                    path,
                    members,
                    edge_by_id=edge_by_id,
                    base_graph=base_graph,
                    augmented_graph=augmented,
                    node_to_label=node_to_label,
                    effective_power_kw=recharge["power"] * recharge["efficiency"],
                    representative_id=group.representative_id,
                )
            )
            path_by_source_index[source_group_id, path_index] = path
        library[source_group_id] = retained
        grouping_records.append(
            {
                "group_id": source_group_id,
                "canonical_group_id": group.id.to_data(),
                "representative_id": vehicle_source_id[group.representative_id],
                "member_ids": [vehicle_source_id[item] for item in group.member_ids],
                "member_count": len(group.member_ids),
                "candidate_path_count": len(group.candidate_path_ids),
                "retained_path_count": len(retained),
                "rejected_path_count": 0,
                "objective_semantics": "conservative_upper_bound",
            }
        )
    horizon_seconds = prepared.planning_horizon_seconds
    source_objectives = []
    for objective in (
        prepared.evaluation.objectives if objectives is None else objectives
    ):
        if objective.name != "completion_time" or objective.norm not in {"mean", "max"}:
            raise UnsupportedCapabilityError(
                "MILP supports only canonical completion_time/mean|max."
            )
        source_objectives.append(
            {
                "name": "arrival_time",
                "norm": "l1" if objective.norm == "mean" else "linf",
                "weight": objective.weight,
                "priority": objective.priority,
            }
        )
    hyperparameter = {
        "unit_system": dict(CANONICAL_UNIT_SYSTEM),
        "soc": {
            "vehicle": _homogeneous_bounds(
                ((item.minimum_soc, item.maximum_soc) for item in prepared.vehicles),
                label="vehicle",
            ),
            "mcs": _homogeneous_bounds(
                ((item.minimum_soc, item.maximum_soc) for item in prepared.mcs_units),
                label="MCS",
                empty=(0.0, 1.0),
            ),
        },
        "cluster": {
            "number": prepared.preparation_policy.requested_cluster_count,
            "seed": prepared.preparation_policy.grouping_seed,
        },
        "traffic": prepared.traffic.exogenous_profile.to_data(),
        "recharge": {
            **recharge,
            "amount": {"time": prepared.preparation_policy.charging_increment_seconds},
            "limit": {
                "stop": prepared.preparation_policy.charging_stop_limit,
                "remaining": max(1, prepared.preparation_policy.charging_path_limit),
            },
        },
        "path": {
            "allocation": False,
            "limit": {
                "geographical": prepared.preparation_policy.geographical_path_limit,
                "virtual": prepared.preparation_policy.charging_path_limit,
            },
        },
        "optimizer": {
            "window": {"number": window_count, "size": dt},
            "horizon_end_time": horizon_seconds,
            "objectives": source_objectives,
        },
    }
    traffic_model = prepared.traffic.exogenous_profile.to_data()
    grouping = {
        "schema": "canonical_prepared_group_projection_v1",
        "source_preparation_identity": prepared.identity.value,
        "groups": grouping_records,
    }
    normalized_demand = demand.drop(columns=["group_id", "representative"]).copy()
    normalized_input = {
        "unit_system": dict(CANONICAL_UNIT_SYSTEM),
        "map": base_graph,
        "demand": normalized_demand,
        "supply": supply.copy(),
    }
    input_component = fingerprint_component(
        "input",
        NORMALIZED_INPUT_SCHEMA,
        normalized_input_payload(normalized_input, base_graph=base_graph),
    )
    group_component = fingerprint_component(
        "group",
        GROUP_CERTIFICATION_SCHEMA,
        {
            "metadata": group_certification_payload(grouping),
            "membership": demand[["id", "group_id"]].copy(),
        },
    )
    path_component = fingerprint_component(
        "path",
        PATH_LIBRARY_SCHEMA,
        {
            "library": path_library_payload(library),
            "path_parameters": hyperparameter["path"],
            "recharge_parameters": hyperparameter["recharge"],
            "soc_parameters": hyperparameter["soc"],
        },
    )
    pack = {
        "unit_system": dict(CANONICAL_UNIT_SYSTEM),
        "normalized_input": normalized_input,
        "input": {
            "unit_system": dict(CANONICAL_UNIT_SYSTEM),
            "hyperparameter": hyperparameter,
            "demand": demand,
            "supply": supply,
        },
        "library": library,
        "network": {"base": base_graph, "augmented": augmented},
        "grouping": grouping,
        "traffic": prepared.traffic,
        "traffic_model": traffic_model,
        "model_components": {
            "input": input_component.metadata(),
            "group": group_component.metadata(),
            "path": path_component.metadata(),
        },
    }
    return MILPProjection(
        prepared=prepared,
        pack=pack,
        group_by_source_id=group_by_source_id,
        path_by_source_index=path_by_source_index,
        vehicle_by_source_id=vehicle_by_source_id,
        mcs_by_source_id=mcs_by_source_id,
        node_by_source_label=label_to_node,
    )


def _window_size(prepared: PreparedScenario) -> float:
    windows = prepared.departure_windows
    dt = windows[0].end_seconds - windows[0].start_seconds
    for index, window in enumerate(windows):
        if window.index != index or not math.isclose(window.start_seconds, index * dt):
            raise UnsupportedCapabilityError(
                "The MILP formulation requires zero-based, contiguous, uniform departure windows."
            )
        if not math.isclose(window.end_seconds, (index + 1) * dt):
            raise UnsupportedCapabilityError(
                "The MILP formulation requires zero-based, contiguous, uniform departure windows."
            )
    return dt


def _typed_label(value: str | int) -> str:
    return json.dumps(
        {"type": "str" if isinstance(value, str) else "int", "value": value},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _node_labels(
    prepared: PreparedScenario,
) -> tuple[dict[NodeId, str], dict[str, NodeId]]:
    forward = {
        node.id: f"node:{_typed_label(node.id.value)}"
        for node in prepared.network.nodes
    }
    reverse = {label: node for node, label in forward.items()}
    if len(reverse) != len(forward):
        raise ValidationError(
            "Canonical node identifiers collide at the MILP boundary."
        )
    return (forward, reverse)


def _base_graph(
    prepared: PreparedScenario, node_to_label: Mapping[NodeId, str]
) -> tuple[dict[EdgeId, Any], nx.DiGraph]:
    graph = nx.DiGraph()
    site_limits = dict(prepared.mcs_deployment_domain.site_limits)
    for node in prepared.network.nodes:
        fcs = [item for item in node.charging_providers if item.kind == "fcs"]
        unsupported = [
            item.kind for item in node.charging_providers if item.kind != "fcs"
        ]
        if unsupported:
            raise UnsupportedCapabilityError(
                f"The MILP formulation cannot represent fixed provider kinds {sorted(set(unsupported))!r}."
            )
        graph.add_node(
            node_to_label[node.id],
            coordinate=list(node.coordinate),
            parent=node_to_label[node.id],
            isreal=True,
            fcs={"port": sum((item.port_count for item in fcs))},
            mcs={"limit": site_limits.get(ChargingSiteId(node.id.value), 0)},
        )
    edge_by_id = {item.id: item for item in prepared.network.edges}
    seen_pairs: set[tuple[str, str]] = set()
    for edge in prepared.network.edges:
        source = node_to_label[edge.source]
        target = node_to_label[edge.target]
        pair = (source, target)
        if pair in seen_pairs:
            raise UnsupportedCapabilityError(
                "The MILP formulation cannot losslessly represent parallel directed edges."
            )
        seen_pairs.add(pair)
        implied_speed = edge.distance_m / edge.free_flow_duration_seconds
        if edge.speed_limit_m_per_second is not None and (
            not math.isclose(
                edge.speed_limit_m_per_second,
                implied_speed,
                rel_tol=_TOLERANCE,
                abs_tol=_TOLERANCE,
            )
        ):
            raise UnsupportedCapabilityError(
                "The MILP formulation requires edge free-flow duration to equal distance/speed."
            )
        attributes: dict[str, Any] = {
            "id": _typed_label(edge.id.value),
            "distance": edge.distance_m,
            "speed": implied_speed,
            "time": edge.free_flow_duration_seconds,
        }
        graph.add_edge(source, target, **attributes)
    return (edge_by_id, graph)


def _mcs_per_port_power(unit: Any) -> float:
    return mcs_per_port_power_kw(
        unit.power_kw, unit.port_count, unit.power_is_unit_total
    )


def _recharge_contract(prepared: PreparedScenario) -> dict[str, float]:
    rates: set[tuple[float, float]] = {
        (_mcs_per_port_power(item), item.charging_efficiency)
        for item in prepared.mcs_units
    }
    for path in prepared.candidate_paths:
        for action in path.charging_actions:
            unknown = set(action.eligible_provider_kinds) - {"fcs", "mcs"}
            if unknown:
                raise UnsupportedCapabilityError(
                    f"The MILP formulation cannot represent charging provider kinds {sorted(unknown)!r}."
                )
            for kind in action.eligible_provider_kinds:
                if kind == "fcs":
                    providers = [
                        provider
                        for node in prepared.network.nodes
                        for provider in node.charging_providers
                        if provider.site_id == action.site_id and provider.kind == "fcs"
                    ]
                    if not providers:
                        raise ValidationError(
                            "A candidate action references absent FCS service."
                        )
                    rates.update(
                        ((item.power_kw, item.efficiency) for item in providers)
                    )
                else:
                    if (
                        action.site_id
                        not in prepared.mcs_deployment_domain.eligible_sites
                    ):
                        raise ValidationError(
                            "A candidate action references an ineligible MCS site."
                        )
                    if not prepared.mcs_units:
                        raise ValidationError(
                            "A candidate action requires MCS service without MCS units."
                        )
                    rates.update(
                        (
                            (_mcs_per_port_power(item), item.charging_efficiency)
                            for item in prepared.mcs_units
                        )
                    )
    if not rates:
        rates.update(
            (
                (provider.power_kw, provider.efficiency)
                for node in prepared.network.nodes
                for provider in node.charging_providers
                if provider.kind == "fcs"
            )
        )
        rates.update(
            (
                (_mcs_per_port_power(item), item.charging_efficiency)
                for item in prepared.mcs_units
            )
        )
    if not rates:
        rates.add((1.0, 1.0))
    if len(rates) != 1:
        raise UnsupportedCapabilityError(
            f"The MILP formulation has one global per-port charging power and efficiency; canonical service rates are heterogeneous: {sorted(rates)!r}."
        )
    power, efficiency = next(iter(rates))
    discharge = {item.discharge_efficiency for item in prepared.mcs_units}
    if len(discharge) > 1:
        raise UnsupportedCapabilityError(
            "The MILP formulation has one global MCS discharge efficiency."
        )
    return {
        "power": power,
        "efficiency": efficiency,
        "discharge_efficiency": next(iter(discharge), 1.0),
    }


def _homogeneous_bounds(
    bounds: Any, *, label: str, empty: tuple[float, float] | None = None
) -> dict[str, float]:
    values = set(bounds)
    if not values and empty is not None:
        values.add(empty)
    if len(values) != 1:
        raise UnsupportedCapabilityError(
            f"The MILP formulation requires homogeneous global {label} SOC bounds."
        )
    minimum, maximum = next(iter(values))
    return {"min": minimum, "max": maximum}


def _path_frame(
    path: CandidatePath,
    members: tuple[Any, ...],
    *,
    edge_by_id: Mapping[EdgeId, Any],
    base_graph: nx.DiGraph,
    augmented_graph: nx.DiGraph,
    node_to_label: Mapping[NodeId, str],
    effective_power_kw: float,
    representative_id: VehicleId,
) -> pd.DataFrame:
    edges = [edge_by_id[item] for item in path.edge_ids]
    if not edges:
        raise UnsupportedCapabilityError(
            "The MILP path library requires nonempty paths."
        )
    route_nodes = [edges[0].source]
    for edge in edges:
        route_nodes.append(edge.target)
    action_by_site = {
        (type(action.site_id.value), action.site_id.value): action
        for action in path.charging_actions
    }
    if len(action_by_site) != len(path.charging_actions):
        raise UnsupportedCapabilityError(
            "The MILP formulation supports at most one charging action per route-node visit."
        )
    nodes = [node_to_label[route_nodes[0]]]
    is_real = [True]
    distance_steps = [0.0]
    charge_energy_steps = [0.0]
    real_edge_steps: list[Any | None] = [None]
    action_kinds: list[set[str]] = []
    for edge, route_node in zip(edges, route_nodes):
        action = action_by_site.get((type(route_node.value), route_node.value))
        if action is not None:
            parent = node_to_label[route_node]
            virtual = f"virtual:{parent}:{_typed_label(action.id.value)}"
            duration = 3600.0 * action.requested_energy_kwh / effective_power_kw
            charger_type = "fcs" if "fcs" in action.eligible_provider_kinds else "mcs"
            attributes = {
                "time": duration,
                "energy": action.requested_energy_kwh,
                "type": charger_type,
                "distance": 0.0,
                "capacity": base_graph.nodes[parent]["fcs"]["port"],
                "mcs_limit": base_graph.nodes[parent]["mcs"]["limit"],
            }
            if virtual not in augmented_graph:
                augmented_graph.add_node(virtual, parent=parent, isreal=False)
                augmented_graph.add_edge(parent, virtual, **attributes)
                for _, successor, data in base_graph.out_edges(parent, data=True):
                    augmented_graph.add_edge(virtual, successor, **dict(data))
            elif augmented_graph[parent][virtual] != attributes:
                raise ValidationError(
                    "Charging action identity maps to conflicting MILP semantics."
                )
            nodes.append(virtual)
            is_real.append(False)
            distance_steps.append(0.0)
            charge_energy_steps.append(action.requested_energy_kwh)
            real_edge_steps.append(None)
            action_kinds.append(set(action.eligible_provider_kinds))
        nodes.append(node_to_label[edge.target])
        is_real.append(True)
        distance_steps.append(edge.distance_m)
        charge_energy_steps.append(0.0)
        real_edge_steps.append(edge)
    duration_rows: list[list[float]] = []
    energy_rows: list[list[float]] = []
    soc_rows: list[list[float]] = []
    for member in members:
        durations = [0.0]
        energy = [member.battery_kwh * member.initial_soc]
        for index in range(1, len(nodes)):
            edge = real_edge_steps[index]
            if edge is None:
                duration = 3600.0 * charge_energy_steps[index] / effective_power_kw
                next_energy = energy[-1] + charge_energy_steps[index]
            else:
                edge_speed = edge.distance_m / edge.free_flow_duration_seconds
                duration = edge.distance_m / min(member.speed_m_per_second, edge_speed)
                next_energy = energy[-1] - edge.distance_m / member.efficiency_m_per_kwh
            durations.append(duration)
            energy.append(next_energy)
        duration_rows.append(durations)
        energy_rows.append(energy)
        soc_rows.append([value / member.battery_kwh for value in energy])
    duration_matrix = np.asarray(duration_rows, dtype=float)
    energy_matrix = np.asarray(energy_rows, dtype=float)
    soc_matrix = np.asarray(soc_rows, dtype=float)
    representative_index = next(
        (
            index
            for index, member in enumerate(members)
            if member.id == representative_id
        )
    )
    cumulative_distance = np.cumsum(np.asarray(distance_steps, dtype=float))
    kinds = set().union(*action_kinds) if action_kinds else set()
    path_type = "direct"
    if kinds == {"fcs"}:
        path_type = "fcs"
    elif kinds == {"mcs"}:
        path_type = "mcs"
    elif kinds:
        path_type = "mixed"
    return pd.DataFrame(
        {
            "node": nodes,
            "time": np.cumsum(duration_matrix.max(axis=0)),
            "energy": energy_matrix[representative_index],
            "distance": cumulative_distance,
            "type": [path_type] * len(nodes),
            "soc": soc_matrix[representative_index],
            "isreal": is_real,
            "certified_energy_min": energy_matrix.min(axis=0),
            "certified_energy_max": energy_matrix.max(axis=0),
            "certified_soc_min": soc_matrix.min(axis=0),
            "certified_soc_max": soc_matrix.max(axis=0),
        }
    )


__all__ = ["MILPProjection", "compile_prepared_scenario"]
