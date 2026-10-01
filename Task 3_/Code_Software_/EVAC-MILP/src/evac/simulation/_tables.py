from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from evac.domain import (
    ChargingProviderId,
    ChargingSiteId,
    MCSUnitId,
    NodeId,
    PreparedScenario,
)
from evac.domain.ordering import identifier_key
from evac.physics.units import mcs_per_port_power_kw

NO_MCS_UNIT_INDEX = -1
FCS_KIND_CODE = 0
MCS_KIND_CODE = 1
FORBIDDEN_QUEUE_POLICY_CODE = 0
FIFO_QUEUE_POLICY_CODE = 1


@dataclass(frozen=True, slots=True)
class _CompiledScenario:
    vehicle_battery_kwh: np.ndarray
    vehicle_initial_soc: np.ndarray
    vehicle_minimum_soc: np.ndarray
    vehicle_maximum_soc: np.ndarray
    vehicle_efficiency_m_per_kwh: np.ndarray
    vehicle_speed_m_per_second: np.ndarray
    vehicle_ids: tuple[object, ...]
    vehicle_index_of_id: dict[object, int]
    edge_distance_m: np.ndarray
    edge_speed_limit_m_per_second: np.ndarray
    edge_source_node_index: np.ndarray
    edge_target_node_index: np.ndarray
    edge_ids: tuple[object, ...]
    path_edge_offsets: np.ndarray
    path_edge_index: np.ndarray
    path_action_offsets: np.ndarray
    action_site_index: np.ndarray
    action_requested_energy_kwh: np.ndarray
    action_eligible_fcs: np.ndarray
    action_eligible_mcs: np.ndarray
    action_required_post_charge_soc: np.ndarray
    path_ids: tuple[object, ...]
    path_index_of_id: dict[object, int]
    provider_site_index: np.ndarray
    provider_kind_code: np.ndarray
    provider_port_count: np.ndarray
    provider_power_kw: np.ndarray
    provider_charging_efficiency: np.ndarray
    provider_mcs_unit_index: np.ndarray
    provider_source_to_battery_efficiency: np.ndarray
    provider_ids: tuple[ChargingProviderId, ...]
    provider_index_of_id: dict[ChargingProviderId, int]
    fcs_site_offsets: np.ndarray
    fcs_site_members: np.ndarray
    mcs_battery_kwh: np.ndarray
    mcs_initial_soc: np.ndarray
    mcs_minimum_soc: np.ndarray
    mcs_maximum_soc: np.ndarray
    mcs_unit_ids: tuple[MCSUnitId, ...]
    mcs_unit_index_of_id: dict[MCSUnitId, int]
    provider_index_of_mcs_unit: np.ndarray
    traffic_times: np.ndarray
    traffic_multipliers: np.ndarray
    tolerance: float
    queue_policy_code: int
    queue_capacity: int
    site_ids: tuple[ChargingSiteId, ...]
    site_index_of_id: dict[ChargingSiteId, int]
    node_to_site_index: np.ndarray
    site_to_node_index: np.ndarray
    node_ids: tuple[NodeId, ...]
    node_index_of_id: dict[NodeId, int]


def compiled_scenario(prepared: PreparedScenario) -> _CompiledScenario:
    return _build_compiled_scenario(prepared)


def _build_compiled_scenario(prepared: PreparedScenario) -> _CompiledScenario:
    nodes = sorted(prepared.network.nodes, key=lambda item: identifier_key(item.id))
    node_ids = tuple((item.id for item in nodes))
    node_index_of_id = {item: index for index, item in enumerate(node_ids)}
    vehicles = sorted(prepared.vehicles, key=lambda item: identifier_key(item.id))
    vehicle_ids = tuple((item.id for item in vehicles))
    vehicle_index_of_id = {item: index for index, item in enumerate(vehicle_ids)}
    edges = sorted(prepared.network.edges, key=lambda item: identifier_key(item.id))
    edge_ids = tuple((item.id for item in edges))
    edge_index_of_id = {item: index for index, item in enumerate(edge_ids)}
    site_values: set[ChargingSiteId] = set()
    for node in nodes:
        for provider in node.charging_providers:
            site_values.add(provider.site_id)
    for path in prepared.candidate_paths:
        for action in path.charging_actions:
            site_values.add(action.site_id)
    site_values.update(prepared.mcs_deployment_domain.eligible_sites)
    site_ids = tuple(sorted(site_values, key=identifier_key))
    site_index_of_id = {item: index for index, item in enumerate(site_ids)}
    node_value_to_site_index = {
        (type(site.value), site.value): index for index, site in enumerate(site_ids)
    }
    node_to_site_index = np.full(len(node_ids), -1, dtype=np.int64)
    site_to_node_index = np.full(len(site_ids), -1, dtype=np.int64)
    for node_index, node in enumerate(nodes):
        site_index = node_value_to_site_index.get((type(node.id.value), node.id.value))
        if site_index is not None:
            node_to_site_index[node_index] = site_index
            site_to_node_index[site_index] = node_index
    vehicle_battery_kwh = np.array([v.battery_kwh for v in vehicles], dtype=np.float64)
    vehicle_initial_soc = np.array([v.initial_soc for v in vehicles], dtype=np.float64)
    vehicle_minimum_soc = np.array([v.minimum_soc for v in vehicles], dtype=np.float64)
    vehicle_maximum_soc = np.array([v.maximum_soc for v in vehicles], dtype=np.float64)
    vehicle_efficiency_m_per_kwh = np.array(
        [v.efficiency_m_per_kwh for v in vehicles], dtype=np.float64
    )
    vehicle_speed_m_per_second = np.array(
        [v.speed_m_per_second for v in vehicles], dtype=np.float64
    )
    edge_distance_m = np.array([e.distance_m for e in edges], dtype=np.float64)
    edge_speed_limit_m_per_second = np.array(
        [
            e.speed_limit_m_per_second
            if e.speed_limit_m_per_second is not None
            else np.inf
            for e in edges
        ],
        dtype=np.float64,
    )
    edge_source_node_index = np.array(
        [node_index_of_id[e.source] for e in edges], dtype=np.int64
    )
    edge_target_node_index = np.array(
        [node_index_of_id[e.target] for e in edges], dtype=np.int64
    )
    profile = prepared.traffic.exogenous_profile
    traffic_times = np.asarray(
        [p.elapsed_seconds for p in profile.points], dtype=np.float64
    )
    traffic_multipliers = np.asarray(
        [p.multiplier for p in profile.points], dtype=np.float64
    )
    paths = sorted(prepared.candidate_paths, key=lambda item: identifier_key(item.id))
    path_ids = tuple((item.id for item in paths))
    path_index_of_id = {item: index for index, item in enumerate(path_ids)}
    path_edge_offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    path_action_offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    path_edge_index_list: list[int] = []
    action_site_index_list: list[int] = []
    action_requested_energy_kwh_list: list[float] = []
    action_eligible_fcs_list: list[bool] = []
    action_eligible_mcs_list: list[bool] = []
    action_required_post_charge_soc_list: list[float] = []
    for index, path in enumerate(paths):
        for edge_id in path.edge_ids:
            path_edge_index_list.append(edge_index_of_id[edge_id])
        path_edge_offsets[index + 1] = len(path_edge_index_list)
        for action in path.charging_actions:
            action_site_index_list.append(site_index_of_id[action.site_id])
            action_requested_energy_kwh_list.append(action.requested_energy_kwh)
            action_eligible_fcs_list.append("fcs" in action.eligible_provider_kinds)
            action_eligible_mcs_list.append("mcs" in action.eligible_provider_kinds)
            action_required_post_charge_soc_list.append(action.required_post_charge_soc)
        path_action_offsets[index + 1] = len(action_site_index_list)
    path_edge_index = np.array(path_edge_index_list, dtype=np.int64)
    action_site_index = np.array(action_site_index_list, dtype=np.int64)
    action_requested_energy_kwh = np.array(
        action_requested_energy_kwh_list, dtype=np.float64
    )
    action_eligible_fcs = np.array(action_eligible_fcs_list, dtype=np.bool_)
    action_eligible_mcs = np.array(action_eligible_mcs_list, dtype=np.bool_)
    action_required_post_charge_soc = np.array(
        action_required_post_charge_soc_list, dtype=np.float64
    )
    mcs_units = sorted(prepared.mcs_units, key=lambda item: identifier_key(item.id))
    mcs_unit_ids = tuple((item.id for item in mcs_units))
    mcs_unit_index_of_id = {item: index for index, item in enumerate(mcs_unit_ids)}
    mcs_battery_kwh = np.array([u.battery_kwh for u in mcs_units], dtype=np.float64)
    mcs_initial_soc = np.array([u.initial_soc for u in mcs_units], dtype=np.float64)
    mcs_minimum_soc = np.array([u.minimum_soc for u in mcs_units], dtype=np.float64)
    mcs_maximum_soc = np.array([u.maximum_soc for u in mcs_units], dtype=np.float64)
    fixed_entries = [
        (provider.id, node, provider)
        for node in nodes
        for provider in node.charging_providers
    ]
    mcs_provider_entries = [
        (ChargingProviderId(f"mcs:{unit.id.value}"), unit) for unit in mcs_units
    ]
    provider_ids = tuple(
        sorted(
            tuple((item[0] for item in fixed_entries))
            + tuple((item[0] for item in mcs_provider_entries)),
            key=identifier_key,
        )
    )
    fixed_by_id = {item[0]: (item[1], item[2]) for item in fixed_entries}
    mcs_by_id = {item[0]: item[1] for item in mcs_provider_entries}
    n_providers = len(provider_ids)
    provider_site_index = np.zeros(n_providers, dtype=np.int64)
    provider_kind_code = np.zeros(n_providers, dtype=np.int64)
    provider_port_count = np.zeros(n_providers, dtype=np.int64)
    provider_power_kw = np.zeros(n_providers, dtype=np.float64)
    provider_charging_efficiency = np.zeros(n_providers, dtype=np.float64)
    provider_mcs_unit_index = np.full(n_providers, NO_MCS_UNIT_INDEX, dtype=np.int64)
    provider_source_to_battery_efficiency = np.ones(n_providers, dtype=np.float64)
    provider_index_of_mcs_unit = np.full(len(mcs_units), -1, dtype=np.int64)
    for provider_index, provider_id in enumerate(provider_ids):
        if provider_id in fixed_by_id:
            node, spec = fixed_by_id[provider_id]
            provider_site_index[provider_index] = site_index_of_id[spec.site_id]
            provider_kind_code[provider_index] = FCS_KIND_CODE
            provider_port_count[provider_index] = spec.port_count
            provider_power_kw[provider_index] = spec.power_kw
            provider_charging_efficiency[provider_index] = spec.efficiency
        else:
            unit = mcs_by_id[provider_id]
            unit_index = mcs_unit_index_of_id[unit.id]
            provider_site_index[provider_index] = -1
            provider_kind_code[provider_index] = MCS_KIND_CODE
            provider_port_count[provider_index] = unit.port_count
            provider_power_kw[provider_index] = mcs_per_port_power_kw(
                unit.power_kw, unit.port_count, unit.power_is_unit_total
            )
            provider_charging_efficiency[provider_index] = unit.charging_efficiency
            provider_mcs_unit_index[provider_index] = unit_index
            provider_source_to_battery_efficiency[provider_index] = (
                unit.discharge_efficiency * unit.charging_efficiency
            )
            provider_index_of_mcs_unit[unit_index] = provider_index
    fcs_rows = [
        index for index, kind in enumerate(provider_kind_code) if kind == FCS_KIND_CODE
    ]
    fcs_rows.sort(
        key=lambda index: (
            provider_site_index[index],
            identifier_key(provider_ids[index]),
        )
    )
    fcs_site_offsets = np.zeros(len(site_ids) + 1, dtype=np.int64)
    for index in fcs_rows:
        fcs_site_offsets[provider_site_index[index] + 1] += 1
    fcs_site_offsets = np.cumsum(fcs_site_offsets)
    fcs_site_members = np.array(fcs_rows, dtype=np.int64)
    tolerance = prepared.numerical_policy.event_time_tolerance_seconds
    queue_policy_code = (
        FORBIDDEN_QUEUE_POLICY_CODE
        if prepared.charging_policy.queueing.value == "forbidden"
        else FIFO_QUEUE_POLICY_CODE
    )
    queue_capacity = (
        -1
        if prepared.charging_policy.queue_capacity_vehicles is None
        else prepared.charging_policy.queue_capacity_vehicles
    )
    return _CompiledScenario(
        vehicle_battery_kwh=vehicle_battery_kwh,
        vehicle_initial_soc=vehicle_initial_soc,
        vehicle_minimum_soc=vehicle_minimum_soc,
        vehicle_maximum_soc=vehicle_maximum_soc,
        vehicle_efficiency_m_per_kwh=vehicle_efficiency_m_per_kwh,
        vehicle_speed_m_per_second=vehicle_speed_m_per_second,
        vehicle_ids=vehicle_ids,
        vehicle_index_of_id=vehicle_index_of_id,
        edge_distance_m=edge_distance_m,
        edge_speed_limit_m_per_second=edge_speed_limit_m_per_second,
        edge_source_node_index=edge_source_node_index,
        edge_target_node_index=edge_target_node_index,
        edge_ids=edge_ids,
        path_edge_offsets=path_edge_offsets,
        path_edge_index=path_edge_index,
        path_action_offsets=path_action_offsets,
        action_site_index=action_site_index,
        action_requested_energy_kwh=action_requested_energy_kwh,
        action_eligible_fcs=action_eligible_fcs,
        action_eligible_mcs=action_eligible_mcs,
        action_required_post_charge_soc=action_required_post_charge_soc,
        path_ids=path_ids,
        path_index_of_id=path_index_of_id,
        provider_site_index=provider_site_index,
        provider_kind_code=provider_kind_code,
        provider_port_count=provider_port_count,
        provider_power_kw=provider_power_kw,
        provider_charging_efficiency=provider_charging_efficiency,
        provider_mcs_unit_index=provider_mcs_unit_index,
        provider_source_to_battery_efficiency=provider_source_to_battery_efficiency,
        provider_ids=provider_ids,
        provider_index_of_id={item: index for index, item in enumerate(provider_ids)},
        fcs_site_offsets=fcs_site_offsets,
        fcs_site_members=fcs_site_members,
        mcs_battery_kwh=mcs_battery_kwh,
        mcs_initial_soc=mcs_initial_soc,
        mcs_minimum_soc=mcs_minimum_soc,
        mcs_maximum_soc=mcs_maximum_soc,
        mcs_unit_ids=mcs_unit_ids,
        mcs_unit_index_of_id=mcs_unit_index_of_id,
        provider_index_of_mcs_unit=provider_index_of_mcs_unit,
        traffic_times=traffic_times,
        traffic_multipliers=traffic_multipliers,
        tolerance=tolerance,
        queue_policy_code=queue_policy_code,
        queue_capacity=queue_capacity,
        site_ids=site_ids,
        site_index_of_id=site_index_of_id,
        node_to_site_index=node_to_site_index,
        site_to_node_index=site_to_node_index,
        node_ids=node_ids,
        node_index_of_id=node_index_of_id,
    )


__all__ = ["compiled_scenario"]
