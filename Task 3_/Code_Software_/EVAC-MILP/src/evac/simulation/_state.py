from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from evac.simulation._tables import _CompiledScenario

V_F_SOC = 0
V_F_COMPLETION_SECONDS = 1
V_F_SESSION_REMAINING_ENERGY_KWH = 2
V_F_SESSION_REQUESTED_ENERGY_KWH = 3
N_VEHICLE_F64 = 4
V_I_EDGE_INDEX = 0
V_I_ACTION_INDEX = 1
V_I_ACTIVE_PROVIDER = 2
V_I_TERMINAL_REASON_CODE = 3
V_I_SESSION_TOKEN = 4
N_VEHICLE_I64 = 5
V_B_WAITING = 0
V_B_FINISHED = 1
V_B_FAILED = 2
V_B_QUEUED = 3
V_B_QUEUED_EVENT = 4
N_VEHICLE_BOOL = 5
P_F_LAST_SETTLE = 0
N_PROVIDER_F64 = 1
P_I_SITE_INDEX = 0
P_I_ACTIVE_SESSIONS = 1
N_PROVIDER_I64 = 2
P_B_AVAILABLE = 0
N_PROVIDER_BOOL = 1
M_F_UNRESERVED_SOURCE_ENERGY_KWH = 0
M_F_DELIVERED_ENERGY_KWH = 1
N_MCS_F64 = 2
M_I_LIFECYCLE_STATE = 0
M_I_LOCATION_NODE = 1
N_MCS_I64 = 2
S_I_HEAP_SIZE = 0
S_I_SEQUENCE = 1
S_I_NEXT_FREE_SLOT = 2
S_I_TERMINAL_STATE_COUNT = 3
S_I_LOG_COUNT = 4
N_SCALARS_I64 = 5
S_F_LAST_EVENT_SECONDS = 0
N_SCALARS_F64 = 1


@dataclass(frozen=True, slots=True)
class SimulationCapacities:
    n_vehicles: int
    n_edges: int
    n_providers: int
    n_mcs: int
    n_sites: int
    n_intervals: int
    max_port_count: int
    heap_capacity: int
    request_capacity: int
    log_capacity: int


def _base_capacities(tables: _CompiledScenario) -> tuple[int, int, int, int, int, int]:
    n_vehicles = len(tables.vehicle_ids)
    n_edges = len(tables.edge_ids)
    n_providers = len(tables.provider_ids)
    n_mcs = len(tables.mcs_unit_ids)
    n_sites = len(tables.site_ids)
    max_port_count = 1
    for count in tables.provider_port_count:
        if int(count) > max_port_count:
            max_port_count = int(count)
    return (n_vehicles, n_edges, n_providers, n_mcs, n_sites, max_port_count)


def _capacities_from_totals(
    tables: _CompiledScenario,
    *,
    n_vehicles: int,
    n_edges: int,
    n_providers: int,
    n_mcs: int,
    n_sites: int,
    max_port_count: int,
    n_intervals: int,
    sum_edges: int,
    sum_actions: int,
    record_events: bool,
) -> SimulationCapacities:
    del tables
    heap_capacity = (
        n_vehicles + sum_edges + sum_actions * 2 * max_port_count + 5 * n_intervals
    )
    if heap_capacity < 1:
        heap_capacity = 1
    log_capacity = (
        n_vehicles * (1 + 3) + 4 * sum_edges + 5 * sum_actions + 5 * n_intervals
        if record_events
        else 1
    )
    if log_capacity < 1:
        log_capacity = 1
    request_capacity = 1 + sum_actions
    return SimulationCapacities(
        n_vehicles=n_vehicles,
        n_edges=n_edges,
        n_providers=n_providers,
        n_mcs=n_mcs,
        n_sites=n_sites,
        n_intervals=n_intervals,
        max_port_count=max_port_count,
        heap_capacity=heap_capacity,
        request_capacity=request_capacity,
        log_capacity=log_capacity,
    )


def run_capacities(
    tables: _CompiledScenario, run_inputs, *, record_events: bool
) -> SimulationCapacities:
    n_vehicles, n_edges, n_providers, n_mcs, n_sites, max_port_count = _base_capacities(
        tables
    )
    sum_edges = 0
    sum_actions = 0
    for v in range(n_vehicles):
        p = int(run_inputs.path_index[v])
        sum_edges += int(tables.path_edge_offsets[p + 1] - tables.path_edge_offsets[p])
        sum_actions += int(
            tables.path_action_offsets[p + 1] - tables.path_action_offsets[p]
        )
    n_intervals = int(run_inputs.unit_index.shape[0])
    return _capacities_from_totals(
        tables,
        n_vehicles=n_vehicles,
        n_edges=n_edges,
        n_providers=n_providers,
        n_mcs=n_mcs,
        n_sites=n_sites,
        max_port_count=max_port_count,
        n_intervals=n_intervals,
        sum_edges=sum_edges,
        sum_actions=sum_actions,
        record_events=record_events,
    )


@dataclass(slots=True)
class SimulationState:
    capacities: SimulationCapacities
    vehicle_f64: np.ndarray
    vehicle_i64: np.ndarray
    vehicle_bool: np.ndarray
    provider_f64: np.ndarray
    provider_i64: np.ndarray
    provider_bool: np.ndarray
    provider_session_vehicles: np.ndarray
    mcs_f64: np.ndarray
    mcs_i64: np.ndarray
    request_vehicle_index: np.ndarray
    request_action_slot: np.ndarray
    request_binding_slot: np.ndarray
    request_time: np.ndarray
    request_next: np.ndarray
    queue_head: np.ndarray
    queue_tail: np.ndarray
    dirty_site: np.ndarray
    active_edge_count: np.ndarray
    charge_req_vehicle: np.ndarray
    charge_req_action_slot: np.ndarray
    charge_req_binding_slot: np.ndarray
    charge_req_time: np.ndarray
    road_req_vehicle: np.ndarray
    road_req_edge_progress: np.ndarray
    scratch_candidates: np.ndarray
    scratch_road_order: np.ndarray
    scratch_edge_of: np.ndarray
    scratch_free_flow: np.ndarray
    scratch_charge_order: np.ndarray
    h_time: np.ndarray
    h_key: np.ndarray
    h_kind: np.ndarray
    h_payload: np.ndarray
    b_time: np.ndarray
    b_key: np.ndarray
    b_kind: np.ndarray
    b_payload: np.ndarray
    b_order: np.ndarray
    log_time: np.ndarray
    log_kind: np.ndarray
    log_namespace: np.ndarray
    log_subject: np.ndarray
    log_v: np.ndarray
    scalars_i64: np.ndarray
    scalars_f64: np.ndarray

    @classmethod
    def allocate(cls, capacities: SimulationCapacities) -> "SimulationState":
        c = capacities
        return cls(
            capacities=c,
            vehicle_f64=np.empty((c.n_vehicles, N_VEHICLE_F64), dtype=np.float64),
            vehicle_i64=np.empty((c.n_vehicles, N_VEHICLE_I64), dtype=np.int64),
            vehicle_bool=np.empty((c.n_vehicles, N_VEHICLE_BOOL), dtype=np.bool_),
            provider_f64=np.empty((c.n_providers, N_PROVIDER_F64), dtype=np.float64),
            provider_i64=np.empty((c.n_providers, N_PROVIDER_I64), dtype=np.int64),
            provider_bool=np.empty((c.n_providers, N_PROVIDER_BOOL), dtype=np.bool_),
            provider_session_vehicles=np.empty(
                (c.n_providers, c.max_port_count), dtype=np.int64
            ),
            mcs_f64=np.empty((c.n_mcs, N_MCS_F64), dtype=np.float64),
            mcs_i64=np.empty((c.n_mcs, N_MCS_I64), dtype=np.int64),
            request_vehicle_index=np.empty(c.request_capacity, dtype=np.int64),
            request_action_slot=np.empty(c.request_capacity, dtype=np.int64),
            request_binding_slot=np.empty(c.request_capacity, dtype=np.int64),
            request_time=np.empty(c.request_capacity, dtype=np.float64),
            request_next=np.empty(c.request_capacity, dtype=np.int64),
            queue_head=np.empty(c.n_sites, dtype=np.int64),
            queue_tail=np.empty(c.n_sites, dtype=np.int64),
            dirty_site=np.empty(c.n_sites, dtype=np.bool_),
            active_edge_count=np.empty(c.n_edges, dtype=np.int64),
            charge_req_vehicle=np.empty(c.n_vehicles, dtype=np.int64),
            charge_req_action_slot=np.empty(c.n_vehicles, dtype=np.int64),
            charge_req_binding_slot=np.empty(c.n_vehicles, dtype=np.int64),
            charge_req_time=np.empty(c.n_vehicles, dtype=np.float64),
            road_req_vehicle=np.empty(c.n_vehicles, dtype=np.int64),
            road_req_edge_progress=np.empty(c.n_vehicles, dtype=np.int64),
            scratch_candidates=np.empty(c.n_providers, dtype=np.int64),
            scratch_road_order=np.empty(c.n_vehicles, dtype=np.int64),
            scratch_edge_of=np.empty(c.n_vehicles, dtype=np.int64),
            scratch_free_flow=np.empty(c.n_vehicles, dtype=np.float64),
            scratch_charge_order=np.empty(c.n_vehicles, dtype=np.int64),
            h_time=np.empty(c.heap_capacity, dtype=np.float64),
            h_key=np.empty(c.heap_capacity, dtype=np.int64),
            h_kind=np.empty(c.heap_capacity, dtype=np.int64),
            h_payload=np.empty(c.heap_capacity, dtype=np.int64),
            b_time=np.empty(c.heap_capacity, dtype=np.float64),
            b_key=np.empty(c.heap_capacity, dtype=np.int64),
            b_kind=np.empty(c.heap_capacity, dtype=np.int64),
            b_payload=np.empty(c.heap_capacity, dtype=np.int64),
            b_order=np.empty(c.heap_capacity, dtype=np.int64),
            log_time=np.empty(c.log_capacity, dtype=np.float64),
            log_kind=np.empty(c.log_capacity, dtype=np.int64),
            log_namespace=np.empty(c.log_capacity, dtype=np.int64),
            log_subject=np.empty(c.log_capacity, dtype=np.int64),
            log_v=np.empty((c.log_capacity, 5), dtype=np.float64),
            scalars_i64=np.empty(N_SCALARS_I64, dtype=np.int64),
            scalars_f64=np.empty(N_SCALARS_F64, dtype=np.float64),
        )


__all__ = ["SimulationCapacities", "SimulationState", "run_capacities"]
