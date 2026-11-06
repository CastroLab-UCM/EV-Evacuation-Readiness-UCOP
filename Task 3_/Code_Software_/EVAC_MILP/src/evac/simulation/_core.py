from __future__ import annotations
import numba
from numba.extending import register_jitable
import numpy as np
from evac.physics.traffic_kernel import traffic_exit_time_kernel
from evac.simulation._state import (
    M_F_DELIVERED_ENERGY_KWH,
    M_F_UNRESERVED_SOURCE_ENERGY_KWH,
    M_I_LIFECYCLE_STATE,
    M_I_LOCATION_NODE,
    P_B_AVAILABLE,
    P_F_LAST_SETTLE,
    P_I_ACTIVE_SESSIONS,
    P_I_SITE_INDEX,
    S_F_LAST_EVENT_SECONDS,
    S_I_HEAP_SIZE,
    S_I_LOG_COUNT,
    S_I_NEXT_FREE_SLOT,
    S_I_SEQUENCE,
    S_I_TERMINAL_STATE_COUNT,
    V_B_FAILED,
    V_B_FINISHED,
    V_B_QUEUED,
    V_B_QUEUED_EVENT,
    V_B_WAITING,
    V_F_COMPLETION_SECONDS,
    V_F_SESSION_REMAINING_ENERGY_KWH,
    V_F_SESSION_REQUESTED_ENERGY_KWH,
    V_F_SOC,
    V_I_ACTION_INDEX,
    V_I_ACTIVE_PROVIDER,
    V_I_EDGE_INDEX,
    V_I_SESSION_TOKEN,
    V_I_TERMINAL_REASON_CODE,
)

KIND_CHARGE_FINISH = 0
KIND_MCS_ARRIVAL = 3
KIND_ROAD_ARRIVAL = 6
KIND_VEHICLE_DEPARTURE = 7
PHASE_CHARGE_FINISH = 2
PHASE_MCS_LIFECYCLE = 3
PHASE_ROAD_ARRIVAL = 4
PHASE_VEHICLE_DEPARTURE = 5
MCS_KIND_PRIORITY_ARRIVAL = 2
FCS_KIND_CODE = 0
MCS_KIND_CODE = 1
PROVIDER_KIND_FCS_BINDING_CODE = 1
PROVIDER_KIND_MCS_BINDING_CODE = 2
PROVIDER_ID_BINDING_CODE = 3
LIFECYCLE_UNAVAILABLE = 0
LIFECYCLE_ACTIVE_AT_SITE = 3
TERMINAL_FINISHED = 0
TERMINAL_CHARGING_ENERGY_UNAVAILABLE = 1
TERMINAL_CHARGING_PORT_UNAVAILABLE = 2
TERMINAL_CHARGING_QUEUE_OVERFLOW = 3
TERMINAL_NONE = -1
QUEUE_POLICY_FORBIDDEN = 0
QUEUE_POLICY_FIFO = 1
EVENT_VEHICLE_DEPARTED = 0
EVENT_EDGE_ENTERED = 1
EVENT_EDGE_ARRIVED = 2
EVENT_EDGE_ENTRY = 3
EVENT_EDGE_EXIT = 4
EVENT_CHARGING_REQUESTED = 5
EVENT_CHARGING_QUEUED = 6
EVENT_CHARGING_STARTED = 7
EVENT_CHARGING_FINISHED = 8
EVENT_CHARGING_REJECTED = 9
EVENT_VEHICLE_FINISHED = 10
EVENT_VEHICLE_ARRIVED = 11
EVENT_VEHICLE_UNFINISHED = 12
EVENT_MCS_DEPLOYED = 14
NAMESPACE_VEHICLE = 0
NAMESPACE_EDGE = 1
NAMESPACE_CHARGING_SITE = 2
NAMESPACE_MCS_UNIT = 3
ERROR_NONE = 0
ERROR_MISSING_TERMINAL_REASON = 1
ERROR_DUPLICATE_QUEUE_JOIN = 4
ERROR_CHARGE_OVERFILL = 5
ERROR_MCS_SESSION_EXCEEDS_RESERVE = 6
ERROR_SETTLE_TIME_BACKWARDS = 7
ERROR_CHARGE_EXCEEDS_MAX_SOC = 8
ERROR_ROAD_ADMISSION_BELOW_MINIMUM_SOC = 9
ERROR_CAPACITY_EXCEEDED = 11
ERROR_KEY_RANGE = 12
PHASE_SHIFT = 56
RANK_SHIFT = 32
PAYLOAD_SHIFT = 32
LOW32_MASK = (1 << 32) - 1
RANK_MODULUS = 1 << 24
SEQUENCE_MODULUS = 1 << 32
_NAN = np.nan


@register_jitable
def _pack_key(phase, rank, sequence):
    return phase << PHASE_SHIFT | rank << RANK_SHIFT | sequence


@register_jitable
def _pack_payload(pa, pb):
    return pa << PAYLOAD_SHIFT | pb + 1


@register_jitable
def _unpack_payload(payload):
    return (payload >> PAYLOAD_SHIFT, (payload & LOW32_MASK) - 1)


@register_jitable
def _heap_less(h_time, h_key, i, j):
    if h_time[i] != h_time[j]:
        return h_time[i] < h_time[j]
    return h_key[i] < h_key[j]


@register_jitable
def _heap_swap(h_time, h_key, h_kind, h_payload, i, j):
    h_time[i], h_time[j] = (h_time[j], h_time[i])
    h_key[i], h_key[j] = (h_key[j], h_key[i])
    h_kind[i], h_kind[j] = (h_kind[j], h_kind[i])
    h_payload[i], h_payload[j] = (h_payload[j], h_payload[i])


@register_jitable
def _heap_push(
    h_time, h_key, h_kind, h_payload, size, capacity, time, key, kind, payload
):
    if size == capacity:
        return (size, ERROR_CAPACITY_EXCEEDED)
    i = size
    h_time[i] = time
    h_key[i] = key
    h_kind[i] = kind
    h_payload[i] = payload
    size += 1
    while i > 0:
        parent = (i - 1) // 2
        if not _heap_less(h_time, h_key, i, parent):
            break
        _heap_swap(h_time, h_key, h_kind, h_payload, i, parent)
        i = parent
    return (size, ERROR_NONE)


@register_jitable
def _heap_pop(h_time, h_key, h_kind, h_payload, size):
    time = h_time[0]
    key = h_key[0]
    kind = h_kind[0]
    payload = h_payload[0]
    size -= 1
    if size > 0:
        h_time[0] = h_time[size]
        h_key[0] = h_key[size]
        h_kind[0] = h_kind[size]
        h_payload[0] = h_payload[size]
        _heap_sift_down(h_time, h_key, h_kind, h_payload, 0, size)
    return (time, key, kind, payload, size)


@register_jitable
def _heap_sift_down(h_time, h_key, h_kind, h_payload, i, size):
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and _heap_less(h_time, h_key, left, smallest):
            smallest = left
        if right < size and _heap_less(h_time, h_key, right, smallest):
            smallest = right
        if smallest == i:
            break
        _heap_swap(h_time, h_key, h_kind, h_payload, i, smallest)
        i = smallest


@register_jitable
def _heapify(h_time, h_key, h_kind, h_payload, size):
    i = size // 2 - 1
    while i >= 0:
        _heap_sift_down(h_time, h_key, h_kind, h_payload, i, size)
        i -= 1


@register_jitable
def _log_event(
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
    time_seconds,
    kind_code,
    namespace_code,
    subject_index,
    v0,
    v1,
    v2,
    v3,
    v4,
):
    if not record_events:
        return (log_count, ERROR_NONE)
    if log_count == log_capacity:
        return (log_count, ERROR_CAPACITY_EXCEEDED)
    log_time[log_count] = time_seconds
    log_kind[log_count] = kind_code
    log_namespace[log_count] = namespace_code
    log_subject[log_count] = subject_index
    log_v[log_count, 0] = v0
    log_v[log_count, 1] = v1
    log_v[log_count, 2] = v2
    log_v[log_count, 3] = v3
    log_v[log_count, 4] = v4
    log_count += 1
    return (log_count, ERROR_NONE)


@register_jitable
def _current_node_index(
    path_edge_offsets,
    path_edge_index,
    edge_source_node_index,
    edge_target_node_index,
    path_index,
    edge_progress,
):
    start = path_edge_offsets[path_index]
    end = path_edge_offsets[path_index + 1]
    n_edges = end - start
    if edge_progress < n_edges:
        return edge_source_node_index[path_edge_index[start + edge_progress]]
    return edge_target_node_index[path_edge_index[end - 1]]


@register_jitable
def _mcs_source_shortfall(
    requested_energy_kwh, discharge_efficiency, unreserved_source_energy_kwh
):
    required = requested_energy_kwh / discharge_efficiency
    shortfall = required - unreserved_source_energy_kwh
    if shortfall > 0.0:
        return shortfall
    return 0.0


@register_jitable
def _eligible_candidates_at_site(
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_slot,
    binding_slot,
    site_index,
    provider_site_index,
    provider_available,
    fcs_site_offsets,
    fcs_site_members,
    provider_mcs_unit_index,
    scratch_candidates,
):
    count = 0
    mode = binding_mode_code[binding_slot]
    if mode == PROVIDER_ID_BINDING_CODE:
        provider_index = bound_provider_index[binding_slot]
        if (
            provider_site_index[provider_index] == site_index
            and provider_available[provider_index]
        ):
            scratch_candidates[0] = provider_index
            count = 1
        return count
    want_fcs = mode == PROVIDER_KIND_FCS_BINDING_CODE
    want_mcs = mode == PROVIDER_KIND_MCS_BINDING_CODE
    if want_fcs:
        start = fcs_site_offsets[site_index]
        end = fcs_site_offsets[site_index + 1]
        for i in range(start, end):
            provider_index = fcs_site_members[i]
            if provider_available[provider_index]:
                scratch_candidates[count] = provider_index
                count += 1
    if want_mcs:
        n_providers = provider_site_index.shape[0]
        for provider_index in range(n_providers):
            if (
                provider_mcs_unit_index[provider_index] != -1
                and provider_site_index[provider_index] == site_index
                and provider_available[provider_index]
            ):
                scratch_candidates[count] = provider_index
                count += 1
    return count


@register_jitable
def _select_server(
    action_slot,
    binding_slot,
    site_index,
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_requested_energy_kwh,
    provider_site_index,
    provider_kind_code,
    provider_available,
    provider_port_count,
    provider_active_sessions,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    mcs_unreserved_source_energy_kwh,
    fcs_site_offsets,
    fcs_site_members,
    scratch_candidates,
):
    count = _eligible_candidates_at_site(
        binding_mode_code,
        bound_provider_index,
        action_eligible_fcs,
        action_eligible_mcs,
        action_slot,
        binding_slot,
        site_index,
        provider_site_index,
        provider_available,
        fcs_site_offsets,
        fcs_site_members,
        provider_mcs_unit_index,
        scratch_candidates,
    )
    for i in range(count):
        provider_index = scratch_candidates[i]
        if (
            provider_kind_code[provider_index] == FCS_KIND_CODE
            and provider_active_sessions[provider_index]
            < provider_port_count[provider_index]
        ):
            return provider_index
    for i in range(count):
        provider_index = scratch_candidates[i]
        if provider_kind_code[provider_index] != MCS_KIND_CODE:
            continue
        if (
            provider_active_sessions[provider_index]
            >= provider_port_count[provider_index]
        ):
            continue
        unit_index = provider_mcs_unit_index[provider_index]
        shortfall = _mcs_source_shortfall(
            action_requested_energy_kwh[action_slot],
            provider_source_to_battery_efficiency[provider_index],
            mcs_unreserved_source_energy_kwh[unit_index],
        )
        if shortfall <= 0.0:
            return provider_index
    return -1


@register_jitable
def _is_stranded(
    action_slot,
    binding_slot,
    site_index,
    time_seconds,
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_requested_energy_kwh,
    provider_site_index,
    provider_kind_code,
    provider_available,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    mcs_unreserved_source_energy_kwh,
    fcs_site_offsets,
    fcs_site_members,
    scratch_candidates,
):
    count = _eligible_candidates_at_site(
        binding_mode_code,
        bound_provider_index,
        action_eligible_fcs,
        action_eligible_mcs,
        action_slot,
        binding_slot,
        site_index,
        provider_site_index,
        provider_available,
        fcs_site_offsets,
        fcs_site_members,
        provider_mcs_unit_index,
        scratch_candidates,
    )
    for i in range(count):
        if provider_kind_code[scratch_candidates[i]] == FCS_KIND_CODE:
            return False
    for i in range(count):
        provider_index = scratch_candidates[i]
        if provider_kind_code[provider_index] != MCS_KIND_CODE:
            continue
        unit_index = provider_mcs_unit_index[provider_index]
        shortfall = _mcs_source_shortfall(
            action_requested_energy_kwh[action_slot],
            provider_source_to_battery_efficiency[provider_index],
            mcs_unreserved_source_energy_kwh[unit_index],
        )
        if shortfall <= 0.0:
            return False
    return True


@register_jitable
def _advance(
    v,
    time_seconds,
    path_index,
    path_edge_offsets,
    path_edge_index,
    path_action_offsets,
    vehicle_action_offsets,
    action_site_index,
    edge_source_node_index,
    edge_target_node_index,
    node_to_site_index,
    vehicle_edge_index,
    vehicle_action_index,
    vehicle_waiting,
    vehicle_finished,
    vehicle_active_provider,
    vehicle_terminal_reason_code,
    vehicle_completion_seconds,
    vehicle_soc,
    charge_req_vehicle,
    charge_req_action_slot,
    charge_req_binding_slot,
    charge_req_time,
    charge_req_count,
    road_req_vehicle,
    road_req_edge_progress,
    road_req_count,
    terminal_state_count,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if vehicle_finished[v] or vehicle_waiting[v] or vehicle_active_provider[v] != -1:
        return (
            ERROR_NONE,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            log_count,
        )
    p = path_index[v]
    action_start = path_action_offsets[p]
    action_end = path_action_offsets[p + 1]
    action_progress = vehicle_action_index[v]
    if action_start + action_progress < action_end:
        action_slot = action_start + action_progress
        node_index = _current_node_index(
            path_edge_offsets,
            path_edge_index,
            edge_source_node_index,
            edge_target_node_index,
            p,
            vehicle_edge_index[v],
        )
        if node_to_site_index[node_index] == action_site_index[action_slot]:
            charge_req_vehicle[charge_req_count] = v
            charge_req_action_slot[charge_req_count] = action_slot
            charge_req_binding_slot[charge_req_count] = (
                vehicle_action_offsets[v] + action_progress
            )
            charge_req_time[charge_req_count] = time_seconds
            charge_req_count += 1
            vehicle_waiting[v] = True
            return (
                ERROR_NONE,
                charge_req_count,
                road_req_count,
                terminal_state_count,
                log_count,
            )
    edge_start = path_edge_offsets[p]
    edge_end = path_edge_offsets[p + 1]
    if edge_start + vehicle_edge_index[v] < edge_end:
        road_req_vehicle[road_req_count] = v
        road_req_edge_progress[road_req_count] = vehicle_edge_index[v]
        road_req_count += 1
        return (
            ERROR_NONE,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            log_count,
        )
    vehicle_finished[v] = True
    terminal_state_count += 1
    vehicle_completion_seconds[v] = time_seconds
    vehicle_terminal_reason_code[v] = TERMINAL_FINISHED
    node_index = _current_node_index(
        path_edge_offsets,
        path_edge_index,
        edge_source_node_index,
        edge_target_node_index,
        p,
        vehicle_edge_index[v],
    )
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_VEHICLE_FINISHED,
        NAMESPACE_VEHICLE,
        v,
        float(node_index),
        vehicle_soc[v],
        _NAN,
        _NAN,
        _NAN,
    )
    if error_code != ERROR_NONE:
        return (
            error_code,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            log_count,
        )
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_VEHICLE_ARRIVED,
        NAMESPACE_VEHICLE,
        v,
        float(node_index),
        vehicle_soc[v],
        _NAN,
        _NAN,
        _NAN,
    )
    return (
        error_code,
        charge_req_count,
        road_req_count,
        terminal_state_count,
        log_count,
    )


@register_jitable
def _settle_mcs_all(
    time_seconds,
    provider_kind_code,
    provider_power_kw,
    provider_charging_efficiency,
    provider_mcs_unit_index,
    provider_active_sessions,
    provider_session_vehicles,
    provider_last_settle,
    vehicle_session_remaining_energy_kwh,
    mcs_delivered_energy_kwh,
):
    n_providers = provider_kind_code.shape[0]
    for p in range(n_providers):
        if provider_kind_code[p] != MCS_KIND_CODE:
            continue
        n_sessions = provider_active_sessions[p]
        if n_sessions == 0:
            provider_last_settle[p] = time_seconds
            continue
        elapsed = time_seconds - provider_last_settle[p]
        if elapsed < 0.0:
            return ERROR_SETTLE_TIME_BACKWARDS
        if elapsed == 0.0:
            continue
        rate = provider_power_kw[p] * provider_charging_efficiency[p] / 3600.0
        delivered_max = rate * elapsed
        unit_index = provider_mcs_unit_index[p]
        for slot in range(n_sessions):
            v = provider_session_vehicles[p, slot]
            remaining = vehicle_session_remaining_energy_kwh[v]
            delivered = delivered_max if delivered_max < remaining else remaining
            vehicle_session_remaining_energy_kwh[v] = remaining - delivered
            mcs_delivered_energy_kwh[unit_index] += delivered
        provider_last_settle[p] = time_seconds
    return ERROR_NONE


@register_jitable
def _start_session(
    v,
    action_slot,
    provider_index,
    site_index,
    time_seconds,
    request_time,
    action_requested_energy_kwh,
    vehicle_battery_kwh,
    vehicle_maximum_soc,
    vehicle_soc,
    vehicle_waiting,
    vehicle_active_provider,
    vehicle_session_remaining_energy_kwh,
    vehicle_session_requested_energy_kwh,
    vehicle_session_token,
    provider_kind_code,
    provider_power_kw,
    provider_charging_efficiency,
    provider_source_to_battery_efficiency,
    provider_mcs_unit_index,
    provider_active_sessions,
    provider_session_vehicles,
    mcs_unreserved_source_energy_kwh,
    slot_rank,
    h_time,
    h_key,
    h_kind,
    h_payload,
    heap_size,
    heap_capacity,
    sequence,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    requested = action_requested_energy_kwh[action_slot]
    if vehicle_soc[v] + requested / vehicle_battery_kwh[v] > vehicle_maximum_soc[v]:
        return (ERROR_CHARGE_OVERFILL, heap_size, sequence, log_count)
    vehicle_waiting[v] = False
    vehicle_active_provider[v] = provider_index
    vehicle_session_remaining_energy_kwh[v] = requested
    vehicle_session_requested_energy_kwh[v] = requested
    slot = provider_active_sessions[provider_index]
    provider_session_vehicles[provider_index, slot] = v
    provider_active_sessions[provider_index] = slot + 1
    is_mcs = provider_kind_code[provider_index] == MCS_KIND_CODE
    if is_mcs:
        unit_index = provider_mcs_unit_index[provider_index]
        source_required = (
            requested / provider_source_to_battery_efficiency[provider_index]
        )
        available = mcs_unreserved_source_energy_kwh[unit_index]
        if source_required > available:
            return (ERROR_MCS_SESSION_EXCEEDS_RESERVE, heap_size, sequence, log_count)
        mcs_unreserved_source_energy_kwh[unit_index] = available - source_required
    occupancy = provider_active_sessions[provider_index]
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_CHARGING_STARTED,
        NAMESPACE_CHARGING_SITE,
        site_index,
        float(v),
        float(provider_index),
        time_seconds - request_time,
        requested,
        float(occupancy),
    )
    if error_code != ERROR_NONE:
        return (error_code, heap_size, sequence, log_count)
    duration = (
        3600.0
        * requested
        / (
            provider_power_kw[provider_index]
            * provider_charging_efficiency[provider_index]
        )
    )
    rank = slot_rank[v]
    sequence += 1
    vehicle_session_token[v] = sequence
    heap_size, error_code = _heap_push(
        h_time,
        h_key,
        h_kind,
        h_payload,
        heap_size,
        heap_capacity,
        time_seconds + duration,
        _pack_key(PHASE_CHARGE_FINISH, rank, sequence),
        KIND_CHARGE_FINISH,
        _pack_payload(provider_index, v),
    )
    if error_code != ERROR_NONE:
        return (error_code, heap_size, sequence, log_count)
    return (ERROR_NONE, heap_size, sequence, log_count)


@register_jitable
def _fail_request(
    v,
    site_index,
    node_index,
    time_seconds,
    reason_code,
    vehicle_waiting,
    vehicle_failed,
    vehicle_terminal_reason_code,
    terminal_state_count,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    vehicle_waiting[v] = False
    vehicle_failed[v] = True
    terminal_state_count += 1
    vehicle_terminal_reason_code[v] = reason_code
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_CHARGING_REJECTED,
        NAMESPACE_CHARGING_SITE,
        site_index,
        float(v),
        float(reason_code),
        _NAN,
        _NAN,
        _NAN,
    )
    if error_code != ERROR_NONE:
        return (error_code, terminal_state_count, log_count)
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_VEHICLE_UNFINISHED,
        NAMESPACE_VEHICLE,
        v,
        float(reason_code),
        float(node_index),
        _NAN,
        _NAN,
        _NAN,
    )
    return (error_code, terminal_state_count, log_count)


@register_jitable
def _finish_charge(
    provider_index,
    v,
    popped_seq,
    time_seconds,
    vehicle_session_token,
    vehicle_session_remaining_energy_kwh,
    vehicle_session_requested_energy_kwh,
    vehicle_battery_kwh,
    vehicle_maximum_soc,
    vehicle_soc,
    vehicle_action_index,
    vehicle_active_provider,
    provider_kind_code,
    provider_site_index,
    provider_power_kw,
    provider_charging_efficiency,
    provider_active_sessions,
    provider_session_vehicles,
    mcs_delivered_energy_kwh,
    provider_mcs_unit_index,
    dirty_site,
    slot_rank,
    path_index,
    path_edge_offsets,
    path_edge_index,
    path_action_offsets,
    vehicle_action_offsets,
    action_site_index,
    edge_source_node_index,
    edge_target_node_index,
    node_to_site_index,
    vehicle_edge_index,
    vehicle_waiting,
    vehicle_finished,
    vehicle_terminal_reason_code,
    vehicle_completion_seconds,
    charge_req_vehicle,
    charge_req_action_slot,
    charge_req_binding_slot,
    charge_req_time,
    charge_req_count,
    road_req_vehicle,
    road_req_edge_progress,
    road_req_count,
    terminal_state_count,
    h_time,
    h_key,
    h_kind,
    h_payload,
    heap_size,
    heap_capacity,
    sequence,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if vehicle_session_token[v] != popped_seq:
        return (
            ERROR_NONE,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            heap_size,
            sequence,
            log_count,
        )
    is_mcs = provider_kind_code[provider_index] == MCS_KIND_CODE
    if is_mcs:
        residual = vehicle_session_remaining_energy_kwh[v]
        if residual > 0.0:
            unit_index = provider_mcs_unit_index[provider_index]
            mcs_delivered_energy_kwh[unit_index] += residual
        vehicle_session_remaining_energy_kwh[v] = 0.0
    requested = vehicle_session_requested_energy_kwh[v]
    new_soc = vehicle_soc[v] + requested / vehicle_battery_kwh[v]
    if new_soc > vehicle_maximum_soc[v]:
        return (
            ERROR_CHARGE_EXCEEDS_MAX_SOC,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            heap_size,
            sequence,
            log_count,
        )
    vehicle_soc[v] = new_soc
    vehicle_action_index[v] += 1
    vehicle_active_provider[v] = -1
    n_sessions = provider_active_sessions[provider_index]
    found_slot = 0
    for slot in range(n_sessions):
        if provider_session_vehicles[provider_index, slot] == v:
            found_slot = slot
            break
    last_slot = n_sessions - 1
    provider_session_vehicles[provider_index, found_slot] = provider_session_vehicles[
        provider_index, last_slot
    ]
    provider_active_sessions[provider_index] = last_slot
    site_index = provider_site_index[provider_index]
    dirty_site[site_index] = True
    occupancy = provider_active_sessions[provider_index]
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_CHARGING_FINISHED,
        NAMESPACE_CHARGING_SITE,
        site_index,
        float(v),
        float(provider_index),
        requested,
        vehicle_soc[v],
        float(occupancy),
    )
    if error_code != ERROR_NONE:
        return (
            error_code,
            charge_req_count,
            road_req_count,
            terminal_state_count,
            heap_size,
            sequence,
            log_count,
        )
    error_code, charge_req_count, road_req_count, terminal_state_count, log_count = (
        _advance(
            v,
            time_seconds,
            path_index,
            path_edge_offsets,
            path_edge_index,
            path_action_offsets,
            vehicle_action_offsets,
            action_site_index,
            edge_source_node_index,
            edge_target_node_index,
            node_to_site_index,
            vehicle_edge_index,
            vehicle_action_index,
            vehicle_waiting,
            vehicle_finished,
            vehicle_active_provider,
            vehicle_terminal_reason_code,
            vehicle_completion_seconds,
            vehicle_soc,
            charge_req_vehicle,
            charge_req_action_slot,
            charge_req_binding_slot,
            charge_req_time,
            charge_req_count,
            road_req_vehicle,
            road_req_edge_progress,
            road_req_count,
            terminal_state_count,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
    )
    return (
        error_code,
        charge_req_count,
        road_req_count,
        terminal_state_count,
        heap_size,
        sequence,
        log_count,
    )


@register_jitable
def _serve_site(
    site_index,
    site_node_index,
    time_seconds,
    queue_head,
    queue_tail,
    request_next,
    request_vehicle_index,
    request_action_slot,
    request_binding_slot,
    request_time,
    queued_flag,
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_requested_energy_kwh,
    vehicle_battery_kwh,
    vehicle_maximum_soc,
    vehicle_soc,
    vehicle_waiting,
    vehicle_active_provider,
    vehicle_session_remaining_energy_kwh,
    vehicle_session_requested_energy_kwh,
    vehicle_session_token,
    vehicle_failed,
    vehicle_terminal_reason_code,
    provider_site_index,
    provider_kind_code,
    provider_available,
    provider_port_count,
    provider_active_sessions,
    provider_session_vehicles,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    provider_power_kw,
    provider_charging_efficiency,
    mcs_unreserved_source_energy_kwh,
    fcs_site_offsets,
    fcs_site_members,
    scratch_candidates,
    slot_rank,
    terminal_state_count,
    h_time,
    h_key,
    h_kind,
    h_payload,
    heap_size,
    heap_capacity,
    sequence,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    error_code = ERROR_NONE
    while queue_head[site_index] != -1:
        head_slot = queue_head[site_index]
        v = request_vehicle_index[head_slot]
        action_slot = request_action_slot[head_slot]
        binding_slot = request_binding_slot[head_slot]
        request_t = request_time[head_slot]
        provider_index = _select_server(
            action_slot,
            binding_slot,
            site_index,
            binding_mode_code,
            bound_provider_index,
            action_eligible_fcs,
            action_eligible_mcs,
            action_requested_energy_kwh,
            provider_site_index,
            provider_kind_code,
            provider_available,
            provider_port_count,
            provider_active_sessions,
            provider_mcs_unit_index,
            provider_source_to_battery_efficiency,
            mcs_unreserved_source_energy_kwh,
            fcs_site_offsets,
            fcs_site_members,
            scratch_candidates,
        )
        if provider_index != -1:
            queue_head[site_index] = request_next[head_slot]
            if queue_head[site_index] == -1:
                queue_tail[site_index] = -1
            queued_flag[v] = False
            error_code, heap_size, sequence, log_count = _start_session(
                v,
                action_slot,
                provider_index,
                site_index,
                time_seconds,
                request_t,
                action_requested_energy_kwh,
                vehicle_battery_kwh,
                vehicle_maximum_soc,
                vehicle_soc,
                vehicle_waiting,
                vehicle_active_provider,
                vehicle_session_remaining_energy_kwh,
                vehicle_session_requested_energy_kwh,
                vehicle_session_token,
                provider_kind_code,
                provider_power_kw,
                provider_charging_efficiency,
                provider_source_to_battery_efficiency,
                provider_mcs_unit_index,
                provider_active_sessions,
                provider_session_vehicles,
                mcs_unreserved_source_energy_kwh,
                slot_rank,
                h_time,
                h_key,
                h_kind,
                h_payload,
                heap_size,
                heap_capacity,
                sequence,
                record_events,
                log_time,
                log_kind,
                log_namespace,
                log_subject,
                log_v,
                log_count,
                log_capacity,
            )
            if error_code != ERROR_NONE:
                break
            continue
        stranded = _is_stranded(
            action_slot,
            binding_slot,
            site_index,
            time_seconds,
            binding_mode_code,
            bound_provider_index,
            action_eligible_fcs,
            action_eligible_mcs,
            action_requested_energy_kwh,
            provider_site_index,
            provider_kind_code,
            provider_available,
            provider_mcs_unit_index,
            provider_source_to_battery_efficiency,
            mcs_unreserved_source_energy_kwh,
            fcs_site_offsets,
            fcs_site_members,
            scratch_candidates,
        )
        if not stranded:
            break
        queue_head[site_index] = request_next[head_slot]
        if queue_head[site_index] == -1:
            queue_tail[site_index] = -1
        queued_flag[v] = False
        error_code, terminal_state_count, log_count = _fail_request(
            v,
            site_index,
            site_node_index,
            time_seconds,
            TERMINAL_CHARGING_ENERGY_UNAVAILABLE,
            vehicle_waiting,
            vehicle_failed,
            vehicle_terminal_reason_code,
            terminal_state_count,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
        if error_code != ERROR_NONE:
            break
    return (error_code, terminal_state_count, heap_size, sequence, log_count)


@register_jitable
def _forbidden_rejection_reason(
    action_slot,
    binding_slot,
    site_index,
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_requested_energy_kwh,
    provider_site_index,
    provider_kind_code,
    provider_available,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    mcs_unreserved_source_energy_kwh,
    fcs_site_offsets,
    fcs_site_members,
    scratch_candidates,
):
    count = _eligible_candidates_at_site(
        binding_mode_code,
        bound_provider_index,
        action_eligible_fcs,
        action_eligible_mcs,
        action_slot,
        binding_slot,
        site_index,
        provider_site_index,
        provider_available,
        fcs_site_offsets,
        fcs_site_members,
        provider_mcs_unit_index,
        scratch_candidates,
    )
    for i in range(count):
        provider_index = scratch_candidates[i]
        if provider_kind_code[provider_index] != MCS_KIND_CODE:
            continue
        unit_index = provider_mcs_unit_index[provider_index]
        shortfall = _mcs_source_shortfall(
            action_requested_energy_kwh[action_slot],
            provider_source_to_battery_efficiency[provider_index],
            mcs_unreserved_source_energy_kwh[unit_index],
        )
        if shortfall > 0.0:
            return TERMINAL_CHARGING_ENERGY_UNAVAILABLE
    return TERMINAL_CHARGING_PORT_UNAVAILABLE


@register_jitable
def _reject_unserved_site(
    site_index,
    site_node_index,
    time_seconds,
    queue_head,
    queue_tail,
    request_next,
    request_vehicle_index,
    request_action_slot,
    request_binding_slot,
    queued_flag,
    binding_mode_code,
    bound_provider_index,
    action_eligible_fcs,
    action_eligible_mcs,
    action_requested_energy_kwh,
    provider_site_index,
    provider_kind_code,
    provider_available,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    mcs_unreserved_source_energy_kwh,
    fcs_site_offsets,
    fcs_site_members,
    scratch_candidates,
    vehicle_waiting,
    vehicle_failed,
    vehicle_terminal_reason_code,
    terminal_state_count,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    while queue_head[site_index] != -1:
        head_slot = queue_head[site_index]
        v = request_vehicle_index[head_slot]
        action_slot = request_action_slot[head_slot]
        binding_slot = request_binding_slot[head_slot]
        queue_head[site_index] = request_next[head_slot]
        if queue_head[site_index] == -1:
            queue_tail[site_index] = -1
        queued_flag[v] = False
        reason_code = _forbidden_rejection_reason(
            action_slot,
            binding_slot,
            site_index,
            binding_mode_code,
            bound_provider_index,
            action_eligible_fcs,
            action_eligible_mcs,
            action_requested_energy_kwh,
            provider_site_index,
            provider_kind_code,
            provider_available,
            provider_mcs_unit_index,
            provider_source_to_battery_efficiency,
            mcs_unreserved_source_energy_kwh,
            fcs_site_offsets,
            fcs_site_members,
            scratch_candidates,
        )
        error_code, terminal_state_count, log_count = _fail_request(
            v,
            site_index,
            site_node_index,
            time_seconds,
            reason_code,
            vehicle_waiting,
            vehicle_failed,
            vehicle_terminal_reason_code,
            terminal_state_count,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
        if error_code != ERROR_NONE:
            return (error_code, terminal_state_count, log_count)
    return (ERROR_NONE, terminal_state_count, log_count)


@register_jitable
def _reject_overflow_site(
    site_index,
    site_node_index,
    time_seconds,
    capacity,
    queue_head,
    queue_tail,
    request_next,
    request_vehicle_index,
    queued_flag,
    vehicle_waiting,
    vehicle_failed,
    vehicle_terminal_reason_code,
    terminal_state_count,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if capacity < 0 or queue_head[site_index] == -1:
        return (ERROR_NONE, terminal_state_count, log_count)
    if capacity == 0:
        overflow_start = queue_head[site_index]
        queue_head[site_index] = -1
        queue_tail[site_index] = -1
    else:
        node = queue_head[site_index]
        steps = 1
        while steps < capacity and node != -1:
            node = request_next[node]
            steps += 1
        if node == -1:
            return (ERROR_NONE, terminal_state_count, log_count)
        overflow_start = request_next[node]
        request_next[node] = -1
        queue_tail[site_index] = node
    cursor = overflow_start
    while cursor != -1:
        v = request_vehicle_index[cursor]
        queued_flag[v] = False
        next_cursor = request_next[cursor]
        error_code, terminal_state_count, log_count = _fail_request(
            v,
            site_index,
            site_node_index,
            time_seconds,
            TERMINAL_CHARGING_QUEUE_OVERFLOW,
            vehicle_waiting,
            vehicle_failed,
            vehicle_terminal_reason_code,
            terminal_state_count,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
        if error_code != ERROR_NONE:
            return (error_code, terminal_state_count, log_count)
        cursor = next_cursor
    return (ERROR_NONE, terminal_state_count, log_count)


@register_jitable
def _record_waiting_site(
    site_index,
    time_seconds,
    queue_head,
    request_next,
    request_vehicle_index,
    queued_event_flag,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if not record_events:
        return (log_count, ERROR_NONE)
    length = 0
    cursor = queue_head[site_index]
    while cursor != -1:
        length += 1
        cursor = request_next[cursor]
    position = 0
    cursor = queue_head[site_index]
    while cursor != -1:
        position += 1
        v = request_vehicle_index[cursor]
        if not queued_event_flag[v]:
            queued_event_flag[v] = True
            log_count, error_code = _log_event(
                record_events,
                log_time,
                log_kind,
                log_namespace,
                log_subject,
                log_v,
                log_count,
                log_capacity,
                time_seconds,
                EVENT_CHARGING_QUEUED,
                NAMESPACE_CHARGING_SITE,
                site_index,
                float(v),
                float(length),
                float(position),
                _NAN,
                _NAN,
            )
            if error_code != ERROR_NONE:
                return (log_count, error_code)
        cursor = request_next[cursor]
    return (log_count, ERROR_NONE)


@register_jitable
def _admit_roads_batch(
    time_seconds,
    road_req_vehicle,
    road_req_edge_progress,
    road_req_n,
    path_index,
    path_edge_offsets,
    path_edge_index,
    edge_distance_m,
    edge_speed_limit_m_per_second,
    vehicle_speed_m_per_second,
    vehicle_efficiency_m_per_kwh,
    vehicle_battery_kwh,
    vehicle_minimum_soc,
    vehicle_soc,
    active_edge_count,
    traffic_times,
    traffic_multipliers,
    slot_rank,
    scratch_order,
    scratch_edge_of,
    scratch_free_flow,
    h_time,
    h_key,
    h_kind,
    h_payload,
    heap_size,
    heap_capacity,
    sequence,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if road_req_n == 0:
        return (ERROR_NONE, heap_size, sequence, log_count)
    order = scratch_order
    edge_of = scratch_edge_of
    for i in range(road_req_n):
        order[i] = i
    for i in range(road_req_n):
        v = road_req_vehicle[i]
        p = path_index[v]
        start = path_edge_offsets[p]
        edge_of[i] = path_edge_index[start + road_req_edge_progress[i]]
    for i in range(1, road_req_n):
        key = order[i]
        key_edge = edge_of[key]
        key_vehicle = road_req_vehicle[key]
        j = i - 1
        while j >= 0:
            cur = order[j]
            if edge_of[cur] < key_edge or (
                edge_of[cur] == key_edge and road_req_vehicle[cur] <= key_vehicle
            ):
                break
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    idx = 0
    while idx < road_req_n:
        edge_id = edge_of[order[idx]]
        group_end = idx + 1
        while group_end < road_req_n and edge_of[order[group_end]] == edge_id:
            group_end += 1
        count = group_end - idx
        occupancy_before = active_edge_count[edge_id]
        free_flow = scratch_free_flow
        for k in range(count):
            i = order[idx + k]
            v = road_req_vehicle[i]
            speed_limit = edge_speed_limit_m_per_second[edge_id]
            speed = vehicle_speed_m_per_second[v]
            effective_speed = speed if speed < speed_limit else speed_limit
            free_flow[k] = edge_distance_m[edge_id] / effective_speed
            energy = edge_distance_m[edge_id] / vehicle_efficiency_m_per_kwh[v]
            next_soc = vehicle_soc[v] - energy / vehicle_battery_kwh[v]
            if next_soc < vehicle_minimum_soc[v]:
                return (
                    ERROR_ROAD_ADMISSION_BELOW_MINIMUM_SOC,
                    heap_size,
                    sequence,
                    log_count,
                )
            vehicle_soc[v] = next_soc
        for k in range(count):
            i = order[idx + k]
            v = road_req_vehicle[i]
            exit_time = traffic_exit_time_kernel(
                traffic_times, traffic_multipliers, time_seconds, free_flow[k]
            )
            rank = slot_rank[v]
            sequence += 1
            heap_size, error_code = _heap_push(
                h_time,
                h_key,
                h_kind,
                h_payload,
                heap_size,
                heap_capacity,
                exit_time,
                _pack_key(PHASE_ROAD_ARRIVAL, rank, sequence),
                KIND_ROAD_ARRIVAL,
                _pack_payload(edge_id, v),
            )
            if error_code != ERROR_NONE:
                return (error_code, heap_size, sequence, log_count)
            log_count, error_code = _log_event(
                record_events,
                log_time,
                log_kind,
                log_namespace,
                log_subject,
                log_v,
                log_count,
                log_capacity,
                time_seconds,
                EVENT_EDGE_ENTERED,
                NAMESPACE_VEHICLE,
                v,
                float(edge_id),
                vehicle_soc[v],
                _NAN,
                _NAN,
                _NAN,
            )
            if error_code != ERROR_NONE:
                return (error_code, heap_size, sequence, log_count)
            log_count, error_code = _log_event(
                record_events,
                log_time,
                log_kind,
                log_namespace,
                log_subject,
                log_v,
                log_count,
                log_capacity,
                time_seconds,
                EVENT_EDGE_ENTRY,
                NAMESPACE_EDGE,
                edge_id,
                float(v),
                float(occupancy_before),
                1.0,
                exit_time,
                _NAN,
            )
            if error_code != ERROR_NONE:
                return (error_code, heap_size, sequence, log_count)
            active_edge_count[edge_id] += 1
        idx = group_end
    return (ERROR_NONE, heap_size, sequence, log_count)


@register_jitable
def _enqueue_charge_requests(
    time_seconds,
    charge_req_vehicle,
    charge_req_action_slot,
    charge_req_binding_slot,
    charge_req_time,
    charge_req_n,
    action_site_index,
    binding_mode_code,
    bound_provider_index,
    queued_flag,
    request_vehicle_index,
    request_action_slot,
    request_binding_slot,
    request_time,
    request_next,
    queue_head,
    queue_tail,
    dirty_site,
    next_free_slot,
    scratch_order,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    if charge_req_n == 0:
        return (ERROR_NONE, next_free_slot, log_count)
    order = scratch_order
    for i in range(charge_req_n):
        order[i] = i
    for i in range(1, charge_req_n):
        key = order[i]
        key_site = action_site_index[charge_req_action_slot[key]]
        key_vehicle = charge_req_vehicle[key]
        j = i - 1
        while j >= 0:
            cur = order[j]
            cur_site = action_site_index[charge_req_action_slot[cur]]
            if cur_site < key_site or (
                cur_site == key_site and charge_req_vehicle[cur] <= key_vehicle
            ):
                break
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    for idx in range(charge_req_n):
        i = order[idx]
        v = charge_req_vehicle[i]
        action_slot = charge_req_action_slot[i]
        binding_slot = charge_req_binding_slot[i]
        site_index = action_site_index[action_slot]
        if queued_flag[v]:
            return (ERROR_DUPLICATE_QUEUE_JOIN, next_free_slot, log_count)
        queued_flag[v] = True
        slot = next_free_slot
        next_free_slot += 1
        request_vehicle_index[slot] = v
        request_action_slot[slot] = action_slot
        request_binding_slot[slot] = binding_slot
        request_time[slot] = charge_req_time[i]
        request_next[slot] = -1
        if queue_tail[site_index] == -1:
            queue_head[site_index] = slot
        else:
            request_next[queue_tail[site_index]] = slot
        queue_tail[site_index] = slot
        dirty_site[site_index] = True
        mode = binding_mode_code[binding_slot]
        log_count, error_code = _log_event(
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
            time_seconds,
            EVENT_CHARGING_REQUESTED,
            NAMESPACE_CHARGING_SITE,
            site_index,
            float(v),
            float(mode),
            float(bound_provider_index[binding_slot]),
            _NAN,
            _NAN,
        )
        if error_code != ERROR_NONE:
            return (error_code, next_free_slot, log_count)
    return (ERROR_NONE, next_free_slot, log_count)


@register_jitable
def _initialize_deployment_events(
    n_deployments,
    unit_index,
    deployment_site_index,
    n_sites,
    n_mcs_units,
    h_time,
    h_key,
    h_kind,
    h_payload,
    heap_size,
    heap_capacity,
    sequence,
):
    within_kind_span = n_sites * n_mcs_units if n_mcs_units > 0 else 1
    for deployment_index in range(n_deployments):
        u = unit_index[deployment_index]
        s = deployment_site_index[deployment_index]
        rank = MCS_KIND_PRIORITY_ARRIVAL * within_kind_span + s * n_mcs_units + u
        sequence += 1
        heap_size, error_code = _heap_push(
            h_time,
            h_key,
            h_kind,
            h_payload,
            heap_size,
            heap_capacity,
            0.0,
            _pack_key(PHASE_MCS_LIFECYCLE, rank, sequence),
            KIND_MCS_ARRIVAL,
            _pack_payload(u, deployment_index),
        )
        if error_code != ERROR_NONE:
            return (heap_size, sequence, error_code)
    return (heap_size, sequence, ERROR_NONE)


@register_jitable
def _process_mcs_lifecycle_event(
    kind,
    unit_index,
    deployment_index,
    time_seconds,
    deployment_site_index,
    site_to_node_index,
    provider_index_of_mcs_unit,
    provider_site_index,
    provider_available,
    provider_active_sessions,
    provider_last_settle,
    mcs_lifecycle_state,
    mcs_location_node,
    mcs_battery_kwh,
    mcs_maximum_soc,
    mcs_minimum_soc,
    mcs_unreserved_source_energy_kwh,
    dirty_site,
    record_events,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    log_count,
    log_capacity,
):
    provider_index = provider_index_of_mcs_unit[unit_index]
    site_index = deployment_site_index[deployment_index]
    provider_site_index[provider_index] = site_index
    provider_available[provider_index] = True
    mcs_lifecycle_state[unit_index] = LIFECYCLE_ACTIVE_AT_SITE
    mcs_location_node[unit_index] = site_to_node_index[site_index]
    provider_last_settle[provider_index] = time_seconds
    dirty_site[site_index] = True
    log_count, error_code = _log_event(
        record_events,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
        log_capacity,
        time_seconds,
        EVENT_MCS_DEPLOYED,
        NAMESPACE_MCS_UNIT,
        unit_index,
        float(site_index),
        _NAN,
        _NAN,
        _NAN,
        _NAN,
    )
    return (error_code, log_count)


@numba.njit(cache=True)
def run_core(
    vehicle_battery_kwh,
    vehicle_initial_soc,
    vehicle_minimum_soc,
    vehicle_maximum_soc,
    vehicle_efficiency_m_per_kwh,
    vehicle_speed_m_per_second,
    edge_distance_m,
    edge_speed_limit_m_per_second,
    edge_source_node_index,
    edge_target_node_index,
    path_edge_offsets,
    path_edge_index,
    path_action_offsets,
    action_site_index,
    action_requested_energy_kwh,
    action_eligible_fcs,
    action_eligible_mcs,
    provider_kind_code,
    provider_port_count,
    provider_power_kw,
    provider_charging_efficiency,
    provider_mcs_unit_index,
    provider_source_to_battery_efficiency,
    provider_site_index_initial,
    fcs_site_offsets,
    fcs_site_members,
    mcs_battery_kwh,
    mcs_initial_soc,
    mcs_minimum_soc,
    mcs_maximum_soc,
    provider_index_of_mcs_unit,
    traffic_times,
    traffic_multipliers,
    tolerance,
    queue_policy_code,
    queue_capacity,
    node_to_site_index,
    site_to_node_index,
    path_index,
    departure_seconds,
    vehicle_action_offsets,
    binding_mode_code,
    bound_provider_index,
    slot_rank,
    unit_index,
    deployment_site_index,
    record_events,
    vehicle_f64,
    vehicle_i64,
    vehicle_bool,
    provider_f64,
    provider_i64,
    provider_bool,
    provider_session_vehicles,
    mcs_f64,
    mcs_i64,
    request_vehicle_index,
    request_action_slot,
    request_binding_slot,
    request_time,
    request_next,
    queue_head,
    queue_tail,
    dirty_site,
    active_edge_count,
    charge_req_vehicle,
    charge_req_action_slot,
    charge_req_binding_slot,
    charge_req_time,
    road_req_vehicle,
    road_req_edge_progress,
    scratch_candidates,
    scratch_road_order,
    scratch_edge_of,
    scratch_free_flow,
    scratch_charge_order,
    h_time,
    h_key,
    h_kind,
    h_payload,
    b_time,
    b_key,
    b_kind,
    b_payload,
    b_order,
    log_time,
    log_kind,
    log_namespace,
    log_subject,
    log_v,
    scalars_i64,
    scalars_f64,
):
    n_vehicles = vehicle_battery_kwh.shape[0]
    n_providers = provider_kind_code.shape[0]
    n_mcs = mcs_battery_kwh.shape[0]
    n_sites = fcs_site_offsets.shape[0] - 1
    n_deployments = unit_index.shape[0]
    vehicle_soc = vehicle_f64[:, V_F_SOC]
    vehicle_completion_seconds = vehicle_f64[:, V_F_COMPLETION_SECONDS]
    vehicle_session_remaining_energy_kwh = vehicle_f64[
        :, V_F_SESSION_REMAINING_ENERGY_KWH
    ]
    vehicle_session_requested_energy_kwh = vehicle_f64[
        :, V_F_SESSION_REQUESTED_ENERGY_KWH
    ]
    vehicle_edge_index = vehicle_i64[:, V_I_EDGE_INDEX]
    vehicle_action_index = vehicle_i64[:, V_I_ACTION_INDEX]
    vehicle_active_provider = vehicle_i64[:, V_I_ACTIVE_PROVIDER]
    vehicle_terminal_reason_code = vehicle_i64[:, V_I_TERMINAL_REASON_CODE]
    vehicle_session_token = vehicle_i64[:, V_I_SESSION_TOKEN]
    vehicle_waiting = vehicle_bool[:, V_B_WAITING]
    vehicle_finished = vehicle_bool[:, V_B_FINISHED]
    vehicle_failed = vehicle_bool[:, V_B_FAILED]
    queued_flag = vehicle_bool[:, V_B_QUEUED]
    queued_event_flag = vehicle_bool[:, V_B_QUEUED_EVENT]
    provider_last_settle = provider_f64[:, P_F_LAST_SETTLE]
    provider_site_index_run = provider_i64[:, P_I_SITE_INDEX]
    provider_active_sessions = provider_i64[:, P_I_ACTIVE_SESSIONS]
    provider_available = provider_bool[:, P_B_AVAILABLE]
    mcs_unreserved_source_energy_kwh = mcs_f64[:, M_F_UNRESERVED_SOURCE_ENERGY_KWH]
    mcs_delivered_energy_kwh = mcs_f64[:, M_F_DELIVERED_ENERGY_KWH]
    mcs_lifecycle_state = mcs_i64[:, M_I_LIFECYCLE_STATE]
    mcs_location_node = mcs_i64[:, M_I_LOCATION_NODE]
    heap_capacity = h_time.shape[0]
    batch_capacity = b_time.shape[0]
    log_capacity = log_time.shape[0]
    vehicle_soc[:] = vehicle_initial_soc
    vehicle_edge_index[:] = 0
    vehicle_action_index[:] = 0
    vehicle_waiting[:] = False
    vehicle_finished[:] = False
    vehicle_failed[:] = False
    vehicle_active_provider[:] = -1
    vehicle_terminal_reason_code[:] = TERMINAL_NONE
    vehicle_completion_seconds[:] = np.nan
    vehicle_session_remaining_energy_kwh[:] = 0.0
    vehicle_session_requested_energy_kwh[:] = 0.0
    vehicle_session_token[:] = -1
    queued_flag[:] = False
    queued_event_flag[:] = False
    provider_site_index_run[:] = provider_site_index_initial
    provider_available[:] = False
    provider_active_sessions[:] = 0
    provider_session_vehicles[:, :] = -1
    provider_last_settle[:] = 0.0
    for p in range(n_providers):
        if provider_kind_code[p] == FCS_KIND_CODE:
            provider_available[p] = True
    mcs_unreserved_source_energy_kwh[:] = mcs_battery_kwh * (
        mcs_initial_soc - mcs_minimum_soc
    )
    mcs_delivered_energy_kwh[:] = 0.0
    mcs_lifecycle_state[:] = 0
    mcs_location_node[:] = -1
    queue_head[:] = -1
    queue_tail[:] = -1
    dirty_site[:] = False
    active_edge_count[:] = 0
    next_free_slot = 0
    heap_size = 0
    sequence = 0
    terminal_state_count = 0
    log_count = 0
    last_event_seconds = 0.0
    rank_bound = max(n_vehicles, 5 * n_sites * n_mcs)
    key_range_error = ERROR_NONE
    if rank_bound >= RANK_MODULUS or heap_capacity >= SEQUENCE_MODULUS:
        key_range_error = ERROR_KEY_RANGE
    error_code = key_range_error
    if error_code == ERROR_NONE:
        for v in range(n_vehicles):
            sequence += 1
            heap_size, error_code = _heap_push(
                h_time,
                h_key,
                h_kind,
                h_payload,
                heap_size,
                heap_capacity,
                departure_seconds[v],
                _pack_key(PHASE_VEHICLE_DEPARTURE, slot_rank[v], sequence),
                KIND_VEHICLE_DEPARTURE,
                _pack_payload(v, -1),
            )
            if error_code != ERROR_NONE:
                break
    if error_code == ERROR_NONE:
        heap_size, sequence, error_code = _initialize_deployment_events(
            n_deployments,
            unit_index,
            deployment_site_index,
            n_sites,
            n_mcs,
            h_time,
            h_key,
            h_kind,
            h_payload,
            heap_size,
            heap_capacity,
            sequence,
        )
    while heap_size > 0 and error_code == ERROR_NONE:
        anchor = h_time[0]
        last_event_seconds = anchor
        error_code = _settle_mcs_all(
            anchor,
            provider_kind_code,
            provider_power_kw,
            provider_charging_efficiency,
            provider_mcs_unit_index,
            provider_active_sessions,
            provider_session_vehicles,
            provider_last_settle,
            vehicle_session_remaining_energy_kwh,
            mcs_delivered_energy_kwh,
        )
        if error_code != ERROR_NONE:
            break
        batch_n = 0
        while heap_size > 0 and h_time[0] <= anchor + tolerance:
            t, ky, kd, pl, heap_size = _heap_pop(
                h_time, h_key, h_kind, h_payload, heap_size
            )
            if batch_n == batch_capacity:
                error_code = ERROR_CAPACITY_EXCEEDED
                break
            b_time[batch_n] = t
            b_key[batch_n] = ky
            b_kind[batch_n] = kd
            b_payload[batch_n] = pl
            batch_n += 1
        if error_code != ERROR_NONE:
            break
        order = b_order
        for i in range(batch_n):
            order[i] = i
        for i in range(1, batch_n):
            idx = order[i]
            idx_key = b_key[idx]
            j = i - 1
            while j >= 0:
                cur = order[j]
                if b_key[cur] <= idx_key:
                    break
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = idx
        charge_req_count = 0
        road_req_count = 0
        for oi in range(batch_n):
            i = order[oi]
            kind = b_kind[i]
            pa, pb = _unpack_payload(b_payload[i])
            seq = b_key[i] & LOW32_MASK
            if kind == KIND_CHARGE_FINISH:
                (
                    error_code,
                    charge_req_count,
                    road_req_count,
                    terminal_state_count,
                    heap_size,
                    sequence,
                    log_count,
                ) = _finish_charge(
                    pa,
                    pb,
                    seq,
                    anchor,
                    vehicle_session_token,
                    vehicle_session_remaining_energy_kwh,
                    vehicle_session_requested_energy_kwh,
                    vehicle_battery_kwh,
                    vehicle_maximum_soc,
                    vehicle_soc,
                    vehicle_action_index,
                    vehicle_active_provider,
                    provider_kind_code,
                    provider_site_index_run,
                    provider_power_kw,
                    provider_charging_efficiency,
                    provider_active_sessions,
                    provider_session_vehicles,
                    mcs_delivered_energy_kwh,
                    provider_mcs_unit_index,
                    dirty_site,
                    slot_rank,
                    path_index,
                    path_edge_offsets,
                    path_edge_index,
                    path_action_offsets,
                    vehicle_action_offsets,
                    action_site_index,
                    edge_source_node_index,
                    edge_target_node_index,
                    node_to_site_index,
                    vehicle_edge_index,
                    vehicle_waiting,
                    vehicle_finished,
                    vehicle_terminal_reason_code,
                    vehicle_completion_seconds,
                    charge_req_vehicle,
                    charge_req_action_slot,
                    charge_req_binding_slot,
                    charge_req_time,
                    charge_req_count,
                    road_req_vehicle,
                    road_req_edge_progress,
                    road_req_count,
                    terminal_state_count,
                    h_time,
                    h_key,
                    h_kind,
                    h_payload,
                    heap_size,
                    heap_capacity,
                    sequence,
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                )
            elif kind == KIND_ROAD_ARRIVAL:
                v = pb
                edge_id = pa
                active_edge_count[edge_id] -= 1
                vehicle_edge_index[v] = vehicle_edge_index[v] + 1
                node_index = edge_target_node_index[edge_id]
                log_count, error_code = _log_event(
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                    anchor,
                    EVENT_EDGE_EXIT,
                    NAMESPACE_EDGE,
                    edge_id,
                    float(v),
                    float(active_edge_count[edge_id]),
                    _NAN,
                    _NAN,
                    _NAN,
                )
                if error_code == ERROR_NONE:
                    log_count, error_code = _log_event(
                        record_events,
                        log_time,
                        log_kind,
                        log_namespace,
                        log_subject,
                        log_v,
                        log_count,
                        log_capacity,
                        anchor,
                        EVENT_EDGE_ARRIVED,
                        NAMESPACE_VEHICLE,
                        v,
                        float(node_index),
                        float(edge_id),
                        vehicle_soc[v],
                        _NAN,
                        _NAN,
                    )
                if error_code == ERROR_NONE:
                    (
                        error_code,
                        charge_req_count,
                        road_req_count,
                        terminal_state_count,
                        log_count,
                    ) = _advance(
                        v,
                        anchor,
                        path_index,
                        path_edge_offsets,
                        path_edge_index,
                        path_action_offsets,
                        vehicle_action_offsets,
                        action_site_index,
                        edge_source_node_index,
                        edge_target_node_index,
                        node_to_site_index,
                        vehicle_edge_index,
                        vehicle_action_index,
                        vehicle_waiting,
                        vehicle_finished,
                        vehicle_active_provider,
                        vehicle_terminal_reason_code,
                        vehicle_completion_seconds,
                        vehicle_soc,
                        charge_req_vehicle,
                        charge_req_action_slot,
                        charge_req_binding_slot,
                        charge_req_time,
                        charge_req_count,
                        road_req_vehicle,
                        road_req_edge_progress,
                        road_req_count,
                        terminal_state_count,
                        record_events,
                        log_time,
                        log_kind,
                        log_namespace,
                        log_subject,
                        log_v,
                        log_count,
                        log_capacity,
                    )
            elif kind == KIND_VEHICLE_DEPARTURE:
                v = pa
                node_index = _current_node_index(
                    path_edge_offsets,
                    path_edge_index,
                    edge_source_node_index,
                    edge_target_node_index,
                    path_index[v],
                    vehicle_edge_index[v],
                )
                log_count, error_code = _log_event(
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                    anchor,
                    EVENT_VEHICLE_DEPARTED,
                    NAMESPACE_VEHICLE,
                    v,
                    float(node_index),
                    vehicle_soc[v],
                    _NAN,
                    _NAN,
                    _NAN,
                )
                if error_code == ERROR_NONE:
                    (
                        error_code,
                        charge_req_count,
                        road_req_count,
                        terminal_state_count,
                        log_count,
                    ) = _advance(
                        v,
                        anchor,
                        path_index,
                        path_edge_offsets,
                        path_edge_index,
                        path_action_offsets,
                        vehicle_action_offsets,
                        action_site_index,
                        edge_source_node_index,
                        edge_target_node_index,
                        node_to_site_index,
                        vehicle_edge_index,
                        vehicle_action_index,
                        vehicle_waiting,
                        vehicle_finished,
                        vehicle_active_provider,
                        vehicle_terminal_reason_code,
                        vehicle_completion_seconds,
                        vehicle_soc,
                        charge_req_vehicle,
                        charge_req_action_slot,
                        charge_req_binding_slot,
                        charge_req_time,
                        charge_req_count,
                        road_req_vehicle,
                        road_req_edge_progress,
                        road_req_count,
                        terminal_state_count,
                        record_events,
                        log_time,
                        log_kind,
                        log_namespace,
                        log_subject,
                        log_v,
                        log_count,
                        log_capacity,
                    )
            else:
                error_code, log_count = _process_mcs_lifecycle_event(
                    kind,
                    pa,
                    pb,
                    anchor,
                    deployment_site_index,
                    site_to_node_index,
                    provider_index_of_mcs_unit,
                    provider_site_index_run,
                    provider_available,
                    provider_active_sessions,
                    provider_last_settle,
                    mcs_lifecycle_state,
                    mcs_location_node,
                    mcs_battery_kwh,
                    mcs_maximum_soc,
                    mcs_minimum_soc,
                    mcs_unreserved_source_energy_kwh,
                    dirty_site,
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                )
            if error_code != ERROR_NONE:
                break
        if error_code != ERROR_NONE:
            break
        error_code, next_free_slot, log_count = _enqueue_charge_requests(
            anchor,
            charge_req_vehicle,
            charge_req_action_slot,
            charge_req_binding_slot,
            charge_req_time,
            charge_req_count,
            action_site_index,
            binding_mode_code,
            bound_provider_index,
            queued_flag,
            request_vehicle_index,
            request_action_slot,
            request_binding_slot,
            request_time,
            request_next,
            queue_head,
            queue_tail,
            dirty_site,
            next_free_slot,
            scratch_charge_order,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
        if error_code != ERROR_NONE:
            break
        for s in range(n_sites):
            if not dirty_site[s]:
                continue
            dirty_site[s] = False
            site_node = site_to_node_index[s]
            error_code, terminal_state_count, heap_size, sequence, log_count = (
                _serve_site(
                    s,
                    site_node,
                    anchor,
                    queue_head,
                    queue_tail,
                    request_next,
                    request_vehicle_index,
                    request_action_slot,
                    request_binding_slot,
                    request_time,
                    queued_flag,
                    binding_mode_code,
                    bound_provider_index,
                    action_eligible_fcs,
                    action_eligible_mcs,
                    action_requested_energy_kwh,
                    vehicle_battery_kwh,
                    vehicle_maximum_soc,
                    vehicle_soc,
                    vehicle_waiting,
                    vehicle_active_provider,
                    vehicle_session_remaining_energy_kwh,
                    vehicle_session_requested_energy_kwh,
                    vehicle_session_token,
                    vehicle_failed,
                    vehicle_terminal_reason_code,
                    provider_site_index_run,
                    provider_kind_code,
                    provider_available,
                    provider_port_count,
                    provider_active_sessions,
                    provider_session_vehicles,
                    provider_mcs_unit_index,
                    provider_source_to_battery_efficiency,
                    provider_power_kw,
                    provider_charging_efficiency,
                    mcs_unreserved_source_energy_kwh,
                    fcs_site_offsets,
                    fcs_site_members,
                    scratch_candidates,
                    slot_rank,
                    terminal_state_count,
                    h_time,
                    h_key,
                    h_kind,
                    h_payload,
                    heap_size,
                    heap_capacity,
                    sequence,
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                )
            )
            if error_code != ERROR_NONE:
                break
            if queue_policy_code == QUEUE_POLICY_FORBIDDEN:
                error_code, terminal_state_count, log_count = _reject_unserved_site(
                    s,
                    site_node,
                    anchor,
                    queue_head,
                    queue_tail,
                    request_next,
                    request_vehicle_index,
                    request_action_slot,
                    request_binding_slot,
                    queued_flag,
                    binding_mode_code,
                    bound_provider_index,
                    action_eligible_fcs,
                    action_eligible_mcs,
                    action_requested_energy_kwh,
                    provider_site_index_run,
                    provider_kind_code,
                    provider_available,
                    provider_mcs_unit_index,
                    provider_source_to_battery_efficiency,
                    mcs_unreserved_source_energy_kwh,
                    fcs_site_offsets,
                    fcs_site_members,
                    scratch_candidates,
                    vehicle_waiting,
                    vehicle_failed,
                    vehicle_terminal_reason_code,
                    terminal_state_count,
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                )
            else:
                error_code, terminal_state_count, log_count = _reject_overflow_site(
                    s,
                    site_node,
                    anchor,
                    queue_capacity,
                    queue_head,
                    queue_tail,
                    request_next,
                    request_vehicle_index,
                    queued_flag,
                    vehicle_waiting,
                    vehicle_failed,
                    vehicle_terminal_reason_code,
                    terminal_state_count,
                    record_events,
                    log_time,
                    log_kind,
                    log_namespace,
                    log_subject,
                    log_v,
                    log_count,
                    log_capacity,
                )
                if error_code == ERROR_NONE:
                    log_count, error_code = _record_waiting_site(
                        s,
                        anchor,
                        queue_head,
                        request_next,
                        request_vehicle_index,
                        queued_event_flag,
                        record_events,
                        log_time,
                        log_kind,
                        log_namespace,
                        log_subject,
                        log_v,
                        log_count,
                        log_capacity,
                    )
            if error_code != ERROR_NONE:
                break
        if error_code != ERROR_NONE:
            break
        error_code, heap_size, sequence, log_count = _admit_roads_batch(
            anchor,
            road_req_vehicle,
            road_req_edge_progress,
            road_req_count,
            path_index,
            path_edge_offsets,
            path_edge_index,
            edge_distance_m,
            edge_speed_limit_m_per_second,
            vehicle_speed_m_per_second,
            vehicle_efficiency_m_per_kwh,
            vehicle_battery_kwh,
            vehicle_minimum_soc,
            vehicle_soc,
            active_edge_count,
            traffic_times,
            traffic_multipliers,
            slot_rank,
            scratch_road_order,
            scratch_edge_of,
            scratch_free_flow,
            h_time,
            h_key,
            h_kind,
            h_payload,
            heap_size,
            heap_capacity,
            sequence,
            record_events,
            log_time,
            log_kind,
            log_namespace,
            log_subject,
            log_v,
            log_count,
            log_capacity,
        )
    if error_code == ERROR_NONE and heap_size == 0:
        for v in range(n_vehicles):
            if vehicle_terminal_reason_code[v] == TERMINAL_NONE:
                error_code = ERROR_MISSING_TERMINAL_REASON
                break
    scalars_i64[S_I_HEAP_SIZE] = heap_size
    scalars_i64[S_I_SEQUENCE] = sequence
    scalars_i64[S_I_NEXT_FREE_SLOT] = next_free_slot
    scalars_i64[S_I_TERMINAL_STATE_COUNT] = terminal_state_count
    scalars_i64[S_I_LOG_COUNT] = log_count
    scalars_f64[S_F_LAST_EVENT_SECONDS] = last_event_seconds
    return (
        error_code,
        vehicle_terminal_reason_code,
        vehicle_completion_seconds,
        last_event_seconds,
        mcs_delivered_energy_kwh,
        mcs_unreserved_source_energy_kwh,
        mcs_lifecycle_state,
        mcs_location_node,
        log_time,
        log_kind,
        log_namespace,
        log_subject,
        log_v,
        log_count,
    )


__all__ = ["run_core"]
