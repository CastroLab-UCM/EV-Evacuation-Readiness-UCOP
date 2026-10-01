from __future__ import annotations
import json
from evac.domain import (
    EvacuationPlan,
    PreparedScenario,
    SemanticIdentity,
    SimulationResult,
    TimelineEvent,
)
from evac.domain.ordering import identifier_key
from evac.errors import InvariantError, ValidationError
from evac.simulation import _core
from evac.simulation._tables import _CompiledScenario

_TERMINAL_REASON_NAME = {
    _core.TERMINAL_FINISHED: "finished",
    _core.TERMINAL_CHARGING_ENERGY_UNAVAILABLE: "charging_energy_unavailable",
    _core.TERMINAL_CHARGING_PORT_UNAVAILABLE: "charging_port_unavailable",
    _core.TERMINAL_CHARGING_QUEUE_OVERFLOW: "charging_queue_overflow",
}
_LIFECYCLE_STATE_NAME = {
    _core.LIFECYCLE_UNAVAILABLE: "unavailable",
    _core.LIFECYCLE_ACTIVE_AT_SITE: "active_at_site",
}
_NAMESPACE_NAME = {
    _core.NAMESPACE_VEHICLE: "vehicle",
    _core.NAMESPACE_EDGE: "edge",
    _core.NAMESPACE_CHARGING_SITE: "charging_site",
    _core.NAMESPACE_MCS_UNIT: "mcs_unit",
}
_EVENT_KIND_NAME = {
    _core.EVENT_VEHICLE_DEPARTED: "vehicle_departed",
    _core.EVENT_EDGE_ENTERED: "edge_entered",
    _core.EVENT_EDGE_ARRIVED: "edge_arrived",
    _core.EVENT_EDGE_ENTRY: "edge_entry",
    _core.EVENT_EDGE_EXIT: "edge_exit",
    _core.EVENT_CHARGING_REQUESTED: "charging_requested",
    _core.EVENT_CHARGING_QUEUED: "charging_queued",
    _core.EVENT_CHARGING_STARTED: "charging_started",
    _core.EVENT_CHARGING_FINISHED: "charging_finished",
    _core.EVENT_CHARGING_REJECTED: "charging_rejected",
    _core.EVENT_VEHICLE_FINISHED: "vehicle_finished",
    _core.EVENT_VEHICLE_ARRIVED: "vehicle_arrived",
    _core.EVENT_VEHICLE_UNFINISHED: "vehicle_unfinished",
    _core.EVENT_MCS_DEPLOYED: "mcs_deployed",
}
_ERROR_MESSAGE = {
    _core.ERROR_DUPLICATE_QUEUE_JOIN: "A vehicle joined the same charging queue twice.",
    _core.ERROR_CHARGE_OVERFILL: "A charging action would overfill the vehicle battery.",
    _core.ERROR_MCS_SESSION_EXCEEDS_RESERVE: "An MCS charging session exceeded its reserved source energy.",
    _core.ERROR_SETTLE_TIME_BACKWARDS: "MCS settlement time moved backwards.",
    _core.ERROR_CHARGE_EXCEEDS_MAX_SOC: "Charging completion exceeded vehicle maximum SOC.",
    _core.ERROR_ROAD_ADMISSION_BELOW_MINIMUM_SOC: "Plan replay would violate the exact vehicle SOC reserve.",
    _core.ERROR_CAPACITY_EXCEEDED: "The compiled core's provable heap/log/batch capacity bound was exceeded; this is an internal invariant failure in the bound's own derivation, not a property of the scenario or plan being simulated.",
    _core.ERROR_KEY_RANGE: "The compiled core's packed heap key/sequence range was exceeded by this scenario's size; this is an internal invariant failure in the packed key's own range bound, not a property of the plan being simulated.",
}
_VALIDATION_ERROR_CODES = frozenset(
    {_core.ERROR_CHARGE_OVERFILL, _core.ERROR_ROAD_ADMISSION_BELOW_MINIMUM_SOC}
)


def _event_key(event: TimelineEvent):
    value = event.subject_value
    return (
        event.time_seconds,
        event.kind,
        event.subject_namespace,
        "int" if isinstance(value, int) else "str",
        str(value),
    )


def _raise_for_error_code(
    error_code: int, tables: _CompiledScenario, vehicle_terminal_reason_code
) -> None:
    if error_code == _core.ERROR_MISSING_TERMINAL_REASON:
        missing = sorted(
            (
                vehicle_id
                for vehicle_id, index in tables.vehicle_index_of_id.items()
                if vehicle_terminal_reason_code[index] == _core.TERMINAL_NONE
            ),
            key=identifier_key,
        )
        raise InvariantError(
            f"Simulation ended with the event queue empty but vehicles lacking a terminal reason: {[item.value for item in missing]!r}."
        )
    message = _ERROR_MESSAGE[error_code]
    if error_code in _VALIDATION_ERROR_CODES:
        raise ValidationError(message)
    raise InvariantError(message)


def _decode_binding(mode_code: int, provider_index: int, tables: _CompiledScenario):
    if mode_code == _core.PROVIDER_KIND_FCS_BINDING_CODE:
        return ("provider_kind", None, "fcs")
    if mode_code == _core.PROVIDER_KIND_MCS_BINDING_CODE:
        return ("provider_kind", None, "mcs")
    return ("provider_id", tables.provider_ids[provider_index].value, None)


def _mcs_terminal_state(
    unit_index: int,
    tables: _CompiledScenario,
    mcs_lifecycle_state,
    mcs_location_node,
    source_energy_kwh: float,
) -> dict:
    state_code = mcs_lifecycle_state[unit_index]
    state_name = _LIFECYCLE_STATE_NAME[state_code]
    result: dict = {"state": state_name, "source_energy_kwh": source_energy_kwh}
    mcs_location_node[unit_index]
    result["site_id"] = tables.site_ids[
        tables.node_to_site_index[mcs_location_node[unit_index]]
    ].value
    return result


def decode_simulation_result(
    prepared: PreparedScenario,
    plan: EvacuationPlan,
    tables: _CompiledScenario,
    raw: tuple,
) -> SimulationResult:
    (
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
    ) = raw
    del last_event_seconds
    if error_code != _core.ERROR_NONE:
        _raise_for_error_code(error_code, tables, vehicle_terminal_reason_code)
    vehicle_events: list[TimelineEvent] = []
    edge_events: list[TimelineEvent] = []
    charging_site_events: list[TimelineEvent] = []
    mcs_events: list[TimelineEvent] = []
    by_namespace = {
        _core.NAMESPACE_VEHICLE: vehicle_events,
        _core.NAMESPACE_EDGE: edge_events,
        _core.NAMESPACE_CHARGING_SITE: charging_site_events,
        _core.NAMESPACE_MCS_UNIT: mcs_events,
    }
    namespace_codes = log_namespace[:log_count].tolist()
    kind_codes = log_kind[:log_count].tolist()
    subject_indices = log_subject[:log_count].tolist()
    times = log_time[:log_count].tolist()
    values_rows = log_v[:log_count].tolist()
    for i in range(log_count):
        namespace_code = namespace_codes[i]
        kind_code = kind_codes[i]
        subject_index = subject_indices[i]
        v = values_rows[i]
        time_seconds = times[i]
        kind = _EVENT_KIND_NAME[kind_code]
        namespace = _NAMESPACE_NAME[namespace_code]
        if namespace_code == _core.NAMESPACE_VEHICLE:
            subject_value = tables.vehicle_ids[subject_index].value
        elif namespace_code == _core.NAMESPACE_EDGE:
            subject_value = tables.edge_ids[subject_index].value
        elif namespace_code == _core.NAMESPACE_CHARGING_SITE:
            subject_value = tables.site_ids[subject_index].value
        else:
            subject_value = tables.mcs_unit_ids[subject_index].value
        if kind_code == _core.EVENT_VEHICLE_DEPARTED:
            values = {"node_id": tables.node_ids[int(v[0])].value, "soc": float(v[1])}
        elif kind_code == _core.EVENT_EDGE_ENTERED:
            values = {"edge_id": tables.edge_ids[int(v[0])].value, "soc": float(v[1])}
        elif kind_code == _core.EVENT_EDGE_ARRIVED:
            values = {
                "node_id": tables.node_ids[int(v[0])].value,
                "edge_id": tables.edge_ids[int(v[1])].value,
                "soc": float(v[2]),
            }
        elif kind_code == _core.EVENT_EDGE_ENTRY:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "occupancy_before": float(v[1]),
                "exit_time_seconds": float(v[3]),
            }
        elif kind_code == _core.EVENT_EDGE_EXIT:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "occupancy": int(v[1]),
            }
        elif kind_code == _core.EVENT_CHARGING_REQUESTED:
            binding_mode, provider_id, provider_kind = _decode_binding(
                int(v[1]), int(v[2]), tables
            )
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "provider_binding_mode": binding_mode,
                "provider_id": provider_id,
                "provider_kind": provider_kind,
            }
        elif kind_code == _core.EVENT_CHARGING_QUEUED:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "queue": int(v[1]),
                "position": int(v[2]),
            }
        elif kind_code == _core.EVENT_CHARGING_STARTED:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "provider_id": tables.provider_ids[int(v[1])].value,
                "queue_wait_seconds": float(v[2]),
                "requested_energy_kwh": float(v[3]),
                "occupancy": int(v[4]),
            }
        elif kind_code == _core.EVENT_CHARGING_FINISHED:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "provider_id": tables.provider_ids[int(v[1])].value,
                "delivered_energy_kwh": float(v[2]),
                "soc": float(v[3]),
                "occupancy": int(v[4]),
            }
        elif kind_code == _core.EVENT_CHARGING_REJECTED:
            values = {
                "vehicle_id": tables.vehicle_ids[int(v[0])].value,
                "reason": _TERMINAL_REASON_NAME[int(v[1])],
            }
        elif kind_code == _core.EVENT_VEHICLE_FINISHED:
            values = {"node_id": tables.node_ids[int(v[0])].value, "soc": float(v[1])}
        elif kind_code == _core.EVENT_VEHICLE_ARRIVED:
            values = {
                "destination": tables.node_ids[int(v[0])].value,
                "soc": float(v[1]),
            }
        elif kind_code == _core.EVENT_VEHICLE_UNFINISHED:
            values = {
                "reason": _TERMINAL_REASON_NAME[int(v[0])],
                "node_id": tables.node_ids[int(v[1])].value,
            }
        elif kind_code == _core.EVENT_MCS_DEPLOYED:
            values = {"site_id": tables.site_ids[int(v[0])].value}
        else:
            values = {"source_energy_kwh": float(v[1])}
        by_namespace[namespace_code].append(
            TimelineEvent(time_seconds, kind, namespace, subject_value, values)
        )
    vehicle_events.sort(key=_event_key)
    edge_events.sort(key=_event_key)
    charging_site_events.sort(key=_event_key)
    mcs_events.sort(key=_event_key)
    vehicle_terminal_reasons = tuple(
        sorted(
            (
                (
                    vehicle_id,
                    _TERMINAL_REASON_NAME[int(vehicle_terminal_reason_code[index])],
                )
                for vehicle_id, index in tables.vehicle_index_of_id.items()
            ),
            key=lambda pair: identifier_key(pair[0]),
        )
    )
    terminal = (
        "complete"
        if all((reason == "finished" for _, reason in vehicle_terminal_reasons))
        else "incomplete"
    )
    mcs_source_energy_kwh = {
        str(unit_id.value): float(mcs_unreserved_source_energy_kwh[index])
        for unit_id, index in sorted(
            tables.mcs_unit_index_of_id.items(),
            key=lambda pair: identifier_key(pair[0]),
        )
    }
    metadata = {
        "simulator": "deterministic_discrete_event",
        "mcs_source_energy_kwh": mcs_source_energy_kwh,
        "mcs_delivered_energy_kwh": {
            str(unit_id.value): float(mcs_delivered_energy_kwh[index])
            for unit_id, index in sorted(
                tables.mcs_unit_index_of_id.items(),
                key=lambda pair: identifier_key(pair[0]),
            )
        },
        "mcs_terminal_states": {
            str(unit_id.value): _mcs_terminal_state(
                index,
                tables,
                mcs_lifecycle_state,
                mcs_location_node,
                mcs_source_energy_kwh[str(unit_id.value)],
            )
            for unit_id, index in tables.mcs_unit_index_of_id.items()
        },
        "finished_at": {
            str(vehicle_id.value): float(vehicle_completion_seconds[index])
            for vehicle_id, index in sorted(
                tables.vehicle_index_of_id.items(),
                key=lambda pair: identifier_key(pair[0]),
            )
            if vehicle_terminal_reason_code[index] == _core.TERMINAL_FINISHED
        },
    }
    identity = SemanticIdentity(
        "simulation",
        json.dumps(
            {
                "preparation_identity": prepared.identity.value,
                "plan_identity": plan.identity.value,
                "event_policy": "same_time_phases_fifo",
                "event_time_tolerance_seconds": tables.tolerance,
                "terminal_reason": terminal,
                "vehicle_terminal_reasons": [
                    (item.to_data(), reason)
                    for item, reason in vehicle_terminal_reasons
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return SimulationResult(
        identity,
        prepared.identity,
        plan.identity,
        tuple(vehicle_events),
        tuple(edge_events),
        tuple(charging_site_events),
        tuple(mcs_events),
        metadata,
        vehicle_terminal_reasons,
        terminal,
    )


__all__ = ["decode_simulation_result"]
