from __future__ import annotations
import json
import math
from collections.abc import Mapping
from itertools import chain
from statistics import fmean
from typing import Any
from evac.domain import (
    EvaluationResult,
    EvaluationViolation,
    PreparedScenario,
    SemanticIdentity,
    SimulationResult,
)
from evac.errors import ValidationError

VehicleKey = tuple[type[str] | type[int], str | int]


def _vehicle_key(value: str | int) -> VehicleKey:
    return (type(value), value)


def _validated_vehicle_outcomes(
    prepared: PreparedScenario, simulation: SimulationResult
) -> tuple[dict[VehicleKey, float], dict[VehicleKey, float], bool]:
    expected = {_vehicle_key(item.id.value): item.id for item in prepared.vehicles}
    terminal = {
        _vehicle_key(vehicle_id.value): reason
        for vehicle_id, reason in simulation.vehicle_terminal_reasons
    }
    if set(terminal) != set(expected):
        raise ValidationError(
            "SimulationResult terminal VehicleIds must exactly match PreparedScenario vehicles."
        )
    departures: dict[VehicleKey, float] = {}
    completions: dict[VehicleKey, float] = {}
    for event in simulation.vehicle_events:
        if event.subject_namespace != "vehicle":
            raise ValidationError(
                "SimulationResult vehicle events must use the vehicle namespace."
            )
        key = _vehicle_key(event.subject_value)
        if key not in expected:
            raise ValidationError(
                "SimulationResult contains an event for an unknown vehicle."
            )
        if event.kind == "vehicle_departed":
            if key in departures:
                raise ValidationError("A vehicle cannot depart more than once.")
            departures[key] = event.time_seconds
        elif event.kind == "vehicle_finished":
            if key in completions:
                raise ValidationError("A vehicle cannot finish more than once.")
            completions[key] = event.time_seconds
    complete = all((reason == "finished" for reason in terminal.values()))
    for key, reason in terminal.items():
        if reason == "finished":
            if key not in departures or key not in completions:
                raise ValidationError(
                    "Every finished vehicle requires exactly one departure and finish event."
                )
            if completions[key] < departures[key]:
                raise ValidationError("A vehicle cannot finish before it departs.")
        elif key in completions:
            raise ValidationError("An unfinished vehicle cannot have a finish event.")
    if complete != (simulation.terminal_reason == "complete"):
        raise ValidationError(
            "SimulationResult overall and per-vehicle terminal reasons are inconsistent."
        )
    return (departures, completions, complete)


def _mean(values: list[float]) -> float | None:
    return None if not values else fmean(values)


def _simulation_end_seconds(simulation: SimulationResult) -> float:
    return max(
        (
            event.time_seconds
            for event in chain(
                simulation.vehicle_events,
                simulation.edge_events,
                simulation.charging_site_events,
                simulation.mcs_events,
            )
        ),
        default=0.0,
    )


def _mcs_available_port_seconds(
    prepared: PreparedScenario,
    simulation: SimulationResult,
    observation_end_seconds: float,
) -> float:
    port_count_by_unit = {item.id.value: item.port_count for item in prepared.mcs_units}
    deployed_at: dict[str | int, float] = {}
    available = 0.0
    for event in simulation.mcs_events:
        unit_id = event.subject_value
        if event.kind == "mcs_deployed":
            if unit_id not in port_count_by_unit:
                raise ValidationError("MCS deployment references an unknown unit.")
            if unit_id in deployed_at:
                raise ValidationError(
                    "An MCS unit cannot be deployed twice concurrently."
                )
            deployed_at[unit_id] = event.time_seconds
        elif event.kind == "mcs_undeployed":
            if unit_id not in port_count_by_unit:
                raise ValidationError("MCS undeployment references an unknown unit.")
            start = deployed_at.pop(unit_id, None)
            if start is None:
                raise ValidationError("An MCS undeployment has no matching deployment.")
            available += (
                max(0.0, min(event.time_seconds, observation_end_seconds) - start)
                * port_count_by_unit[unit_id]
            )
    for unit_id, start in deployed_at.items():
        available += (
            max(0.0, observation_end_seconds - start) * port_count_by_unit[unit_id]
        )
    return available


def _mcs_delivered_energy_kwh(
    prepared: PreparedScenario,
    simulation: SimulationResult,
    completed_session_energy_kwh: float,
) -> float:
    value = simulation.metadata.get("mcs_delivered_energy_kwh")
    if value is None:
        return completed_session_energy_kwh
    if not isinstance(value, Mapping):
        raise ValidationError(
            "Simulation MCS delivered-energy metadata must be a mapping."
        )
    expected = {str(item.id.value) for item in prepared.mcs_units}
    if set(value) != expected:
        raise ValidationError(
            "Simulation MCS delivered-energy metadata must cover every MCS unit exactly."
        )
    total = 0.0
    for item in value.values():
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or (not math.isfinite(float(item)))
            or (item < 0.0)
        ):
            raise ValidationError(
                "Simulation MCS delivered-energy values must be nonnegative numbers."
            )
        total += float(item)
    return total


def _metrics(
    prepared: PreparedScenario,
    simulation: SimulationResult,
    departures: dict[VehicleKey, float],
    completions: dict[VehicleKey, float],
    complete: bool,
) -> dict[str, Any]:
    departure_values = list(departures.values())
    completion_values = list(completions.values())
    durations = [
        completions[key] - departures[key]
        for key in completions.keys() & departures.keys()
    ]
    mcs_provider_ids = {f"mcs:{item.id.value}" for item in prepared.mcs_units}
    active_charging: dict[tuple[str | int, str | int], tuple[float, bool]] = {}
    charging_durations: list[float] = []
    queue_waits: list[float] = []
    mcs_charging_durations: list[float] = []
    mcs_queue_waits: list[float] = []
    mcs_completed_session_energy_kwh = 0.0
    mcs_active_sessions = 0
    mcs_peak_concurrent_sessions = 0
    peak_queue = 0
    for event in simulation.charging_site_events:
        vehicle_id = event.values.get("vehicle_id")
        provider_id = event.values.get("provider_id")
        key = (vehicle_id, provider_id)
        is_mcs = provider_id in mcs_provider_ids
        if event.kind == "charging_queued":
            peak_queue = max(peak_queue, int(event.values["queue"]))
        elif event.kind == "charging_started":
            active_charging[key] = (event.time_seconds, is_mcs)
            queue_wait = float(event.values["queue_wait_seconds"])
            queue_waits.append(queue_wait)
            if is_mcs:
                mcs_queue_waits.append(queue_wait)
                mcs_active_sessions += 1
                mcs_peak_concurrent_sessions = max(
                    mcs_peak_concurrent_sessions, mcs_active_sessions
                )
        elif event.kind == "charging_finished":
            active = active_charging.pop(key, None)
            if active is None:
                raise ValidationError(
                    "Charging completion has no matching start event."
                )
            start, started_on_mcs = active
            duration = event.time_seconds - start
            charging_durations.append(duration)
            if started_on_mcs:
                mcs_charging_durations.append(duration)
                mcs_completed_session_energy_kwh += float(
                    event.values["delivered_energy_kwh"]
                )
                mcs_active_sessions -= 1
    observation_end_seconds = _simulation_end_seconds(simulation)
    mcs_occupied_port_seconds = sum(mcs_charging_durations) + sum(
        (
            observation_end_seconds - start
            for start, is_mcs in active_charging.values()
            if is_mcs
        )
    )
    mcs_available_port_seconds = _mcs_available_port_seconds(
        prepared, simulation, observation_end_seconds
    )
    mcs_port_time_utilization = (
        None
        if mcs_available_port_seconds == 0.0
        else mcs_occupied_port_seconds / mcs_available_port_seconds
    )
    return {
        "vehicle_count": len(prepared.vehicles),
        "finished_vehicle_count": len(completions),
        "mean_departure_seconds": _mean(departure_values),
        "max_departure_seconds": max(departure_values, default=None),
        "mean_completion_seconds": _mean(completion_values) if complete else None,
        "max_completion_seconds": max(completion_values) if complete else None,
        "mean_evacuation_duration_seconds": _mean(durations) if complete else None,
        "max_evacuation_duration_seconds": max(durations) if complete else None,
        "charging_session_count": len(charging_durations),
        "total_charging_duration_seconds": sum(charging_durations),
        "total_queue_wait_seconds": sum(queue_waits),
        "peak_queue_vehicles": peak_queue,
        "mcs_charging_session_count": len(mcs_charging_durations),
        "mcs_delivered_energy_kwh": _mcs_delivered_energy_kwh(
            prepared, simulation, mcs_completed_session_energy_kwh
        ),
        "mcs_total_charging_duration_seconds": sum(mcs_charging_durations),
        "mcs_total_queue_wait_seconds": sum(mcs_queue_waits),
        "mcs_peak_concurrent_sessions": mcs_peak_concurrent_sessions,
        "mcs_available_port_seconds": mcs_available_port_seconds,
        "mcs_occupied_port_seconds": mcs_occupied_port_seconds,
        "mcs_port_time_utilization": mcs_port_time_utilization,
    }


def _violations(simulation: SimulationResult) -> tuple[EvaluationViolation, ...]:
    return tuple(
        (
            EvaluationViolation(
                "non_completion", "vehicle", vehicle_id.value, 1.0, reason
            )
            for vehicle_id, reason in simulation.vehicle_terminal_reasons
            if reason != "finished"
        )
    )


def evaluate(
    prepared_scenario: PreparedScenario, simulation_result: SimulationResult
) -> EvaluationResult:
    if not isinstance(prepared_scenario, PreparedScenario) or not isinstance(
        simulation_result, SimulationResult
    ):
        raise ValidationError(
            "evaluate requires a PreparedScenario and SimulationResult."
        )
    if simulation_result.preparation_identity != prepared_scenario.identity:
        raise ValidationError(
            "SimulationResult is bound to a different PreparedScenario."
        )
    departures, completions, complete = _validated_vehicle_outcomes(
        prepared_scenario, simulation_result
    )
    completion_values = list(completions.values())
    objective = prepared_scenario.evaluation.objectives[0]
    objective_name = f"completion_time/{objective.norm}"
    if not complete:
        scientific_objective = None
    elif objective.norm == "mean":
        scientific_objective = fmean(completion_values)
    elif objective.norm == "max":
        scientific_objective = max(completion_values)
    else:
        raise ValidationError(
            f"Unsupported completion_time reducer {objective.norm!r}."
        )
    violations = _violations(simulation_result)
    metrics = _metrics(
        prepared_scenario, simulation_result, departures, completions, complete
    )
    identity = SemanticIdentity(
        "evaluation",
        json.dumps(
            {
                "simulation_identity": simulation_result.identity.value,
                "objective_name": objective_name,
                "scientific_objective": scientific_objective,
                "violations": [
                    (
                        item.kind,
                        item.subject_namespace,
                        item.subject_value,
                        item.magnitude,
                        item.detail,
                    )
                    for item in violations
                ],
                "metrics": metrics,
                "event_time_tolerance_seconds": prepared_scenario.numerical_policy.event_time_tolerance_seconds,
                "formula_version": "canonical_completion_time_v2",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return EvaluationResult(
        identity,
        simulation_result.identity,
        objective_name,
        scientific_objective,
        complete and (not violations),
        violations,
        metrics,
    )


__all__ = ["evaluate"]
