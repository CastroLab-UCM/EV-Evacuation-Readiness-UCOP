from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from evac.domain import (
    CandidatePath,
    ChargingAction,
    ChargingProviderId,
    ChargingSiteId,
    EdgeId,
    EvacuationPlan,
    MCSUnitId,
    MCSUnitSpec,
    PathId,
    PhysicalEdge,
    PreparedScenario,
    ProviderBindingMode,
    SimulationResult,
    VehicleId,
    VehiclePlan,
    VehicleSpec,
)
from evac.errors import InvariantError, ValidationError
from evac.physics import mcs_unit_total_power_kw
from evac.simulation._events import decode_simulation_result
from evac.simulation._run_inputs import build_run_inputs
from evac.simulation._state import SimulationState, run_capacities
from evac.simulation._tables import compiled_scenario
from evac.simulation._core import run_core
from evac.simulation.inspection import PlanInspection, PlanInspectionIssue


@dataclass(frozen=True, slots=True)
class _PreparedSimulationContext:
    prepared: PreparedScenario
    edges: dict[EdgeId, PhysicalEdge]
    vehicles: dict[VehicleId, VehicleSpec]
    paths: dict[PathId, CandidatePath]
    mcs_units: dict[MCSUnitId, MCSUnitSpec]
    path_route_nodes: dict[PathId, tuple[Any, ...]]
    energy_trajectories: dict[
        PathId, tuple[tuple[ChargingAction | None, PhysicalEdge], ...]
    ]
    group_paths: dict[VehicleId, frozenset[PathId]]
    fixed_provider_bindings: dict[
        ChargingProviderId, frozenset[tuple[ChargingSiteId, str]]
    ]


def _prepare_simulation_context(
    prepared: PreparedScenario, *, eager_paths: bool = False
) -> _PreparedSimulationContext:
    edges = {item.id: item for item in prepared.network.edges}
    vehicles = {item.id: item for item in prepared.vehicles}
    paths = {item.id: item for item in prepared.candidate_paths}
    path_route_nodes = {}
    energy_trajectories = {}
    if eager_paths:
        for path in prepared.candidate_paths:
            route_nodes = _route_nodes(edges, path)
            path_route_nodes[path.id] = route_nodes
            actions = {item.site_id.value: item for item in path.charging_actions}
            energy_trajectories[path.id] = tuple(
                (
                    (actions.get(node.value), edges[edge_id])
                    for node, edge_id in zip(
                        route_nodes[:-1], path.edge_ids, strict=True
                    )
                )
            )
    group_paths = {
        vehicle_id: frozenset(group.candidate_path_ids)
        for group in prepared.demand_groups
        for vehicle_id in group.member_ids
    }
    fixed_provider_bindings = {
        provider.id: frozenset(((provider.site_id, provider.kind),))
        for node in prepared.network.nodes
        for provider in node.charging_providers
    }
    return _PreparedSimulationContext(
        prepared,
        edges,
        vehicles,
        paths,
        {item.id: item for item in prepared.mcs_units},
        path_route_nodes,
        energy_trajectories,
        group_paths,
        fixed_provider_bindings,
    )


def _route_nodes(
    edges: dict[EdgeId, PhysicalEdge], path: CandidatePath
) -> tuple[Any, ...]:
    first = edges[path.edge_ids[0]]
    result = [first.source]
    for edge_id in path.edge_ids:
        edge = edges[edge_id]
        if edge.source != result[-1]:
            raise ValidationError("CandidatePath edges are not contiguous.")
        result.append(edge.target)
    return tuple(result)


def _validate_homogeneous_provider_pool(
    prepared: PreparedScenario,
    plan: EvacuationPlan,
    action: ChargingAction,
    provider_kind: str,
) -> None:
    fingerprints: list[tuple[Any, ...]] = []
    for node in prepared.network.nodes:
        if node.id.value != action.site_id.value:
            continue
        fingerprints.extend(
            (
                ("fixed", provider.port_count, provider.power_kw, provider.efficiency)
                for provider in node.charging_providers
                if provider.kind == provider_kind
            )
        )
    if provider_kind == "mcs":
        units = {item.id: item for item in prepared.mcs_units}
        for deployment in plan.mcs_deployments:
            if deployment.site_id != action.site_id:
                continue
            unit = units[deployment.mcs_unit_id]
            fingerprints.append(
                (
                    "mcs",
                    unit.port_count,
                    mcs_unit_total_power_kw(
                        unit.power_kw, unit.port_count, unit.power_is_unit_total
                    ),
                    unit.charging_efficiency,
                    unit.discharge_efficiency,
                    unit.battery_kwh,
                    unit.initial_soc,
                    unit.minimum_soc,
                    unit.maximum_soc,
                )
            )
    if not fingerprints:
        raise ValidationError("VehiclePlan provider pool has no concrete provider.")
    if len(set(fingerprints)) != 1:
        raise ValidationError(
            "VehiclePlan provider pool is heterogeneous and cannot hide concrete dispatch."
        )


def validate_plan(prepared: PreparedScenario, plan: EvacuationPlan) -> None:
    _validate_plan(_prepare_simulation_context(prepared), plan)


def inspect_plan(prepared: PreparedScenario, plan: EvacuationPlan) -> PlanInspection:
    context = _prepare_simulation_context(prepared)
    return PlanInspection(
        prepared.identity,
        plan.identity,
        _plan_validation_issues(context, plan, collect_all=True),
    )


def _validate_plan(
    context: _PreparedSimulationContext,
    plan: EvacuationPlan,
) -> None:
    issues = _plan_validation_issues(
        context,
        plan,
        collect_all=False,
    )
    if issues:
        raise ValidationError(issues[0].detail)


def _plan_validation_issues(
    context: _PreparedSimulationContext,
    plan: EvacuationPlan,
    *,
    collect_all: bool,
) -> tuple[PlanInspectionIssue, ...]:
    prepared = context.prepared
    vehicles = context.vehicles
    issues: list[PlanInspectionIssue] = []
    if plan.preparation_identity != prepared.identity:
        detail = "EvacuationPlan is bound to a different PreparedScenario."
        return (_inspection_issue(detail, "plan", plan.identity.value),)
    if {item.vehicle_id for item in plan.vehicle_plans} != set(vehicles):
        detail = "EvacuationPlan must contain exactly one plan for every vehicle."
        issues.append(_inspection_issue(detail, "plan", plan.identity.value))
        if not collect_all:
            return tuple(issues)
    provider_ids = {
        provider_id: set(bindings)
        for provider_id, bindings in context.fixed_provider_bindings.items()
    }
    for deployment in plan.mcs_deployments:
        provider_ids.setdefault(
            ChargingProviderId(f"mcs:{deployment.mcs_unit_id.value}"), set()
        ).add((deployment.site_id, "mcs"))
    for vehicle_plan in plan.vehicle_plans:
        if vehicle_plan.vehicle_id not in vehicles:
            continue
        try:
            _validate_vehicle_plan(
                context,
                plan,
                vehicle_plan,
                provider_ids,
            )
        except ValidationError as exc:
            issues.append(
                _inspection_issue(str(exc), "vehicle", vehicle_plan.vehicle_id.value)
            )
            if not collect_all:
                return tuple(issues)
    try:
        _validate_deployments(prepared, plan)
    except ValidationError as exc:
        issues.append(_inspection_issue(str(exc), "plan", plan.identity.value))
    return tuple(issues)


def _validate_vehicle_plan(
    context: _PreparedSimulationContext,
    plan: EvacuationPlan,
    vehicle_plan: VehiclePlan,
    provider_ids: dict[ChargingProviderId, set[tuple[ChargingSiteId, str]]],
) -> None:
    prepared = context.prepared
    if vehicle_plan.path_id not in context.group_paths[vehicle_plan.vehicle_id]:
        raise ValidationError(
            "VehiclePlan selects a path not certified for its demand group."
        )
    path = context.paths[vehicle_plan.path_id]
    vehicle = context.vehicles[vehicle_plan.vehicle_id]
    if path.origin != vehicle.origin or path.destination != vehicle.destination:
        raise ValidationError("VehiclePlan path has the wrong origin or destination.")
    if len(vehicle_plan.charging_actions) != len(path.charging_actions):
        raise ValidationError(
            "VehiclePlan must bind every charging action exactly once."
        )
    for planned, action in zip(
        vehicle_plan.charging_actions, path.charging_actions, strict=True
    ):
        if planned.action_id != action.id:
            raise ValidationError(
                "VehiclePlan charging-action order does not match its path."
            )
        provider_id = planned.provider_id
        provider_kind = planned.provider_kind
        binding_mode = planned.binding_mode
        if binding_mode is ProviderBindingMode.PROVIDER_KIND:
            if provider_kind is None:
                raise InvariantError("Provider-kind binding has no provider kind.")
            if provider_kind not in action.eligible_provider_kinds:
                raise ValidationError(
                    "VehiclePlan charging provider kind is not eligible for its action."
                )
            if not any(
                (
                    site_id == action.site_id and kind == provider_kind
                    for bindings in provider_ids.values()
                    for site_id, kind in bindings
                )
            ):
                raise ValidationError(
                    "VehiclePlan charging provider kind has no provider at the action site."
                )
            _validate_homogeneous_provider_pool(prepared, plan, action, provider_kind)
            continue
        if binding_mode is not ProviderBindingMode.PROVIDER_ID:
            raise InvariantError(
                "Charging action has an unsupported provider binding mode."
            )
        if provider_id is None:
            raise InvariantError("Concrete provider binding has no provider ID.")
        if provider_id not in provider_ids:
            raise ValidationError(
                "VehiclePlan references an unknown charging provider."
            )
        valid_bindings = provider_ids[provider_id]
        if not any(
            (
                site_id == action.site_id and kind in action.eligible_provider_kinds
                for site_id, kind in valid_bindings
            )
        ):
            raise ValidationError(
                "VehiclePlan charging provider is not eligible at the action site."
            )
    if not any(
        (
            window.start_seconds <= vehicle_plan.departure_seconds < window.end_seconds
            for window in prepared.departure_windows
        )
    ):
        raise ValidationError(
            "Vehicle departure lies outside the canonical departure windows."
        )
    trajectory = context.energy_trajectories.get(path.id)
    if trajectory is None:
        route_nodes = _route_nodes(context.edges, path)
        context.path_route_nodes[path.id] = route_nodes
        actions = {item.site_id.value: item for item in path.charging_actions}
        trajectory = tuple(
            (
                (actions.get(node.value), context.edges[edge_id])
                for node, edge_id in zip(route_nodes[:-1], path.edge_ids, strict=True)
            )
        )
        context.energy_trajectories[path.id] = trajectory
    _validate_vehicle_energy(vehicle, trajectory)


def _inspection_issue(
    detail: str, subject_namespace: str, subject_value: str | int
) -> PlanInspectionIssue:
    return PlanInspectionIssue(
        _PLAN_ISSUE_CODES.get(detail, "plan_invalid"),
        detail,
        subject_namespace,
        subject_value,
    )


_PLAN_ISSUE_CODES = {
    "EvacuationPlan is bound to a different PreparedScenario.": "preparation_identity_mismatch",
    "EvacuationPlan must contain exactly one plan for every vehicle.": "vehicle_plan_cardinality",
    "VehiclePlan selects a path not certified for its demand group.": "uncertified_path",
    "VehiclePlan path has the wrong origin or destination.": "path_endpoint_mismatch",
    "VehiclePlan must bind every charging action exactly once.": "charging_action_cardinality",
    "VehiclePlan charging-action order does not match its path.": "charging_action_order",
    "VehiclePlan automatic charging binding has no eligible provider at the action site.": "automatic_provider_unavailable",
    "VehiclePlan charging provider kind is not eligible for its action.": "provider_kind_ineligible",
    "VehiclePlan charging provider kind has no provider at the action site.": "provider_kind_unavailable",
    "VehiclePlan provider pool has no concrete provider.": "provider_pool_empty",
    "VehiclePlan provider pool is heterogeneous and cannot hide concrete dispatch.": "provider_pool_heterogeneous",
    "VehiclePlan references an unknown charging provider.": "provider_unknown",
    "VehiclePlan charging provider is not eligible at the action site.": "provider_ineligible",
    "Vehicle departure lies outside the canonical departure windows.": "departure_window",
    "Vehicle charging trajectory violates exact SOC bounds.": "vehicle_charge_soc",
    "Vehicle route trajectory violates exact SOC reserve.": "vehicle_route_soc",
    "MCS deployment references an unknown unit.": "mcs_unit_unknown",
    "MCS deployment references an ineligible site.": "mcs_site_ineligible",
    "MCS site capacity is exceeded.": "mcs_site_capacity",
    "Each MCS must have exactly one deployment.": "mcs_deployment_count",
}


def _validate_vehicle_energy(
    vehicle: VehicleSpec,
    trajectory: tuple[tuple[ChargingAction | None, PhysicalEdge], ...],
) -> None:
    soc = vehicle.initial_soc
    for action, edge in trajectory:
        if action is not None:
            soc += action.requested_energy_kwh / vehicle.battery_kwh
            if soc > vehicle.maximum_soc or soc < action.required_post_charge_soc:
                raise ValidationError(
                    "Vehicle charging trajectory violates exact SOC bounds."
                )
        soc -= edge.distance_m / vehicle.efficiency_m_per_kwh / vehicle.battery_kwh
        if soc < vehicle.minimum_soc:
            raise ValidationError(
                "Vehicle route trajectory violates exact SOC reserve."
            )


def _validate_deployments(prepared: PreparedScenario, plan: EvacuationPlan) -> None:
    units = {item.id for item in prepared.mcs_units}
    assigned = set()
    counts = {}
    for deployment in plan.mcs_deployments:
        if deployment.mcs_unit_id not in units:
            raise ValidationError("MCS deployment references an unknown unit.")
        if deployment.mcs_unit_id in assigned:
            raise ValidationError("Each MCS must have exactly one deployment.")
        assigned.add(deployment.mcs_unit_id)
        if deployment.site_id not in prepared.mcs_deployment_domain.eligible_sites:
            raise ValidationError("MCS deployment references an ineligible site.")
        counts[deployment.site_id] = counts.get(deployment.site_id, 0) + 1
    if assigned != units:
        raise ValidationError("Each MCS must have exactly one deployment.")
    for site, limit in prepared.mcs_deployment_domain.site_limits:
        if counts.get(site, 0) > limit:
            raise ValidationError("MCS site capacity is exceeded.")


def _run_core_from_tables(
    tables, run_inputs, state: SimulationState, *, record_events: bool
):
    return run_core(
        tables.vehicle_battery_kwh,
        tables.vehicle_initial_soc,
        tables.vehicle_minimum_soc,
        tables.vehicle_maximum_soc,
        tables.vehicle_efficiency_m_per_kwh,
        tables.vehicle_speed_m_per_second,
        tables.edge_distance_m,
        tables.edge_speed_limit_m_per_second,
        tables.edge_source_node_index,
        tables.edge_target_node_index,
        tables.path_edge_offsets,
        tables.path_edge_index,
        tables.path_action_offsets,
        tables.action_site_index,
        tables.action_requested_energy_kwh,
        tables.action_eligible_fcs,
        tables.action_eligible_mcs,
        tables.provider_kind_code,
        tables.provider_port_count,
        tables.provider_power_kw,
        tables.provider_charging_efficiency,
        tables.provider_mcs_unit_index,
        tables.provider_source_to_battery_efficiency,
        tables.provider_site_index,
        tables.fcs_site_offsets,
        tables.fcs_site_members,
        tables.mcs_battery_kwh,
        tables.mcs_initial_soc,
        tables.mcs_minimum_soc,
        tables.mcs_maximum_soc,
        tables.provider_index_of_mcs_unit,
        tables.traffic_times,
        tables.traffic_multipliers,
        tables.tolerance,
        tables.queue_policy_code,
        tables.queue_capacity,
        tables.node_to_site_index,
        tables.site_to_node_index,
        run_inputs.path_index,
        run_inputs.departure_seconds,
        run_inputs.vehicle_action_offsets,
        run_inputs.binding_mode_code,
        run_inputs.bound_provider_index,
        run_inputs.slot_rank,
        run_inputs.unit_index,
        run_inputs.deployment_site_index,
        record_events,
        state.vehicle_f64,
        state.vehicle_i64,
        state.vehicle_bool,
        state.provider_f64,
        state.provider_i64,
        state.provider_bool,
        state.provider_session_vehicles,
        state.mcs_f64,
        state.mcs_i64,
        state.request_vehicle_index,
        state.request_action_slot,
        state.request_binding_slot,
        state.request_time,
        state.request_next,
        state.queue_head,
        state.queue_tail,
        state.dirty_site,
        state.active_edge_count,
        state.charge_req_vehicle,
        state.charge_req_action_slot,
        state.charge_req_binding_slot,
        state.charge_req_time,
        state.road_req_vehicle,
        state.road_req_edge_progress,
        state.scratch_candidates,
        state.scratch_road_order,
        state.scratch_edge_of,
        state.scratch_free_flow,
        state.scratch_charge_order,
        state.h_time,
        state.h_key,
        state.h_kind,
        state.h_payload,
        state.b_time,
        state.b_key,
        state.b_kind,
        state.b_payload,
        state.b_order,
        state.log_time,
        state.log_kind,
        state.log_namespace,
        state.log_subject,
        state.log_v,
        state.scalars_i64,
        state.scalars_f64,
    )


def _simulate_via_core(
    prepared: PreparedScenario, plan: EvacuationPlan
) -> SimulationResult:
    tables = compiled_scenario(prepared)
    run_inputs = build_run_inputs(tables, plan, prepared)
    state = SimulationState.allocate(
        run_capacities(tables, run_inputs, record_events=True)
    )
    raw = _run_core_from_tables(tables, run_inputs, state, record_events=True)
    return decode_simulation_result(prepared, plan, tables, raw)


def simulate(prepared: PreparedScenario, plan: EvacuationPlan) -> SimulationResult:
    _validate_plan(_prepare_simulation_context(prepared), plan)
    return _simulate_via_core(prepared, plan)


__all__ = ["inspect_plan", "simulate", "validate_plan"]
