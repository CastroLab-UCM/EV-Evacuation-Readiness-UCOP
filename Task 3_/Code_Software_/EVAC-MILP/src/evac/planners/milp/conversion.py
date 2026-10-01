from __future__ import annotations
from typing import Any, Mapping
from evac.domain import (
    ChargingSiteId,
    DeploymentId,
    EvacuationPlan,
    MCSDeployment,
    PlannedChargingAction,
    ProviderBindingMode,
    VehiclePlan,
)
from evac.errors import InvariantError
from evac.planning.construction import plan_identity
from evac.planners.milp.compiler import MILPProjection


def canonical_plan_from_source(
    projection: MILPProjection, source_document: Mapping[str, Any]
) -> EvacuationPlan:
    prepared = projection.prepared
    source_vehicles = source_document.get("vehicle")
    source_mcs = source_document.get("mcs")
    if not isinstance(source_vehicles, Mapping) or not isinstance(source_mcs, Mapping):
        raise InvariantError("Source MILP result is missing vehicle or MCS decisions.")
    vehicle_plans: list[VehiclePlan] = []
    for raw_source_id, source_plan in source_vehicles.items():
        source_id = int(raw_source_id)
        if source_id not in projection.vehicle_by_source_id:
            raise InvariantError(f"MILP returned unknown vehicle ID {source_id}.")
        if not isinstance(source_plan, Mapping):
            raise InvariantError("Source MILP vehicle decisions must be mappings.")
        group_id = int(source_plan["group_id"])
        path_index = int(source_plan["path_index"])
        try:
            path = projection.path_by_source_index[group_id, path_index]
        except KeyError as exc:
            raise InvariantError(
                f"Source MILP selected unknown group/path {(group_id, path_index)!r}."
            ) from exc
        expected_route = _source_real_route(projection, path)
        if list(source_plan.get("route", ())) != expected_route:
            raise InvariantError(
                "Source MILP route cannot be losslessly mapped to its Path ID."
            )
        source_actions = source_plan.get("charge", ())
        if not isinstance(source_actions, list) or len(source_actions) != len(
            path.charging_actions
        ):
            raise InvariantError(
                "Source MILP charge decisions do not match the canonical Path."
            )
        bindings: list[PlannedChargingAction] = []
        for action, source_action in zip(
            path.charging_actions, source_actions, strict=True
        ):
            if not isinstance(source_action, Mapping):
                raise InvariantError("Source MILP charge decision must be a mapping.")
            node = projection.node_by_source_label.get(str(source_action.get("node")))
            if node is None or (type(node.value), node.value) != (
                type(action.site_id.value),
                action.site_id.value,
            ):
                raise InvariantError(
                    "Source MILP charge location does not match its Path action."
                )
            service = source_action.get("service")
            if service is None:
                if len(action.eligible_provider_kinds) == 1:
                    service = action.eligible_provider_kinds[0]
                elif not prepared.mcs_units and "fcs" in action.eligible_provider_kinds:
                    service = "fcs"
                else:
                    raise InvariantError(
                        "Source MILP omitted a provider-pool decision needed for lossless conversion."
                    )
            if service not in action.eligible_provider_kinds:
                raise InvariantError(
                    "Source MILP selected an ineligible charging service pool."
                )
            bindings.append(
                PlannedChargingAction(
                    action.id,
                    provider_kind=str(service),
                    binding_mode=ProviderBindingMode.PROVIDER_KIND,
                )
            )
        vehicle_plans.append(
            VehiclePlan(
                projection.vehicle_by_source_id[source_id],
                path.id,
                float(source_plan["departure"]),
                tuple(bindings),
            )
        )
    deployments: list[MCSDeployment] = []
    for raw_source_id, source_plan in source_mcs.items():
        source_id = int(raw_source_id)
        if source_id not in projection.mcs_by_source_id:
            raise InvariantError(f"MILP returned unknown MCS ID {source_id}.")
        service_mode = "pooled_node"
        if (
            not isinstance(source_plan, Mapping)
            or source_plan.get("service_mode") != service_mode
        ):
            raise InvariantError(
                "Source MILP MCS service mode is not canonical pooled-node service."
            )
        node = _mapped_node(projection, source_plan.get("node"), label="deployment node")
        unit_id = projection.mcs_by_source_id[source_id]
        deployments.append(
            MCSDeployment(
                DeploymentId(f"milp:{unit_id.value}:0"),
                unit_id,
                ChargingSiteId(node.value),
            )
        )
    vehicle_plans.sort(key=lambda item: _identifier_sort_key(item.vehicle_id.value))
    deployments.sort(
        key=lambda item: (
            _identifier_sort_key(item.mcs_unit_id.value),
            _identifier_sort_key(item.site_id.value),
        )
    )
    identity = plan_identity(prepared.identity, vehicle_plans, deployments)
    result = EvacuationPlan(
        identity, prepared.identity, tuple(vehicle_plans), tuple(deployments)
    )
    from evac.simulation import validate_plan

    validate_plan(prepared, result)
    return result


def _source_real_route(projection: MILPProjection, path: Any) -> list[str]:
    edge_by_id = {item.id: item for item in projection.prepared.network.edges}
    first = edge_by_id[path.edge_ids[0]]
    nodes = [first.source]
    for edge_id in path.edge_ids:
        edge = edge_by_id[edge_id]
        if edge.source != nodes[-1]:
            raise InvariantError("Canonical Path is not contiguous.")
        nodes.append(edge.target)
    labels_by_node = {
        node: label for label, node in projection.node_by_source_label.items()
    }
    return [labels_by_node[node] for node in nodes]


def _mapped_node(projection: MILPProjection, value: object, *, label: str) -> Any:
    node = projection.node_by_source_label.get(str(value))
    if node is None:
        raise InvariantError(f"Source MILP {label} {value!r} is unknown.")
    return node


def _identifier_sort_key(value: str | int) -> tuple[str, str]:
    return (type(value).__name__, str(value))
