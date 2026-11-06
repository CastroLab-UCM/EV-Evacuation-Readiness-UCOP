from __future__ import annotations
import json
from collections.abc import Iterable
from evac.domain import MCSDeployment, SemanticIdentity, VehiclePlan


def plan_identity(
    preparation_identity: SemanticIdentity,
    vehicle_plans: Iterable[VehiclePlan],
    deployments: Iterable[MCSDeployment],
) -> SemanticIdentity:
    plans = tuple(vehicle_plans)
    deployments = tuple(deployments)
    payload = {
        "preparation_identity": preparation_identity.value,
        "vehicle_plans": [
            {
                "vehicle_id": item.vehicle_id.to_data(),
                "path_id": item.path_id.to_data(),
                "departure_seconds": item.departure_seconds,
                "actions": [
                    {
                        "action_id": action.action_id.to_data(),
                        "provider_id": None
                        if action.provider_id is None
                        else action.provider_id.to_data(),
                        "provider_kind": action.provider_kind,
                        "binding_mode": action.binding_mode.value,
                    }
                    for action in item.charging_actions
                ],
            }
            for item in plans
        ],
        "mcs_deployments": [
            {
                "id": item.id.to_data(),
                "unit": item.mcs_unit_id.to_data(),
                "site": item.site_id.to_data(),
            }
            for item in deployments
        ],
    }
    return SemanticIdentity(
        "plan",
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


__all__ = ["plan_identity"]
