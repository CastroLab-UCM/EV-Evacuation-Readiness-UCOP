from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from evac.domain import EvacuationPlan, PreparedScenario, ProviderBindingMode
from evac.domain.ordering import identifier_key
from evac.errors import InvariantError
from evac.simulation._tables import _CompiledScenario

PROVIDER_KIND_FCS_BINDING_CODE = 1
PROVIDER_KIND_MCS_BINDING_CODE = 2
PROVIDER_ID_BINDING_CODE = 3


@dataclass(frozen=True, slots=True)
class _RunInputs:
    path_index: np.ndarray
    departure_seconds: np.ndarray
    vehicle_action_offsets: np.ndarray
    binding_mode_code: np.ndarray
    bound_provider_index: np.ndarray
    slot_rank: np.ndarray
    unit_index: np.ndarray
    deployment_site_index: np.ndarray
    deployment_ids: tuple[object, ...]


def _slot_rank_from_plan(
    tables: _CompiledScenario, plan: EvacuationPlan, prepared: PreparedScenario
) -> np.ndarray:
    group_index_of_vehicle: dict[object, int] = {}
    path_rank_of_vehicle: dict[object, dict[object, int]] = {}
    for group_index, group in enumerate(prepared.demand_groups):
        path_rank = {
            path_id: rank for rank, path_id in enumerate(group.candidate_path_ids)
        }
        for vehicle_id in group.member_ids:
            group_index_of_vehicle[vehicle_id] = group_index
            path_rank_of_vehicle[vehicle_id] = path_rank
    windows = prepared.departure_windows

    def _window_index(departure_seconds: float, vehicle_id: object) -> int:
        for window_index, window in enumerate(windows):
            if window.start_seconds <= departure_seconds < window.end_seconds:
                return window_index
        raise InvariantError(
            f"Vehicle {vehicle_id.value!r} departs at {departure_seconds!r}, which is not contained in any departure window."
        )

    vehicle_plans = {item.vehicle_id: item for item in plan.vehicle_plans}
    sort_keys = []
    for vehicle_id, vehicle_index in tables.vehicle_index_of_id.items():
        vehicle_plan = vehicle_plans[vehicle_id]
        if vehicle_id not in group_index_of_vehicle:
            raise InvariantError(
                f"Vehicle {vehicle_id.value!r} does not belong to any demand group."
            )
        group_index = group_index_of_vehicle[vehicle_id]
        path_rank = path_rank_of_vehicle[vehicle_id].get(vehicle_plan.path_id)
        if path_rank is None:
            raise InvariantError(
                f"Vehicle {vehicle_id.value!r}'s path {vehicle_plan.path_id.value!r} is not a member of its demand group's candidate_path_ids."
            )
        window_index = _window_index(vehicle_plan.departure_seconds, vehicle_id)
        sort_keys.append(
            (
                group_index,
                path_rank,
                window_index,
                identifier_key(vehicle_id),
                vehicle_index,
            )
        )
    slot_rank = np.empty(len(sort_keys), dtype=np.int64)
    for rank, key in enumerate(sorted(sort_keys)):
        slot_rank[key[-1]] = rank
    return slot_rank


def build_run_inputs(
    tables: _CompiledScenario, plan: EvacuationPlan, prepared: PreparedScenario
) -> _RunInputs:
    n_vehicles = len(tables.vehicle_ids)
    path_index = np.empty(n_vehicles, dtype=np.int64)
    departure_seconds = np.empty(n_vehicles, dtype=np.float64)
    vehicle_action_offsets = np.zeros(n_vehicles + 1, dtype=np.int64)
    binding_mode_list: list[int] = []
    bound_provider_list: list[int] = []
    vehicle_plans = {item.vehicle_id: item for item in plan.vehicle_plans}
    for vehicle_id, vehicle_index in tables.vehicle_index_of_id.items():
        vehicle_plan = vehicle_plans[vehicle_id]
        path_index[vehicle_index] = tables.path_index_of_id[vehicle_plan.path_id]
        departure_seconds[vehicle_index] = vehicle_plan.departure_seconds
        for binding in vehicle_plan.charging_actions:
            if binding.binding_mode is ProviderBindingMode.PROVIDER_KIND:
                if binding.provider_kind == "fcs":
                    binding_mode_list.append(PROVIDER_KIND_FCS_BINDING_CODE)
                elif binding.provider_kind == "mcs":
                    binding_mode_list.append(PROVIDER_KIND_MCS_BINDING_CODE)
                else:
                    raise InvariantError(
                        f"Unsupported provider_kind {binding.provider_kind!r}."
                    )
                bound_provider_list.append(-1)
            elif binding.binding_mode is ProviderBindingMode.PROVIDER_ID:
                binding_mode_list.append(PROVIDER_ID_BINDING_CODE)
                bound_provider_list.append(
                    tables.provider_index_of_id[binding.provider_id]
                )
            else:
                raise InvariantError(
                    "Charging action has an unsupported provider binding mode."
                )
        vehicle_action_offsets[vehicle_index + 1] = len(binding_mode_list)
    binding_mode_code = np.array(binding_mode_list, dtype=np.int64)
    bound_provider_index = np.array(bound_provider_list, dtype=np.int64)
    deployments = plan.mcs_deployments
    n_deployments = len(deployments)
    unit_index = np.empty(n_deployments, dtype=np.int64)
    deployment_site_index = np.empty(n_deployments, dtype=np.int64)

    for index, deployment in enumerate(deployments):
        unit_index[index] = tables.mcs_unit_index_of_id[deployment.mcs_unit_id]
        deployment_site_index[index] = tables.site_index_of_id[deployment.site_id]
    return _RunInputs(
        path_index=path_index,
        departure_seconds=departure_seconds,
        vehicle_action_offsets=vehicle_action_offsets,
        binding_mode_code=binding_mode_code,
        bound_provider_index=bound_provider_index,
        slot_rank=_slot_rank_from_plan(tables, plan, prepared),
        unit_index=unit_index,
        deployment_site_index=deployment_site_index,
        deployment_ids=tuple((item.id for item in deployments)),
    )


__all__ = ["build_run_inputs"]
