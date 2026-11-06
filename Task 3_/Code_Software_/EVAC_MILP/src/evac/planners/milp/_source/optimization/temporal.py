"""Time incidence calculations for the MILP."""

from __future__ import annotations
from dataclasses import dataclass
import numba
import numpy as np
from evac.physics.traffic_kernel import traffic_exit_time_kernel


@dataclass(frozen=True)
class ObjectivePlan:
    objective_keys: frozenset
    needs_linf_envelope: bool
    needs_dispatch_binary: bool


@dataclass(frozen=True)
class PathIncidenceInputs:
    paths_per_group: np.ndarray
    path_times_arr: np.ndarray
    path_virtual_nodes_arr: np.ndarray
    path_real_nodes_arr: np.ndarray
    path_is_real_arr: np.ndarray
    mask_arr: np.ndarray


@numba.jit(nopython=True, cache=True)
def charging_window_bounds_numba(from_time, to_time, dt, num_timesteps):
    start_tidx = int(np.floor(from_time / dt))
    stop_tidx = int(np.ceil(to_time / dt))
    if start_tidx < 0:
        start_tidx = 0
    if stop_tidx > num_timesteps:
        stop_tidx = num_timesteps
    if stop_tidx < start_tidx:
        stop_tidx = start_tidx
    return (start_tidx, stop_tidx)


@numba.jit(nopython=True, cache=True)
def ends_past_horizon_numba(to_time, dt, num_timesteps):
    """Whether an interval ending at `to_time` needs a slot past the last of
    `num_timesteps`, the entry `charging_window_bounds_numba` would clip."""
    return int(np.ceil(to_time / dt)) > num_timesteps


@numba.njit(cache=True)
def propagate_stage_times_numba(
    base_times,
    path_is_real,
    mask,
    departure_time,
    traffic_times,
    traffic_multipliers,
):
    timeline = np.empty(base_times.shape[0], dtype=np.float64)
    timeline[:] = np.nan
    previous = departure_time
    for stage in range(base_times.shape[0]):
        if mask[stage]:
            continue
        if stage == 0:
            timeline[stage] = departure_time
            previous = departure_time
            continue
        duration = base_times[stage] - base_times[stage - 1]
        if path_is_real[stage]:
            previous = traffic_exit_time_kernel(
                traffic_times,
                traffic_multipliers,
                previous,
                duration,
            )
        else:
            previous = previous + duration
        timeline[stage] = previous
    return timeline


def propagate_path_timeline(path, departure_time, *, traffic, context):
    """Frozen stage timeline of one library path departing at `departure_time`:
    road stages exit through the shared traffic clock,
    charging stages add their fixed duration."""
    base_times = path["time"].to_numpy(dtype=float)
    is_real = path["isreal"].to_numpy(dtype=bool)
    if base_times.ndim != 1 or len(base_times) != len(is_real) or len(base_times) == 0:
        raise ValueError(f"{context} path timing arrays are malformed.")
    durations = np.diff(base_times, prepend=base_times[0])
    if base_times[0] != 0.0 or np.any(~np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError(
            f"{context} requires finite nondecreasing free-flow path times beginning at zero."
        )
    timeline = np.empty(len(base_times), dtype=float)
    timeline[0] = float(departure_time)
    for stage in range(1, len(base_times)):
        try:
            timeline[stage] = (
                traffic.exit_time(
                    entry_time_seconds=timeline[stage - 1],
                    free_flow_duration_seconds=durations[stage],
                )
                if is_real[stage]
                else timeline[stage - 1] + durations[stage]
            )
        except ValueError as exc:
            raise ValueError(
                f"{context} traffic failure at path stage {stage}: {exc}"
            ) from exc
    return timeline
