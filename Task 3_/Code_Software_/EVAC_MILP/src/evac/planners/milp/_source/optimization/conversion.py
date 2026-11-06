import copy
import numpy as np


class EvacuationPlanConverter:
    def __init__(
        self,
        *,
        paths,
        demands,
        supply,
        set_data,
        dt,
        node_map,
        horizon_end_time,
        traffic_metadata,
        grouping_metadata,
    ):
        self.paths = paths
        self.demands = demands
        self.supply = supply
        self.set = set_data
        self.dt = dt
        self.node_map = node_map
        self.horizon_end_time = self._require_horizon(horizon_end_time)
        self.traffic_metadata = copy.deepcopy(traffic_metadata)
        self.grouping_metadata = copy.deepcopy(grouping_metadata)
        self.charge_event_metadata = None
        self.demands_by_group = {
            gid: df for gid, df in self.demands.groupby("group_id")
        }

    @staticmethod
    def _require_horizon(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(
                f"horizon_end_time must be numeric; got {type(value).__name__}."
            )
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"horizon_end_time must be finite and positive; got {value}."
            )
        return value

    def to_evacuation_plan_document(self, solution):
        schedule = solution["x"]["int"]["schedule"]
        deploy = solution["x"]["bin"].get("deploy", np.empty((0, 0, 0), dtype=int))
        metadata = solution.get("solution_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Raw solution is missing required solution_metadata.")
        t_max = self._require_horizon(metadata.get("horizon_end_time"))
        if t_max != self.horizon_end_time:
            raise ValueError(
                f"Raw solution horizon_end_time does not match the solver-owned converter horizon: {t_max} != {self.horizon_end_time}."
            )
        if metadata.get("traffic_model") != self.traffic_metadata:
            raise ValueError(
                "Raw solution traffic metadata does not match converter context."
            )
        timeline_lookup = self._planned_timeline_lookup(
            solution.get("planned_stage_timeline")
        )
        evacuation_plan_document = {
            "vehicle": {},
            "mcs": {},
            "metadata": {
                "schema": "canonical_operational_solution_v1",
                "horizon_end_time": t_max,
                "traffic_model": copy.deepcopy(self.traffic_metadata),
                "grouping": copy.deepcopy(self.grouping_metadata),
                "model_components": copy.deepcopy(metadata.get("model_components")),
                "mathematical_model": copy.deepcopy(metadata.get("mathematical_model")),
                "planned_timeline_schema": metadata.get("planned_timeline_schema"),
            },
        }
        mcs_charge_event_lookup = None
        remaining_mcs_charge_events = {}
        mcs_charge_event_counts = self._mcs_charge_event_counts(solution)
        if mcs_charge_event_counts is not None:
            mcs_charge_event_lookup, remaining_mcs_charge_events = (
                self._mcs_charge_event_lookup(mcs_charge_event_counts)
            )
        if "telemetry" in solution:
            evacuation_plan_document["telemetry"] = solution["telemetry"]
        departures = np.argwhere(schedule)
        group_offset = [0] * len(self.set["group"])
        for grp_idx, path_idx, dep_idx in departures:
            size = int(schedule[grp_idx, path_idx, dep_idx])
            group_id = self.set["group"][grp_idx]
            path = self.paths[group_id][path_idx]
            timeline_key = (int(grp_idx), int(path_idx), int(dep_idx))
            if timeline_key not in timeline_lookup:
                raise ValueError(
                    f"Raw solution is missing planned stage timeline {timeline_key}."
                )
            planned = timeline_lookup[timeline_key]
            stage_times = np.asarray(planned["stage_times"], dtype=float)
            if stage_times.shape != (len(path),):
                raise ValueError(
                    f"Planned stage timeline {timeline_key} has shape {stage_times.shape}; expected {(len(path),)}."
                )
            fleet_start = group_offset[grp_idx]
            charge_step_indices = np.argwhere(~path.isreal.to_numpy(dtype=bool)).ravel()
            mcs_service_count_by_step = {}
            if mcs_charge_event_lookup is not None:
                for c_idx in charge_step_indices:
                    event_key = (int(grp_idx), int(path_idx), int(dep_idx), int(c_idx))
                    selected_count = int(mcs_charge_event_lookup.get(event_key, 0))
                    if selected_count > size:
                        raise ValueError(
                            f"Selected MCS charge-event count exceeds scheduled fleet size for group={grp_idx}, path={path_idx}, window={dep_idx}, path_step={c_idx}: selected={selected_count}, scheduled={size}."
                        )
                    if selected_count > 0:
                        remaining_mcs_charge_events.pop(event_key, None)
                    mcs_service_count_by_step[int(c_idx)] = selected_count
            for demand_index in range(fleet_start, size + fleet_start):
                vehicle_charge = []
                order_in_fleet = demand_index - fleet_start
                for c_idx in charge_step_indices:
                    charge_entry = {
                        "node": path.node.iloc[c_idx - 1],
                        "time": float(stage_times[c_idx] - stage_times[c_idx - 1]),
                        "planned_start": float(stage_times[c_idx - 1]),
                        "planned_end": float(stage_times[c_idx]),
                    }
                    if mcs_charge_event_lookup is not None:
                        charge_entry["service"] = (
                            "mcs"
                            if order_in_fleet
                            < mcs_service_count_by_step.get(int(c_idx), 0)
                            else "fcs"
                        )
                    vehicle_charge.append(charge_entry)
                vehicle_plan = {
                    "route": path.node[path.isreal].tolist(),
                    "charge": vehicle_charge,
                    "departure": float(dep_idx * self.dt),
                    "group_id": int(group_id),
                    "group_index": int(grp_idx),
                    "path_index": int(path_idx),
                    "window_index": int(dep_idx),
                    "planned_timeline": {
                        "schema": "conservative_group_stage_timeline_v1",
                        "nodes": path.node.tolist(),
                        "isreal": path.isreal.astype(bool).tolist(),
                        "stage_times": stage_times.tolist(),
                        "robust_eta": float(planned["robust_eta"]),
                    },
                }
                evacuation_plan_document["vehicle"].update(
                    {
                        self.demands_by_group[group_id].id.iloc[
                            demand_index
                        ]: vehicle_plan
                    }
                )
            group_offset[grp_idx] += size
        if remaining_mcs_charge_events:
            sample_key, sample_count = next(iter(remaining_mcs_charge_events.items()))
            raise ValueError(
                f"Selected MCS charge-event count could not be matched to a scheduled canonical charge entry: event={sample_key}, count={sample_count}."
            )
        if self.supply.empty or deploy.size == 0:
            pass
        else:
            for mcs_idx in self.set["supply"]["id"]:
                mcs_deploy_schedule = deploy[int(mcs_idx)]
                deploy_node = np.argmax(mcs_deploy_schedule)
                node_label = self.node_map["to_label"][deploy_node]
                mcs_decision = {
                    "service_mode": "pooled_node",
                    "node": node_label,
                }
                evacuation_plan_document["mcs"].update({mcs_idx: mcs_decision})
        return evacuation_plan_document

    to_canonical = to_evacuation_plan_document

    @staticmethod
    def _planned_timeline_lookup(records):
        if not isinstance(records, list):
            raise TypeError("planned_stage_timeline must be a list.")
        lookup = {}
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("planned_stage_timeline entries must be dictionaries.")
            key = (
                int(record["group_idx"]),
                int(record["path_idx"]),
                int(record["window_idx"]),
            )
            if key in lookup:
                raise ValueError(f"Duplicate planned stage timeline {key}.")
            lookup[key] = record
        return lookup

    def _mcs_charge_event_counts(self, solution):
        aux_int = solution.get("aux", {}).get("int", {})
        if isinstance(aux_int, dict) and "mcs_charge_event" in aux_int:
            return self._as_integer_array(
                aux_int["mcs_charge_event"], "mcs_charge_event"
            ).reshape(-1)
        return None

    def _mcs_charge_event_lookup(self, charge_event_counts):
        metadata = self._validated_charge_event_metadata(
            self.charge_event_metadata, event_count=int(charge_event_counts.size)
        )
        lookup = {}
        remaining_selected = {}
        for event_idx, selected_count in enumerate(charge_event_counts):
            selected_count = int(selected_count)
            if selected_count < 0:
                raise ValueError(
                    f"MCS charge-event count must be nonnegative; got {selected_count}."
                )
            if selected_count == 0:
                continue
            event_key = (
                int(metadata["group_idx"][event_idx]),
                int(metadata["path_idx"][event_idx]),
                int(metadata["window_idx"][event_idx]),
                int(metadata["path_step_idx"][event_idx]),
            )
            lookup[event_key] = lookup.get(event_key, 0) + selected_count
            remaining_selected[event_key] = (
                remaining_selected.get(event_key, 0) + selected_count
            )
        return (lookup, remaining_selected)

    @staticmethod
    def _validated_charge_event_metadata(metadata, *, event_count):
        if metadata is None:
            raise ValueError("MCS charge-event counts require charge_event_metadata.")
        if not isinstance(metadata, dict):
            raise TypeError("charge_event_metadata must be a dictionary.")
        required = {
            "group_idx",
            "path_idx",
            "window_idx",
            "path_step_idx",
            "real_node_idx",
            "start_time",
            "end_time",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise KeyError(
                f"charge_event_metadata is missing required fields: {missing}"
            )
        arrays = {key: np.asarray(metadata[key]) for key in required}
        lengths = {key: int(value.size) for key, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"charge_event_metadata fields must have equal lengths; got {lengths}."
            )
        actual_count = next(iter(lengths.values()), 0)
        if actual_count != int(event_count):
            raise ValueError(
                f"charge_event_metadata length {actual_count} does not match charge_event count {event_count}."
            )
        for key in (
            "group_idx",
            "path_idx",
            "window_idx",
            "path_step_idx",
            "real_node_idx",
        ):
            values = arrays[key]
            if values.size and (not np.all(np.isfinite(values.astype(float)))):
                raise ValueError(
                    f"charge_event_metadata field {key} must contain finite values."
                )
            rounded = np.rint(values.astype(float))
            if values.size and np.max(np.abs(values.astype(float) - rounded)) > 1e-06:
                raise ValueError(
                    f"charge_event_metadata field {key} must contain integer values."
                )
            arrays[key] = rounded.astype(int)
        for key in ("start_time", "end_time"):
            arrays[key] = arrays[key].astype(float)
            if arrays[key].size and (not np.all(np.isfinite(arrays[key]))):
                raise ValueError(
                    f"charge_event_metadata field {key} must contain finite values."
                )
        if arrays["start_time"].size and np.any(
            arrays["end_time"] <= arrays["start_time"]
        ):
            raise ValueError(
                "charge_event_metadata end_time values must be greater than start_time values."
            )
        return arrays

    @staticmethod
    def _as_integer_array(value, name):
        arr = np.asarray(value)
        rounded = np.rint(arr)
        if arr.size and np.max(np.abs(arr - rounded)) > 1e-06:
            raise ValueError(
                f"MCS charging count '{name}' must contain integer values."
            )
        return rounded.astype(int)


SolutionConverter = EvacuationPlanConverter
