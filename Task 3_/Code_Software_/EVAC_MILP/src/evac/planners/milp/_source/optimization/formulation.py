import copy
import json
import time
import warnings
import numpy as np
import pandas as pd
import numba
from scipy import sparse
from evac.errors import MILPFormulationError
from evac.planners.milp._source.optimization.conversion import EvacuationPlanConverter
from evac.planners.milp._source.optimization.arrays import ragged_to_array
from evac.planners.milp._source.optimization.temporal import (
    ObjectivePlan,
    PathIncidenceInputs,
    charging_window_bounds_numba as _charging_window_bounds_numba,
    ends_past_horizon_numba as _ends_past_horizon_numba,
    propagate_path_timeline,
    propagate_stage_times_numba as _propagate_stage_times_numba,
)
from evac.physics.units import (
    _energy_kwh_from_kw_seconds as energy_kwh_from_kw_seconds,
)
from evac.planners.milp.builder import (
    LinearExpression,
    ScalarVariable,
    SparseMilpBuilder,
    VariableArray,
)
from evac.planners.milp.problem import PlanBinding, VariableType


class BaseSolver:
    def __init__(self, pack, *, tol=1e-06):
        self.verbose = False
        self.traffic = pack["traffic"]
        self.prep_metadata = {
            key: copy.deepcopy(pack.get(key))
            for key in ("grouping", "traffic_model", "model_components")
        }
        self.hyperparameter = pack["input"]["hyperparameter"]
        recharge = self.hyperparameter["recharge"]
        self.recharge_params = {
            "power": recharge["power"],
            "efficiency": recharge["efficiency"],
            "discharge_efficiency": recharge["discharge_efficiency"],
            "amount_time": recharge["amount"]["time"],
            "effective_power": recharge["power"] * recharge["efficiency"],
            "mcs_draw_power": recharge["power"] / recharge["discharge_efficiency"],
        }
        self.demands = pack["input"]["demand"]
        self.supply = pack["input"]["supply"]
        self.paths = {int(k): v for k, v in pack["library"].items()}
        self.base_net = pack["network"]["base"]
        self.aug_net = pack["network"]["augmented"]
        self.dt, self.window_count = self._validated_optimizer_window(
            self.hyperparameter
        )
        self.tol = self._validated_tolerance(tol)
        self._validate_global_charger_power_contract()
        self._setup_sets()
        self._setup_dims()
        self.converter = EvacuationPlanConverter(
            paths=self.paths,
            demands=self.demands,
            supply=self.supply,
            set_data=self.set,
            dt=self.dt,
            node_map=self.node_map,
            horizon_end_time=self.horizon_end_time,
            traffic_metadata=self.prep_metadata["traffic_model"],
            grouping_metadata=self.prep_metadata["grouping"],
        )
        self.solution = None

    @staticmethod
    def _validated_optimizer_window(hyperparameter):
        try:
            window = hyperparameter["optimizer"]["window"]
            raw_size = window["size"]
            raw_number = window["number"]
        except KeyError as exc:
            raise KeyError(
                "Missing required hyperparameter.optimizer.window configuration."
            ) from exc
        if isinstance(raw_size, bool) or not isinstance(
            raw_size, (int, float, np.number)
        ):
            raise TypeError("optimizer.window.size must be numeric.")
        size = float(raw_size)
        if not np.isfinite(size) or size <= 0:
            raise ValueError(
                f"optimizer.window.size must be finite and positive; got {size}."
            )
        if isinstance(raw_number, bool) or not isinstance(
            raw_number, (int, np.integer)
        ):
            raise TypeError("optimizer.window.number must be an integer.")
        number = int(raw_number)
        if number <= 0:
            raise ValueError(f"optimizer.window.number must be positive; got {number}.")
        return (size, number)

    @staticmethod
    def _validated_tolerance(tol):
        if isinstance(tol, bool) or not isinstance(tol, (int, float, np.number)):
            raise TypeError("MILP tolerance must be numeric.")
        value = float(tol)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"MILP tolerance must be finite and nonnegative; got {value}."
            )
        return value

    def _validate_global_charger_power_contract(self):
        if self.supply is None or self.supply.empty:
            return
        if "power" not in self.supply.columns:
            raise KeyError(
                "MCS supply is missing required column 'power'. Current model assumes supply.power equals recharge.power for every MCS."
            )
        try:
            powers = self.supply["power"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("MCS supply.power must contain numeric values.") from exc
        if not np.all(np.isfinite(powers)):
            raise ValueError("MCS supply.power must contain only finite values.")
        recharge_power = self._configured_recharge_power()
        if recharge_power is None:
            raise KeyError(
                "MCS supply power validation requires global recharge.power."
            )
        if not np.isfinite(recharge_power) or recharge_power <= 0:
            raise ValueError(
                f"Global recharge.power must be finite and positive; got {recharge_power}."
            )
        if np.max(np.abs(powers - recharge_power)) > self.tol:
            raise ValueError(
                f"Current charger-service model assumes every MCS supply.power equals global recharge.power={recharge_power}; got {powers.tolist()}. Set supply.power == recharge.power or extend the formulation for provider-specific charging rates."
            )

    def _configured_recharge_power(self):
        return self.recharge_params["power"]

    @staticmethod
    def _shallow_copy_containers(obj):
        if isinstance(obj, dict):
            return {
                key: BaseSolver._shallow_copy_containers(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [BaseSolver._shallow_copy_containers(value) for value in obj]
        return obj

    def report(self, return_all=False):
        if self.solution is None:
            warnings.warn("No solution found to report.", UserWarning)
            return None
        raw_solution_start = time.perf_counter()
        raw_solution = self._shallow_copy_containers(self.solution)
        raw_solution["solution_metadata"] = self._solution_metadata()
        raw_solution["planned_stage_timeline"] = self._selected_planned_stage_timelines(
            raw_solution
        )
        raw_solution_preparation = time.perf_counter() - raw_solution_start
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry["time_raw_solution_preparation"] = raw_solution_preparation
            raw_solution["telemetry"] = copy.deepcopy(telemetry)
        canonical_conversion_start = time.perf_counter()
        evacuation_plan_document = self.converter.to_evacuation_plan_document(
            raw_solution
        )
        canonical_conversion = time.perf_counter() - canonical_conversion_start
        if telemetry is not None:
            telemetry["time_canonical_conversion"] = canonical_conversion
            raw_solution["telemetry"] = copy.deepcopy(telemetry)
            evacuation_plan_document["telemetry"] = copy.deepcopy(telemetry)
        if return_all:
            return (evacuation_plan_document, raw_solution)
        else:
            return evacuation_plan_document

    def _solution_metadata(self):
        return {
            "schema": "raw_mathematical_solution_v1",
            "horizon_end_time": self.horizon_end_time,
            "traffic_model": copy.deepcopy(self.prep_metadata["traffic_model"]),
            "grouping": copy.deepcopy(self.prep_metadata["grouping"]),
            "model_components": copy.deepcopy(self.prep_metadata["model_components"]),
            "mathematical_model": copy.deepcopy(getattr(self, "model_identity", None)),
            "planned_timeline_schema": "conservative_group_stage_timeline_v1",
        }

    def _selected_planned_stage_timelines(self, raw_solution):
        schedule = np.asarray(raw_solution["x"]["int"]["schedule"])
        records = []
        for group_idx, path_idx, window_idx in np.argwhere(schedule > 0):
            group_id = self.set["group"][int(group_idx)]
            path = self.paths[group_id][int(path_idx)]
            departure = float(window_idx * self.dt)
            timeline = self._propagate_path_timeline(
                path,
                departure,
                context=f"selected group={group_id}, path={int(path_idx)}, window={int(window_idx)}",
            )
            records.append(
                {
                    "group_idx": int(group_idx),
                    "group_id": int(group_id),
                    "path_idx": int(path_idx),
                    "window_idx": int(window_idx),
                    "count": int(schedule[group_idx, path_idx, window_idx]),
                    "nodes": path["node"].tolist(),
                    "isreal": path["isreal"].astype(bool).tolist(),
                    "stage_times": timeline.tolist(),
                    "robust_eta": float(timeline[-1]),
                }
            )
        return records

    def _setup_sets(self):
        latest_completion = self._maximum_realized_completion_time()
        window_count = self.window_count
        try:
            configured_horizon = float(
                self.hyperparameter["optimizer"]["horizon_end_time"]
            )
        except KeyError as exc:
            raise KeyError("Missing required optimizer.horizon_end_time.") from exc
        if not np.isfinite(configured_horizon) or configured_horizon <= 0.0:
            raise ValueError("optimizer.horizon_end_time must be finite and positive.")
        horizon_ratio = configured_horizon / self.dt
        horizon_count = int(round(horizon_ratio))
        if abs(horizon_ratio - horizon_count) > self.tol:
            raise ValueError(
                f"optimizer.horizon_end_time must align with optimizer.window.size; got horizon={configured_horizon}, window={self.dt}."
            )
        if configured_horizon + self.tol < latest_completion:
            raise ValueError(
                f"Canonical MCS lifecycle horizon ends before a traffic-realized path can complete: horizon={configured_horizon}, latest_completion={latest_completion}."
            )
        self.horizon_end_time = configured_horizon
        rep_demands = self.demands.loc[self.demands["representative"]]
        self.set = {
            "group": sorted(rep_demands["group_id"].tolist()),
            "od_pair": {
                item.group_id: (item.origin, item.destination)
                for item in rep_demands.itertuples()
            },
            "path": {key: list(range(len(paths))) for key, paths in self.paths.items()},
            "window": np.arange(window_count, dtype=int).tolist(),
            "horizon": np.arange(horizon_count, dtype=int).tolist(),
            "time": (self.dt * np.arange(horizon_count, dtype=int)).tolist(),
            "demand": self.demands.to_dict(orient="list"),
            "supply": self.supply.to_dict(orient="list"),
            "node": {
                "real": sorted(
                    [
                        label
                        for label, attributes in self.aug_net.nodes(data=True)
                        if attributes.get("isreal")
                    ],
                    key=self._node_label_sort_key,
                ),
                "virtual": sorted(
                    [
                        label
                        for label, attributes in self.aug_net.nodes(data=True)
                        if not attributes.get("isreal")
                    ],
                    key=self._node_label_sort_key,
                ),
                "all": sorted(
                    [label for label in self.aug_net.nodes()],
                    key=self._node_label_sort_key,
                ),
            },
        }
        self.set["node"].update(
            {
                "fcs_port": [
                    self.base_net.nodes[node].get("fcs", {}).get("port", 0)
                    for node in self.set["node"]["real"]
                ],
                "mcs_limit": [
                    self.base_net.nodes[node].get("mcs", {}).get("limit", 0)
                    for node in self.set["node"]["real"]
                ],
            }
        )
        self.node_map = {
            "to_index": {
                node_id: ii for ii, node_id in enumerate(self.set["node"]["all"])
            },
            "to_label": {
                ii: node_id for ii, node_id in enumerate(self.set["node"]["all"])
            },
            "virtual_to_index": {
                node_id: ii for ii, node_id in enumerate(self.set["node"]["virtual"])
            },
            "virtual_to_label": {
                ii: node_id for ii, node_id in enumerate(self.set["node"]["virtual"])
            },
        }

    def _propagate_path_timeline(self, path, departure_time, *, context):
        return propagate_path_timeline(
            path,
            departure_time,
            traffic=self.traffic,
            context=context,
        )

    def _maximum_realized_completion_time(self):
        window_count = self.window_count
        if window_count <= 0:
            raise ValueError(
                f"optimizer.window.number must be positive; got {window_count}."
            )
        latest_completion = 0.0
        latest_window_idx = window_count - 1
        for group_id, paths in self.paths.items():
            if not paths:
                raise ValueError(f"Certified group {group_id} has no retained path.")
            members = self.demands.loc[self.demands["group_id"] == group_id]
            od_pairs = sorted(
                {
                    (str(row.origin), str(row.destination))
                    for row in members.itertuples()
                }
            )
            member_ids = members["id"].tolist()
            group_context = f"group={group_id}, od={od_pairs}, member_ids={member_ids}"
            for path_idx, path in enumerate(paths):

                def timeline_at(window_idx):
                    departure = float(window_idx * self.dt)
                    return self._propagate_path_timeline(
                        path,
                        departure,
                        context=f"{group_context}, path={path_idx}, window={window_idx}",
                    )

                first_failing_window = None
                try:
                    timeline = timeline_at(latest_window_idx)
                except ValueError:
                    last_successful_window = -1
                    first_failing_window = latest_window_idx
                    while first_failing_window - last_successful_window > 1:
                        midpoint = (last_successful_window + first_failing_window) // 2
                        try:
                            timeline_at(midpoint)
                        except ValueError:
                            first_failing_window = midpoint
                        else:
                            last_successful_window = midpoint
                if first_failing_window is not None:
                    timeline_at(first_failing_window)
                    raise RuntimeError(
                        "FIFO traffic failure search did not reproduce its failure."
                    )
                latest_completion = max(latest_completion, float(timeline[-1]))
        return latest_completion

    def _setup_dims(self):
        self.dim = {
            "window": self.window_count,
            "time": len(self.set["time"]),
            "group": len(self.set["group"]),
            "path": {key: len(paths) for key, paths in self.set["path"].items()},
            "max_path": max((len(paths) for paths in self.set["path"].values())),
            "node": {
                "real": len(self.set["node"]["real"]),
                "virtual": len(self.set["node"]["virtual"]),
                "all": len(self.set["node"]["all"]),
            },
            "demand": {
                "all": len(self.set["demand"]["id"]),
                "group": {
                    group_id: int((self.demands["group_id"] == group_id).sum())
                    for group_id in self.set["group"]
                },
            },
            "supply": len(self.set["supply"]["id"]),
        }

    def _to_node_index(self, labels):
        if isinstance(labels, str) or not hasattr(labels, "__iter__"):
            return self.node_map["to_index"][labels]
        if isinstance(labels, list):
            return [self._to_node_index(label) for label in labels]
        if isinstance(labels, tuple):
            return tuple((self._to_node_index(label) for label in labels))
        if isinstance(labels, np.ndarray):
            return np.array([self._to_node_index(label) for label in labels])
        if isinstance(labels, pd.Series):
            return labels.map(self.node_map["to_index"])
        else:
            raise TypeError(
                f"Unsupported data type: {type(labels)}. Input must be a string, list, or numpy array."
            )

    def _to_node_label(self, indices):
        if isinstance(indices, int) or not hasattr(indices, "__iter__"):
            return self.node_map["to_label"][indices]
        if isinstance(indices, list):
            return [self._to_node_label(label) for label in indices]
        if isinstance(indices, tuple):
            return tuple((self._to_node_label(label) for label in indices))
        if isinstance(indices, np.ndarray):
            return np.array([self._to_node_label(label) for label in indices])
        if isinstance(indices, pd.Series):
            return indices.map(self.node_map["to_label"])
        else:
            raise TypeError(
                f"Unsupported data type: {type(indices)}. Input must be an integer, list, or numpy array."
            )

    def _node_label_sort_key(self, label):
        text = str(label)

        def _token(value):
            rendered = str(value)
            try:
                return (0, int(rendered), rendered)
            except ValueError:
                return (1, 0, rendered)

        attributes = self.aug_net.nodes[label]
        if attributes.get("isreal"):
            return (0, _token(text), _token(""), text)
        parent = str(attributes.get("parent", text.rsplit("-", 1)[0]))
        suffix = text.rsplit("-", 1)[1] if "-" in text else text
        return (1, _token(parent), _token(suffix), text)


class MILPSolver(BaseSolver):
    def __init__(self, pack, *, tol=1e-06):
        super().__init__(pack, tol=tol)
        self.norm_map = {"l1": 1, "linf": np.inf}
        self.node_branch = {}
        for node_id, attr in self.aug_net.nodes(data=True):
            self.node_branch.setdefault(attr["parent"], []).append(node_id)
        self.model = SparseMilpBuilder("MCS evacuation scheduling")
        self.vtype = {
            "CONTINUOUS": VariableType.CONTINUOUS,
            "INTEGER": VariableType.INTEGER,
            "BINARY": VariableType.BINARY,
        }
        self.vtype_map = {"C": float, "I": int, "B": int}
        self.telemetry = {}
        self.model_identity = {
            "schema": "evac-milp/v1",
            "preparation": pack["grouping"]["source_preparation_identity"],
            "objectives": self.hyperparameter["optimizer"]["objectives"],
        }
        inputs = self._prepare_incidence_generation_inputs()
        generated = self._generate_sparse_cti_direct(
            *self._cti_generation_inputs_from_incidence_inputs(inputs)
        )
        (
            self.cti_sparse,
            self.eta,
            self.charge_event_schedule_sparse,
            self.charge_event_occupancy_sparse,
            self.charge_event_duration_sparse,
            self.charge_event_metadata,
        ) = generated
        self._validate_charge_event_artifacts()
        self.converter.charge_event_metadata = self.charge_event_metadata
        self._setup_dvs()
        self._bind_solver_neutral_variable_blocks()
        self._setup_obj()
        self._setup_constraints()

    @staticmethod
    def _decision_variable_leaves(container, current_path=()):
        leaves = {}
        for key, value in container.items():
            path = current_path + (str(key),)
            if isinstance(value, dict):
                leaves.update(MILPSolver._decision_variable_leaves(value, path))
            elif isinstance(value, (ScalarVariable, VariableArray)):
                leaves[".".join(path)] = value
            else:
                raise TypeError(
                    f"Decision-variable container path {'.'.join(path)} has unsupported type {type(value).__name__}."
                )
        return leaves

    def _bind_solver_neutral_variable_blocks(self):
        for path, variable in self._decision_variable_leaves(self.dvs).items():
            self.model.bind_block_id(variable, path)

    def build_problem(self):
        return self.model.to_problem(
            identity=json.dumps(
                self.model_identity, sort_keys=True, separators=(",", ":")
            ),
            plan_bindings=tuple(
                (
                    PlanBinding(field=path, block_id=path)
                    for path in self._decision_variable_leaves(self.dvs)
                )
            ),
        )

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _count_sparse_cti_entries(
        num_groups,
        max_paths,
        num_windows,
        num_timesteps,
        paths_per_group,
        path_times,
        path_is_real,
        mask_arr,
        dt,
        traffic_times,
        traffic_multipliers,
    ):
        eta = np.zeros((num_groups, max_paths, num_windows), dtype=np.float64)
        row_nnz = np.zeros(num_groups * max_paths * num_windows, dtype=np.int64)
        past_horizon = np.full(5, -1.0)
        for gidx in range(num_groups):
            for pidx in range(paths_per_group[gidx]):
                current_path_times = path_times[gidx][pidx]
                current_path_is_real = path_is_real[gidx][pidx]
                last_valid_index = -1
                for stage in range(len(current_path_is_real)):
                    if not mask_arr[gidx, pidx, stage]:
                        last_valid_index = stage
                for widx in range(num_windows):
                    row = (gidx * max_paths + pidx) * num_windows + widx
                    timeline = _propagate_stage_times_numba(
                        current_path_times,
                        current_path_is_real,
                        mask_arr[gidx, pidx],
                        widx * dt,
                        traffic_times,
                        traffic_multipliers,
                    )
                    if last_valid_index != -1:
                        eta[gidx, pidx, widx] = timeline[last_valid_index]
                    count = 0
                    for stage in range(1, len(current_path_is_real)):
                        if mask_arr[gidx, pidx, stage] or current_path_is_real[stage]:
                            continue
                        if past_horizon[0] < 0 and _ends_past_horizon_numba(
                            timeline[stage], dt, num_timesteps
                        ):
                            past_horizon[0] = gidx
                            past_horizon[1] = pidx
                            past_horizon[2] = widx
                            past_horizon[3] = stage
                            past_horizon[4] = timeline[stage]
                        start, stop = _charging_window_bounds_numba(
                            timeline[stage - 1], timeline[stage], dt, num_timesteps
                        )
                        if stop > start:
                            count += stop - start
                    row_nnz[row] = count
        return (eta, row_nnz, past_horizon)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _fill_sparse_cti_entries(
        num_groups,
        max_paths,
        num_windows,
        num_timesteps,
        paths_per_group,
        path_times,
        path_nodes,
        path_is_real,
        mask_arr,
        dt,
        row_offsets,
        cols,
        data,
        traffic_times,
        traffic_multipliers,
    ):
        for gidx in range(num_groups):
            for pidx in range(paths_per_group[gidx]):
                for widx in range(num_windows):
                    row = (gidx * max_paths + pidx) * num_windows + widx
                    cursor = row_offsets[row]
                    timeline = _propagate_stage_times_numba(
                        path_times[gidx][pidx],
                        path_is_real[gidx][pidx],
                        mask_arr[gidx, pidx],
                        widx * dt,
                        traffic_times,
                        traffic_multipliers,
                    )
                    for stage in range(1, len(path_is_real[gidx][pidx])):
                        if (
                            mask_arr[gidx, pidx, stage]
                            or path_is_real[gidx][pidx][stage]
                        ):
                            continue
                        start, stop = _charging_window_bounds_numba(
                            timeline[stage - 1], timeline[stage], dt, num_timesteps
                        )
                        for tidx in range(start, stop):
                            cols[cursor] = (
                                path_nodes[gidx][pidx][stage] * num_timesteps + tidx
                            )
                            data[cursor] = 1.0
                            cursor += 1

    def _prepare_incidence_generation_inputs(self):
        paths_per_group = np.array(
            [len(self.paths[gid]) for gid in self.set["group"]], dtype=np.int64
        )
        path_times_list = []
        path_nodes_list = []
        path_real_nodes_list = []
        path_is_real_list = []
        mask_list = []
        real_node_index = {
            node: idx for idx, node in enumerate(self.set["node"]["real"])
        }
        for gidx in self.set["group"]:
            group_paths_t = []
            group_paths_n = []
            group_paths_r = []
            group_paths_ir = []
            group_paths_mask = []
            for path in self.paths[gidx]:
                path_nodes = path["node"].tolist()
                path_is_real = path["isreal"].to_numpy(dtype=np.bool_)
                path_virtual_nodes = np.full((len(path),), -1, dtype=np.int64)
                path_real_nodes = np.full((len(path),), -1, dtype=np.int64)
                for node_pos, (node_label, is_real) in enumerate(
                    zip(path_nodes, path_is_real)
                ):
                    if is_real:
                        continue
                    try:
                        path_virtual_nodes[node_pos] = self.node_map[
                            "virtual_to_index"
                        ][node_label]
                    except KeyError as exc:
                        raise KeyError(
                            f"Virtual path node '{node_label}' is missing from the virtual node map."
                        ) from exc
                    parent = self.aug_net.nodes[node_label].get("parent")
                    try:
                        path_real_nodes[node_pos] = real_node_index[parent]
                    except KeyError as exc:
                        raise KeyError(
                            f"Virtual path node '{node_label}' has parent '{parent}', which is missing from the real node map."
                        ) from exc
                group_paths_t.append(
                    np.ascontiguousarray(path["time"].to_numpy(dtype=np.float64))
                )
                group_paths_n.append(path_virtual_nodes)
                group_paths_r.append(path_real_nodes)
                group_paths_ir.append(path_is_real)
                group_paths_mask.append(np.zeros(len(path), dtype=bool))
            path_times_list.append(group_paths_t)
            path_nodes_list.append(group_paths_n)
            path_real_nodes_list.append(group_paths_r)
            path_is_real_list.append(group_paths_ir)
            mask_list.append(group_paths_mask)
        path_times_arr = ragged_to_array(path_times_list, fill=np.nan)
        path_nodes_arr = ragged_to_array(path_nodes_list, fill=-1)
        path_real_nodes_arr = ragged_to_array(path_real_nodes_list, fill=-1)
        path_is_real_arr = ragged_to_array(path_is_real_list, fill=False)
        mask_arr = ragged_to_array(mask_list, fill=True)
        return PathIncidenceInputs(
            paths_per_group=paths_per_group,
            path_times_arr=path_times_arr,
            path_virtual_nodes_arr=path_nodes_arr,
            path_real_nodes_arr=path_real_nodes_arr,
            path_is_real_arr=path_is_real_arr,
            mask_arr=mask_arr,
        )

    @staticmethod
    def _cti_generation_inputs_from_incidence_inputs(inputs):
        return (
            inputs.paths_per_group,
            inputs.path_times_arr,
            inputs.path_virtual_nodes_arr,
            inputs.path_real_nodes_arr,
            inputs.path_is_real_arr,
            inputs.mask_arr,
        )

    def _traffic_kernel_args(self):
        points = self.traffic.exogenous_profile.points
        return (
            np.asarray([point.elapsed_seconds for point in points], dtype=float),
            np.asarray([point.multiplier for point in points], dtype=float),
        )

    def _reject_past_horizon(self, past_horizon, matrix):
        if past_horizon[0] < 0:
            return
        group_idx, path_idx, window_idx, stage = (
            int(value) for value in past_horizon[:4]
        )
        raise MILPFormulationError(
            f"{matrix} cell group={self.set['group'][group_idx]}, path={path_idx}, window={window_idx}, stage={stage} ends at {float(past_horizon[4])} s, past the horizon {self.horizon_end_time} s; the horizon must cover every admissible timeline."
        )

    def _generate_sparse_cti_direct(
        self,
        paths_per_group,
        path_times_arr,
        path_nodes_arr,
        path_real_nodes_arr,
        path_is_real_arr,
        mask_arr,
    ):
        if sparse is None:
            raise ImportError(
                "SciPy sparse is required for direct sparse CTI generation."
            )
        num_rows = self.dim["group"] * self.dim["max_path"] * self.dim["window"]
        num_cols = self.dim["node"]["virtual"] * self.dim["time"]
        eta, row_nnz, past_horizon = self._count_sparse_cti_entries(
            self.dim["group"],
            self.dim["max_path"],
            self.dim["window"],
            self.dim["time"],
            paths_per_group,
            path_times_arr,
            path_is_real_arr,
            mask_arr,
            self.dt,
            *self._traffic_kernel_args(),
        )
        self._reject_past_horizon(past_horizon, "CTI")
        row_offsets = np.empty(num_rows + 1, dtype=np.int64)
        row_offsets[0] = 0
        np.cumsum(row_nnz, dtype=np.int64, out=row_offsets[1:])
        total_nnz = int(row_offsets[-1])
        cols = np.empty(total_nnz, dtype=np.int64)
        data = np.ones(total_nnz, dtype=np.float64)
        self._fill_sparse_cti_entries(
            self.dim["group"],
            self.dim["max_path"],
            self.dim["window"],
            self.dim["time"],
            paths_per_group,
            path_times_arr,
            path_nodes_arr,
            path_is_real_arr,
            mask_arr,
            self.dt,
            row_offsets,
            cols,
            data,
            *self._traffic_kernel_args(),
        )
        cti_sparse = sparse.csr_matrix(
            (data, cols, row_offsets), shape=(num_rows, num_cols), dtype=np.float64
        )
        cti_sparse.sum_duplicates()
        if cti_sparse.nnz > 0:
            cti_sparse.data.fill(1.0)
        cti_sparse.sort_indices()
        (
            charge_event_schedule_sparse,
            charge_event_occupancy_sparse,
            charge_event_duration_sparse,
            charge_event_metadata,
        ) = self._generate_sparse_charge_event_matrices(
            paths_per_group,
            path_times_arr,
            path_real_nodes_arr,
            path_is_real_arr,
            mask_arr,
        )
        return (
            cti_sparse,
            eta,
            charge_event_schedule_sparse,
            charge_event_occupancy_sparse,
            charge_event_duration_sparse,
            charge_event_metadata,
        )

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _count_sparse_charge_event_entries(
        num_groups,
        num_windows,
        num_timesteps,
        paths_per_group,
        path_times,
        path_real_nodes,
        path_is_real,
        mask_arr,
        dt,
        tol,
        traffic_times,
        traffic_multipliers,
    ):
        event_count = 0
        occupancy_nnz = 0
        past_horizon = np.full(5, -1.0)
        for gidx in range(num_groups):
            for pidx in range(paths_per_group[gidx]):
                current_path_times = path_times[gidx][pidx]
                current_path_real_nodes = path_real_nodes[gidx][pidx]
                current_path_is_real = path_is_real[gidx][pidx]
                for widx in range(num_windows):
                    timeline = _propagate_stage_times_numba(
                        current_path_times,
                        current_path_is_real,
                        mask_arr[gidx, pidx],
                        widx * dt,
                        traffic_times,
                        traffic_multipliers,
                    )
                    for stage in range(1, len(current_path_is_real)):
                        if mask_arr[gidx, pidx, stage] or current_path_is_real[stage]:
                            continue
                        if current_path_real_nodes[stage] < 0:
                            continue
                        from_time = timeline[stage - 1]
                        to_time = timeline[stage]
                        if past_horizon[0] < 0 and _ends_past_horizon_numba(
                            to_time, dt, num_timesteps
                        ):
                            past_horizon[0] = gidx
                            past_horizon[1] = pidx
                            past_horizon[2] = widx
                            past_horizon[3] = stage
                            past_horizon[4] = to_time
                        start_tidx, stop_tidx = _charging_window_bounds_numba(
                            from_time, to_time, dt, num_timesteps
                        )
                        if stop_tidx <= start_tidx:
                            continue
                        event_count += 1
                        for tidx in range(start_tidx, stop_tidx):
                            window_start = tidx * dt
                            window_end = window_start + dt
                            overlap = min(to_time, window_end) - max(
                                from_time, window_start
                            )
                            if overlap > tol:
                                occupancy_nnz += 1
        return (event_count, occupancy_nnz, past_horizon)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _fill_sparse_charge_event_entries(
        num_groups,
        max_paths,
        num_windows,
        num_timesteps,
        paths_per_group,
        path_times,
        path_real_nodes,
        path_is_real,
        mask_arr,
        dt,
        tol,
        schedule_cols,
        occupancy_rows,
        occupancy_cols,
        duration_data,
        meta_group_idx,
        meta_path_idx,
        meta_window_idx,
        meta_path_step_idx,
        meta_real_node_idx,
        meta_start_time,
        meta_end_time,
        traffic_times,
        traffic_multipliers,
    ):
        event_idx = 0
        occupancy_idx = 0
        for gidx in range(num_groups):
            for pidx in range(paths_per_group[gidx]):
                current_path_times = path_times[gidx][pidx]
                current_path_real_nodes = path_real_nodes[gidx][pidx]
                current_path_is_real = path_is_real[gidx][pidx]
                for widx in range(num_windows):
                    schedule_row = (gidx * max_paths + pidx) * num_windows + widx
                    timeline = _propagate_stage_times_numba(
                        current_path_times,
                        current_path_is_real,
                        mask_arr[gidx, pidx],
                        widx * dt,
                        traffic_times,
                        traffic_multipliers,
                    )
                    for stage in range(1, len(current_path_is_real)):
                        if mask_arr[gidx, pidx, stage] or current_path_is_real[stage]:
                            continue
                        real_idx = current_path_real_nodes[stage]
                        if real_idx < 0:
                            continue
                        from_time = timeline[stage - 1]
                        to_time = timeline[stage]
                        start_tidx, stop_tidx = _charging_window_bounds_numba(
                            from_time, to_time, dt, num_timesteps
                        )
                        if stop_tidx <= start_tidx:
                            continue
                        schedule_cols[event_idx] = schedule_row
                        meta_group_idx[event_idx] = gidx
                        meta_path_idx[event_idx] = pidx
                        meta_window_idx[event_idx] = widx
                        meta_path_step_idx[event_idx] = stage
                        meta_real_node_idx[event_idx] = real_idx
                        meta_start_time[event_idx] = from_time
                        meta_end_time[event_idx] = to_time
                        for tidx in range(start_tidx, stop_tidx):
                            window_start = tidx * dt
                            window_end = window_start + dt
                            overlap = min(to_time, window_end) - max(
                                from_time, window_start
                            )
                            if overlap <= tol:
                                continue
                            occupancy_rows[occupancy_idx] = (
                                real_idx * num_timesteps + tidx
                            )
                            occupancy_cols[occupancy_idx] = event_idx
                            duration_data[occupancy_idx] = overlap
                            occupancy_idx += 1
                        event_idx += 1
        return (event_idx, occupancy_idx)

    def _generate_sparse_charge_event_matrices(
        self,
        paths_per_group,
        path_times_arr,
        path_real_nodes_arr,
        path_is_real_arr,
        mask_arr,
    ):
        num_schedule_rows = (
            self.dim["group"] * self.dim["max_path"] * self.dim["window"]
        )
        num_real_time_rows = self.dim["node"]["real"] * self.dim["time"]
        tol = getattr(self, "tol", 1e-09)
        event_count, occupancy_nnz, past_horizon = (
            self._count_sparse_charge_event_entries(
                self.dim["group"],
                self.dim["window"],
                self.dim["time"],
                paths_per_group,
                path_times_arr,
                path_real_nodes_arr,
                path_is_real_arr,
                mask_arr,
                self.dt,
                tol,
                *self._traffic_kernel_args(),
            )
        )
        self._reject_past_horizon(past_horizon, "charge-event")
        schedule_cols = np.empty(event_count, dtype=np.int64)
        occupancy_rows = np.empty(occupancy_nnz, dtype=np.int64)
        occupancy_cols = np.empty(occupancy_nnz, dtype=np.int64)
        duration_data = np.empty(occupancy_nnz, dtype=np.float64)
        charge_event_metadata = {
            "group_idx": np.empty(event_count, dtype=np.int64),
            "path_idx": np.empty(event_count, dtype=np.int64),
            "window_idx": np.empty(event_count, dtype=np.int64),
            "path_step_idx": np.empty(event_count, dtype=np.int64),
            "real_node_idx": np.empty(event_count, dtype=np.int64),
            "start_time": np.empty(event_count, dtype=np.float64),
            "end_time": np.empty(event_count, dtype=np.float64),
        }
        filled_events, filled_occupancy = self._fill_sparse_charge_event_entries(
            self.dim["group"],
            self.dim["max_path"],
            self.dim["window"],
            self.dim["time"],
            paths_per_group,
            path_times_arr,
            path_real_nodes_arr,
            path_is_real_arr,
            mask_arr,
            self.dt,
            tol,
            schedule_cols,
            occupancy_rows,
            occupancy_cols,
            duration_data,
            charge_event_metadata["group_idx"],
            charge_event_metadata["path_idx"],
            charge_event_metadata["window_idx"],
            charge_event_metadata["path_step_idx"],
            charge_event_metadata["real_node_idx"],
            charge_event_metadata["start_time"],
            charge_event_metadata["end_time"],
            *self._traffic_kernel_args(),
        )
        if filled_events != event_count or filled_occupancy != occupancy_nnz:
            raise RuntimeError("Charge-event sparse count/fill mismatch.")
        schedule_indptr = np.arange(event_count + 1, dtype=np.int64)
        charge_event_schedule_sparse = sparse.csr_matrix(
            (np.ones(event_count, dtype=np.float64), schedule_cols, schedule_indptr),
            shape=(event_count, num_schedule_rows),
            dtype=np.float64,
        )
        charge_event_occupancy_sparse = sparse.coo_matrix(
            (
                np.ones(occupancy_nnz, dtype=np.float64),
                (occupancy_rows, occupancy_cols),
            ),
            shape=(num_real_time_rows, event_count),
            dtype=np.float64,
        ).tocsr()
        charge_event_duration_sparse = sparse.coo_matrix(
            (duration_data, (occupancy_rows, occupancy_cols)),
            shape=(num_real_time_rows, event_count),
            dtype=np.float64,
        ).tocsr()
        for matrix in (
            charge_event_schedule_sparse,
            charge_event_occupancy_sparse,
            charge_event_duration_sparse,
        ):
            matrix.sum_duplicates()
            matrix.sort_indices()
        if charge_event_occupancy_sparse.nnz > 0:
            charge_event_occupancy_sparse.data.fill(1.0)
        return (
            charge_event_schedule_sparse,
            charge_event_occupancy_sparse,
            charge_event_duration_sparse,
            charge_event_metadata,
        )

    def _expected_sparse_cti_shape(self):
        return (
            self.dim["group"] * self.dim["max_path"] * self.dim["window"],
            self.dim["node"]["virtual"] * self.dim["time"],
        )

    def _extract(self, container, from_callback=False):
        output = {}
        for key, value in container.items():
            if isinstance(value, dict):
                output[key] = self._extract(value, from_callback=from_callback)
            elif isinstance(value, ScalarVariable):
                vtype = self.vtype_map[value.VType]
                val = self.model.cbGetSolution(value) if from_callback else value.X
                if vtype is int:
                    rounded = round(float(val))
                    if abs(float(val) - rounded) > self.tol:
                        raise RuntimeError(
                            f"Accepted integer variable {value.VarName!r} is outside the conversion tolerance: value={val}, tolerance={self.tol}."
                        )
                    output[key] = int(rounded)
                else:
                    output[key] = float(val)
            elif isinstance(value, VariableArray):
                if value.size == 0:
                    output[key] = np.empty(value.shape, dtype=float)
                    continue
                vtype = self.vtype_map[np.ravel(value.VType)[0]]
                arr = self.model.cbGetSolution(value) if from_callback else value.X
                if vtype is int:
                    raw = np.asarray(arr, dtype=float)
                    rounded = np.rint(raw)
                    violation = float(np.max(np.abs(raw - rounded)))
                    if violation > self.tol:
                        raise RuntimeError(
                            f"Accepted integer variable block {value.block_id!r} is outside the conversion tolerance: max_violation={violation}, tolerance={self.tol}."
                        )
                    output[key] = rounded.astype(int)
                else:
                    output[key] = np.asarray(arr, dtype=float)
        return output

    def _validate_charge_event_artifacts(self):
        expected_schedule_cols = (
            self.dim["group"] * self.dim["max_path"] * self.dim["window"]
        )
        expected_real_time_rows = self.dim["node"]["real"] * self.dim["time"]
        if self.charge_event_schedule_sparse is None:
            raise ValueError("Sparse CTI cache is missing charge event schedule data.")
        if self.charge_event_occupancy_sparse is None:
            raise ValueError("Sparse CTI cache is missing charge event occupancy data.")
        if self.charge_event_duration_sparse is None:
            raise ValueError("Sparse CTI cache is missing charge event duration data.")
        event_count = int(self.charge_event_schedule_sparse.shape[0])
        if self.charge_event_schedule_sparse.shape != (
            event_count,
            expected_schedule_cols,
        ):
            raise ValueError(
                f"Charge event schedule shape mismatch: expected ({event_count}, {expected_schedule_cols}), got {self.charge_event_schedule_sparse.shape}."
            )
        expected_event_shape = (expected_real_time_rows, event_count)
        if self.charge_event_occupancy_sparse.shape != expected_event_shape:
            raise ValueError(
                f"Charge event occupancy shape mismatch: expected {expected_event_shape}, got {self.charge_event_occupancy_sparse.shape}."
            )
        if self.charge_event_duration_sparse.shape != expected_event_shape:
            raise ValueError(
                f"Charge event duration shape mismatch: expected {expected_event_shape}, got {self.charge_event_duration_sparse.shape}."
            )
        self.dim["charge_event"] = event_count
        self._validate_charge_event_metadata(event_count)

    def _validate_charge_event_metadata(self, event_count):
        if self.charge_event_metadata is None:
            raise ValueError(
                "Charge event metadata is required for sparse CTI artifacts."
            )
        required = {
            "group_idx",
            "path_idx",
            "window_idx",
            "path_step_idx",
            "real_node_idx",
            "start_time",
            "end_time",
        }
        missing = sorted(required - set(self.charge_event_metadata))
        if missing:
            raise ValueError(
                f"Charge event metadata is missing required fields: {missing}"
            )
        lengths = {}
        for key in required:
            values = np.asarray(self.charge_event_metadata[key])
            lengths[key] = int(values.size)
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Charge event metadata fields must have equal lengths; got {lengths}."
            )
        if next(iter(lengths.values()), 0) != int(event_count):
            raise ValueError(
                f"Charge event metadata length {next(iter(lengths.values()), 0)} does not match charge_event count {event_count}."
            )
        for key in (
            "group_idx",
            "path_idx",
            "window_idx",
            "path_step_idx",
            "real_node_idx",
        ):
            values = np.asarray(self.charge_event_metadata[key], dtype=float)
            if values.size and (not np.all(np.isfinite(values))):
                raise ValueError(
                    f"Charge event metadata field {key} must contain finite values."
                )
            rounded = np.rint(values)
            if values.size and np.max(np.abs(values - rounded)) > self.tol:
                raise ValueError(
                    f"Charge event metadata field {key} must contain integer values."
                )
            self.charge_event_metadata[key] = rounded.astype(np.int64)
        for key in ("start_time", "end_time"):
            values = np.asarray(self.charge_event_metadata[key], dtype=float)
            if values.size and (not np.all(np.isfinite(values))):
                raise ValueError(
                    f"Charge event metadata field {key} must contain finite values."
                )
            self.charge_event_metadata[key] = values.astype(np.float64)
        if event_count > 0 and np.any(
            self.charge_event_metadata["end_time"]
            <= self.charge_event_metadata["start_time"]
        ):
            raise ValueError(
                "Charge event metadata end_time values must be greater than start_time values."
            )

    def _configured_objective_specs(self):
        optimizer_cfg = self.hyperparameter.get("optimizer", {})
        objectives = optimizer_cfg.get("objectives", [])
        if not objectives:
            raise ValueError(
                "No objectives defined in hyperparameter.json. You must specify at least one objective under ['optimizer']['objectives']."
            )
        if not isinstance(objectives, (list, tuple)):
            raise TypeError(
                "optimizer.objectives must be a list of objective dictionaries."
            )
        supported_names = set(getattr(self, "obj_times", ["arrival_time"]))
        supported_norms = set(getattr(self, "norm_map", {"l1": 1, "linf": np.inf}))
        normalized = []
        for idx, obj_spec in enumerate(objectives):
            if not isinstance(obj_spec, dict):
                raise TypeError(f"optimizer.objectives[{idx}] must be a dictionary.")
            spec = dict(obj_spec)
            name = spec.get("name", "arrival_time")
            norm = spec.get("norm", "l1")
            if name not in supported_names:
                raise ValueError(
                    f"Unsupported objective name {name!r}. Supported objectives: {sorted(supported_names)}."
                )
            if norm not in supported_norms:
                raise ValueError(
                    f"Unsupported objective norm {norm!r}. Supported norms: {sorted(supported_norms)}."
                )
            spec["name"] = name
            spec["norm"] = norm
            normalized.append(spec)
        return normalized

    def _build_objective_plan(self, objectives):
        objective_keys = frozenset(
            ((spec["name"], spec["norm"]) for spec in objectives)
        )
        arrival_norms = {
            norm for name, norm in objective_keys if name == "arrival_time"
        }
        needs_linf_envelope = "linf" in arrival_norms
        return ObjectivePlan(
            objective_keys=objective_keys,
            needs_linf_envelope=needs_linf_envelope,
            needs_dispatch_binary=needs_linf_envelope,
        )

    def _objective_specs(self):
        if not hasattr(self, "objective_specs"):
            self.objective_specs = self._configured_objective_specs()
        return self.objective_specs

    def _objective_plan(self):
        if not hasattr(self, "objective_plan"):
            objectives = self._objective_specs()
            self.objective_plan = self._build_objective_plan(objectives)
        return self.objective_plan

    def _mcs_energy_bounds(self):
        batteries = np.asarray(self.set["supply"]["battery"], dtype=float)
        ports = np.asarray(self.set["supply"]["port"], dtype=float)
        initial_soc = np.asarray(self.set["supply"]["soc"], dtype=float)
        minimum_soc = float(self.hyperparameter["soc"]["mcs"]["min"])
        available_energy = batteries * (initial_soc - minimum_soc)
        slot_service_energy = (
            energy_kwh_from_kw_seconds(self.recharge_params["mcs_draw_power"], self.dt)
            * ports
        )
        unit_bounds = np.minimum(available_energy, slot_service_energy)
        descending_bounds = np.sort(unit_bounds)[::-1]
        node_bounds = np.asarray(
            [
                descending_bounds[: min(int(limit), unit_bounds.size)].sum()
                for limit in self.set["node"]["mcs_limit"]
            ],
            dtype=float,
        )
        return (unit_bounds, node_bounds)

    def _group_demand_bounds(self):
        return np.asarray(
            [self.dim["demand"]["group"][group_id] for group_id in self.set["group"]],
            dtype=float,
        )

    def _charge_event_demand_bounds(self):
        event_count = int(self.dim.get("charge_event", 0))
        if event_count == 0:
            return np.zeros(0, dtype=float)
        group_indices = np.asarray(self.charge_event_metadata["group_idx"], dtype=int)
        if group_indices.shape != (event_count,):
            raise RuntimeError(
                "Charge-event group metadata must contain one group index per event."
            )
        return self._group_demand_bounds()[group_indices]

    def _setup_dvs(self):
        charge_event_count = int(self.dim.get("charge_event", 0))
        has_mcs_charge_events = self.dim["supply"] > 0 and charge_event_count > 0
        group_demand_bounds = self._group_demand_bounds()
        schedule_bounds = np.broadcast_to(
            group_demand_bounds.reshape(-1, 1, 1),
            (self.dim["group"], self.dim["max_path"], self.dim["window"]),
        ).copy()
        charge_event_bounds = self._charge_event_demand_bounds()
        objective_plan = self._objective_plan()
        unit_energy_bounds = None
        node_energy_bounds = None
        unit_energy_bounds, node_energy_bounds = (
            self._mcs_energy_bounds()
        )
        self.mcs_unit_energy_bounds = unit_energy_bounds
        x_bin = {}
        if objective_plan.needs_dispatch_binary:
            x_bin["dispatch"] = self.model.add_variable_block(
                (self.dim["group"], self.dim["max_path"], self.dim["window"]),
                lower=0,
                upper=1,
                variable_type=self.vtype["BINARY"],
                name="dispatch",
            )
        x_bin["deploy"] = self.model.add_variable_block(
            (self.dim["supply"], self.dim["node"]["real"]),
            lower=0,
            upper=1,
            variable_type=self.vtype["BINARY"],
            name="deploy",
        )
        aux_int = {
            "virtual_demand": self.model.add_variable_block(
                (self.dim["node"]["virtual"], self.dim["time"]),
                lower=0,
                upper=self.dim["demand"]["all"],
                variable_type=self.vtype["CONTINUOUS"],
                name="aux_virtual_demand",
            ),
            "demand": self.model.add_variable_block(
                (self.dim["node"]["real"], self.dim["time"]),
                lower=0,
                upper=self.dim["demand"]["all"],
                variable_type=self.vtype["CONTINUOUS"],
                name="aux_demand",
            ),
            "deficit": self.model.add_variable_block(
                (self.dim["node"]["real"], self.dim["time"]),
                lower=-np.array(self.set["node"]["fcs_port"]).reshape(-1, 1),
                upper=self.dim["demand"]["all"],
                variable_type=self.vtype["CONTINUOUS"],
                name="aux_deficit",
            ),
        }
        if has_mcs_charge_events:
            site_limit = np.asarray(self.set["node"]["mcs_limit"], dtype=int)[
                self.charge_event_metadata["real_node_idx"]
            ]
            charge_event_bounds = np.where(
                site_limit > 0, charge_event_bounds, 0
            )
            self.telemetry["mcs_charge_events_at_non_deployable_sites"] = int(
                (site_limit == 0).sum()
            )
            aux_int["mcs_charge_event"] = self.model.add_variable_block(
                (charge_event_count,),
                lower=0,
                upper=charge_event_bounds,
                variable_type=self.vtype["INTEGER"],
                name="aux_mcs_charge_event",
            )
        aux_cont = {}
        aux_int.update(
            {
                "track": self.model.add_variable_block(
                    (self.dim["node"]["real"],),
                    lower=0,
                    upper=self.dim["supply"],
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_track",
                ),
                "overflow": self.model.add_variable_block(
                    (self.dim["node"]["real"], self.dim["time"]),
                    lower=0,
                    upper=self.dim["demand"]["all"],
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_overflow",
                ),
                "port": self.model.add_variable_block(
                    (self.dim["node"]["real"],),
                    lower=0,
                    upper=np.asarray(self.set["node"]["fcs_port"], dtype=float)
                    + np.asarray(
                        [
                            np.sort(
                                np.asarray(self.set["supply"]["port"], dtype=float)
                            )[::-1][: min(int(limit), self.dim["supply"])].sum()
                            for limit in self.set["node"]["mcs_limit"]
                        ],
                        dtype=float,
                    ),
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_port",
                ),
            }
        )
        aux_cont.update(
            {
                "node_unit_energy": self.model.add_variable_block(
                    (self.dim["node"]["real"], self.dim["time"]),
                    lower=0,
                    upper=np.broadcast_to(
                        node_energy_bounds.reshape(-1, 1),
                        (self.dim["node"]["real"], self.dim["time"]),
                    ).copy(),
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_node_unit_energy",
                ),
                "mcs_unit_energy": self.model.add_variable_block(
                    (self.dim["supply"], self.dim["node"]["real"], self.dim["time"]),
                    lower=0,
                    upper=np.broadcast_to(
                        unit_energy_bounds.reshape(-1, 1, 1),
                        (
                            self.dim["supply"],
                            self.dim["node"]["real"],
                            self.dim["time"],
                        ),
                    ).copy(),
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_mcs_unit_energy",
                ),
                "mcs_soc": self.model.add_variable_block(
                    (self.dim["supply"], self.dim["time"] + 1),
                    lower=self.hyperparameter["soc"]["mcs"]["min"],
                    upper=self.hyperparameter["soc"]["mcs"]["max"],
                    variable_type=self.vtype["CONTINUOUS"],
                    name="aux_soc",
                ),
            }
        )
        obj_scalar = {}
        if objective_plan.needs_linf_envelope:
            obj_scalar["arrival_time"] = self.model.add_variable(
                lower=0,
                upper=np.inf,
                variable_type=self.vtype["CONTINUOUS"],
                name="arrival_time_scalar",
            )
        self.dvs = {
            "obj": {"scalar": obj_scalar},
            "x": {
                "bin": x_bin,
                "int": {
                    "schedule": self.model.add_variable_block(
                        (self.dim["group"], self.dim["max_path"], self.dim["window"]),
                        lower=0,
                        upper=schedule_bounds,
                        variable_type=self.vtype["INTEGER"],
                        name="schedule",
                    )
                },
            },
            "aux": {"int": aux_int, "cont": aux_cont},
        }

    def _arrival_time_l1_expr(self):
        eta_flat = np.asarray(self.eta, dtype=float).reshape(-1)
        schedule_flat = self.dvs["x"]["int"]["schedule"].reshape(-1)
        return (eta_flat * schedule_flat).sum()

    def _objective_expr(self, name, norm):
        if name != "arrival_time":
            raise ValueError(f"Unsupported objective name {name!r}.")
        if norm == "l1":
            return self._arrival_time_l1_expr()
        if norm == "linf":
            try:
                return self.dvs["obj"]["scalar"]["arrival_time"]
            except KeyError as exc:
                raise RuntimeError(
                    "Objective plan did not build the arrival_time linf scalar variable."
                ) from exc
        raise ValueError(f"Unsupported objective norm {norm!r}.")

    def _primary_objective_expr(self, obj_spec):
        name = obj_spec.get("name", "arrival_time")
        norm = obj_spec.get("norm", "l1")
        obj_spec.get("weight", 1.0)
        return self._objective_expr(name, norm)

    def _mcs_service_energy_objective_expr(self):
        if int(self.dim.get("charge_event", 0)) == 0:
            return LinearExpression.constant(0.0)
        variables = self.dvs["aux"]["int"].get("mcs_charge_event")
        if variables is None:
            return LinearExpression.constant(0.0)
        durations = np.asarray(
            self.charge_event_duration_sparse.sum(axis=0), dtype=float
        ).reshape(-1)
        coefficients = energy_kwh_from_kw_seconds(
            self.recharge_params["mcs_draw_power"], durations
        )
        return coefficients @ variables

    def _setup_obj(self):
        objectives = self._objective_specs()
        if self.dim["supply"] > 0:
            for index, obj_spec in enumerate(objectives):
                name = obj_spec.get("name", "arrival_time")
                norm = obj_spec.get("norm", "l1")
                weight = obj_spec.get("weight", 1.0)
                priority = obj_spec.get("priority", 1)
                if (
                    isinstance(priority, bool)
                    or not isinstance(priority, (int, np.integer))
                    or int(priority) < 1
                ):
                    raise ValueError(
                        "Configured vehicle objective priorities must be positive integers so MCS service cleanup can occupy lower priority 0."
                    )
                self.model.add_objective(
                    self._primary_objective_expr(obj_spec),
                    index=index,
                    priority=int(priority),
                    weight=weight,
                    name=f"obj_{name}_{norm}",
                )
            self.model.add_objective(
                self._mcs_service_energy_objective_expr(),
                index=len(objectives),
                priority=0,
                weight=1.0,
                absolute_degradation=0.0,
                relative_degradation=0.0,
                name="obj_mcs_service_energy",
            )
            if self.verbose:
                print(
                    "Notice: Set vehicle objectives with lower-priority MCS service cleanup"
                )
            return
        if len(objectives) == 1:
            obj_spec = objectives[0]
            name = obj_spec.get("name", "arrival_time")
            norm = obj_spec.get("norm", "l1")
            weight = obj_spec.get("weight", 1.0)
            obj_expr = self._primary_objective_expr(obj_spec)
            self.model.set_objective(weight * obj_expr)
            if self.verbose:
                print(
                    "Notice: Set 1 objective from hyperparameter.json (single-objective mode)"
                )
        else:
            for index, obj_spec in enumerate(objectives):
                name = obj_spec.get("name", "arrival_time")
                norm = obj_spec.get("norm", "l1")
                weight = obj_spec.get("weight", 1.0)
                priority = obj_spec.get("priority", 1)
                obj_expr = self._primary_objective_expr(obj_spec)
                self.model.add_objective(
                    obj_expr,
                    index=index,
                    priority=priority,
                    weight=weight,
                    name=f"obj_{name}_{norm}",
                )
            if self.verbose:
                print(
                    f"Notice: Set {len(objectives)} objective(s) from hyperparameter.json"
                )

    def _build_virtual_branch_matrix(self):
        if sparse is None:
            raise ImportError(
                "SciPy sparse is required for MILPSolver branch aggregation."
            )
        branch_rows = []
        branch_cols = []
        for real_row, node in enumerate(self.set["node"]["real"]):
            for child in self.node_branch[node]:
                if self.aug_net.nodes[child].get("isreal"):
                    continue
                try:
                    child_idx = self.node_map["virtual_to_index"][child]
                except KeyError as exc:
                    raise KeyError(
                        f"Virtual branch node '{child}' is missing from the virtual node map."
                    ) from exc
                branch_rows.append(real_row)
                branch_cols.append(child_idx)
        branch_rows = np.asarray(branch_rows, dtype=int)
        branch_cols = np.asarray(branch_cols, dtype=int)
        branch_data = np.ones(branch_rows.shape[0], dtype=float)
        return sparse.csr_matrix(
            (branch_data, (branch_rows, branch_cols)),
            shape=(self.dim["node"]["real"], self.dim["node"]["virtual"]),
        )

    def _fcs_port_column(self):
        return np.array(self.set["node"]["fcs_port"]).reshape(-1, 1)

    def _setup_overflow_constraints(
        self, overflow=None, deficit=None, *, name="overflow", mode="positive_part"
    ):
        if overflow is None:
            overflow = self.dvs["aux"]["int"]["overflow"]
        if deficit is None:
            deficit = self.dvs["aux"]["int"]["deficit"]
        if mode not in {"positive_part", "lower_bound"}:
            raise ValueError(
                "overflow constraint mode must be 'positive_part' or 'lower_bound'."
            )
        overflow_flat = overflow.reshape(-1)
        deficit_flat = deficit.reshape(-1)
        if overflow_flat.size != deficit_flat.size:
            raise ValueError(
                f"Overflow and deficit sizes must match; got {overflow_flat.size} and {deficit_flat.size}."
            )
        if mode == "lower_bound":
            self.model.add_constraint(overflow_flat >= deficit_flat, name=name)
            return
        for idx in range(overflow_flat.size):
            self.model.add_exact_maximum(
                overflow_flat[idx], (deficit_flat[idx], 0.0), name=f"{name}[{idx}]"
            )

    def _setup_mcs_charge_event_constraints(
        self, mcs_charge_event, mcs_occupancy, mcs_energy, *, mcs_port_capacity, name
    ):
        expected_shape = (self.dim["node"]["real"], self.dim["time"])
        if tuple(mcs_occupancy.shape) != expected_shape:
            raise ValueError(
                f"MCS occupancy must have real-node/time shape {expected_shape}; got {tuple(mcs_occupancy.shape)}."
            )
        if tuple(mcs_energy.shape) != expected_shape:
            raise ValueError(
                f"MCS energy must have real-node/time shape {expected_shape}; got {tuple(mcs_energy.shape)}."
            )
        occupancy_flat = mcs_occupancy.reshape(-1)
        energy_flat = mcs_energy.reshape(-1)
        if occupancy_flat.size != energy_flat.size:
            raise ValueError(
                f"MCS occupancy and energy sizes must match; got {occupancy_flat.size} and {energy_flat.size}."
            )
        capacity_shape = getattr(mcs_port_capacity, "shape", None)
        capacity_size = getattr(mcs_port_capacity, "size", None)
        if capacity_shape is None or capacity_size is None:
            raise TypeError(
                "MCS port capacity must expose one-dimensional shape and size metadata."
            )
        capacity_shape = tuple(capacity_shape)
        capacity_size = int(capacity_size)
        if len(capacity_shape) != 1:
            raise ValueError(
                f"MCS port capacity must be a one-dimensional real-node or real-node/time vector; got shape {capacity_shape}."
            )
        if capacity_size not in (self.dim["node"]["real"], occupancy_flat.size):
            raise ValueError(
                f"MCS port capacity must provide one value per real node or real-node/time row; got shape {capacity_shape} with size {capacity_size}."
            )
        if capacity_size == self.dim["node"]["real"]:
            node_time_broadcast = sparse.kron(
                sparse.eye(self.dim["node"]["real"], format="csr"),
                np.ones((self.dim["time"], 1), dtype=float),
                format="csr",
            )
            capacity_flat = node_time_broadcast @ mcs_port_capacity
        else:
            capacity_flat = mcs_port_capacity
        if int(self.dim.get("charge_event", 0)) == 0:
            lhs, rhs = (occupancy_flat, np.zeros(occupancy_flat.size))
            self.model.add_constraint(lhs == rhs, name=f"{name}_occupancy_zero")
            lhs, rhs = (energy_flat, np.zeros(energy_flat.size))
            self.model.add_constraint(lhs == rhs, name=f"{name}_energy_zero")
            self.model.add_constraint(
                self.dvs["aux"]["int"]["demand"].reshape(-1) - occupancy_flat
                <= np.repeat(
                    np.asarray(self.set["node"]["fcs_port"], dtype=float),
                    self.dim["time"],
                ),
                name=f"{name}_fcs_residual_capacity",
            )
            return
        if mcs_charge_event is None:
            raise RuntimeError(
                "MCS charge event variables are required when charge events exist."
            )
        schedule_flat = self.dvs["x"]["int"]["schedule"].reshape(-1)
        scheduled_event_count = self.charge_event_schedule_sparse @ schedule_flat
        self.model.add_constraint(
            mcs_charge_event <= scheduled_event_count,
            name=f"{name}_charge_event_schedule_limit",
        )
        lhs, rhs = (
            occupancy_flat,
            self.charge_event_occupancy_sparse @ mcs_charge_event,
        )
        self.model.add_constraint(lhs == rhs, name=f"{name}_charge_event_occupancy")
        lhs, rhs = (
            energy_flat,
            energy_kwh_from_kw_seconds(
                self.recharge_params["mcs_draw_power"],
                self.charge_event_duration_sparse @ mcs_charge_event,
            ),
        )
        self.model.add_constraint(lhs == rhs, name=f"{name}_charge_event_energy")
        self.model.add_constraint(
            self.dvs["aux"]["int"]["demand"].reshape(-1) - occupancy_flat
            <= np.repeat(
                np.asarray(self.set["node"]["fcs_port"], dtype=float), self.dim["time"]
            ),
            name=f"{name}_fcs_residual_capacity",
        )
        self.model.add_constraint(
            occupancy_flat <= capacity_flat, name=f"{name}_mcs_port_capacity"
        )

    def _invalid_path_mask(self):
        path_counts = np.array(
            [self.dim["path"][g] for g in self.set["group"]], dtype=int
        )
        valid_paths = (
            np.arange(self.dim["max_path"])[np.newaxis, :] < path_counts[:, np.newaxis]
        )
        valid_mask = np.broadcast_to(
            valid_paths[:, :, np.newaxis],
            (self.dim["group"], self.dim["max_path"], self.dim["window"]),
        )
        return ~valid_mask

    def _setup_path_validity_bounds(self, invalid_mask):
        schedule = self.dvs["x"]["int"]["schedule"]
        schedule[invalid_mask].LB = 0
        schedule[invalid_mask].UB = 0
        dispatch = self.dvs["x"]["bin"].get("dispatch")
        if dispatch is not None:
            dispatch[invalid_mask].LB = 0
            dispatch[invalid_mask].UB = 0

    def _setup_arrival_linf_constraints(self):
        dispatch = self.dvs["x"]["bin"].get("dispatch")
        if dispatch is None:
            raise RuntimeError(
                "arrival_time linf objective requires dispatch binary variables."
            )
        schedule = self.dvs["x"]["int"]["schedule"]
        group_bounds = np.broadcast_to(
            self._group_demand_bounds().reshape(-1, 1, 1), schedule.shape
        )
        self.model.add_constraint(
            schedule <= group_bounds * dispatch, name="schedule_dispatch_upper"
        )
        self.model.add_constraint(schedule >= dispatch, name="schedule_dispatch_lower")
        max_eta = float(np.nanmax(self.eta))
        dispatch_flat = dispatch.reshape(-1)
        eta_flat = self.eta.reshape(-1)
        arrival_expr = (
            self.dvs["obj"]["scalar"]["arrival_time"] - max_eta * dispatch_flat
        )
        self.model.add_constraint(
            arrival_expr >= eta_flat - max_eta, name="arrival_time_scalar"
        )

    def _setup_arrival_objective_constraints(self, schedule_flat):
        objective_plan = self._objective_plan()
        if objective_plan.needs_linf_envelope:
            self._setup_arrival_linf_constraints()

    def _setup_constraints(self):
        general_constraints_start = time.perf_counter()
        if self.verbose:
            print("Setting up general model constraints...")
        invalid_mask = self._invalid_path_mask()
        self._setup_path_validity_bounds(invalid_mask)
        schedule_flat = self.dvs["x"]["int"]["schedule"].reshape(-1)
        self._setup_arrival_objective_constraints(schedule_flat)
        self.model.add_constraint(
            self.dvs["x"]["int"]["schedule"].sum(axis=(1, 2))
            == np.array([self.dim["demand"]["group"][g] for g in self.set["group"]]),
            name="evacuation_size_demand",
        )
        expected_cti_shape = self._expected_sparse_cti_shape()
        if self.cti_sparse.shape != expected_cti_shape:
            raise RuntimeError(
                f"Sparse CTI shape is incompatible with the current mathematical model: got {self.cti_sparse.shape}, expected {expected_cti_shape}."
            )
        lhs, rhs = (
            self.dvs["aux"]["int"]["virtual_demand"].reshape(-1),
            self.cti_sparse.transpose() @ schedule_flat,
        )
        self.model.add_constraint(lhs == rhs, name="virtual_demand")
        A_branch = self._build_virtual_branch_matrix()
        lhs, rhs = (
            self.dvs["aux"]["int"]["demand"],
            A_branch @ self.dvs["aux"]["int"]["virtual_demand"],
        )
        self.model.add_constraint(lhs == rhs, name="lumped_demand")
        fcs_ports = np.array(self.set["node"]["fcs_port"])
        lhs, rhs = (
            self.dvs["aux"]["int"]["deficit"],
            self.dvs["aux"]["int"]["demand"] - self._fcs_port_column(),
        )
        self.model.add_constraint(lhs == rhs, name="demand_deficit")
        self.telemetry["time_general_constraints"] = (
            time.perf_counter() - general_constraints_start
        )
        deployment_constraints_start = time.perf_counter()
        if self.dim["supply"] > 0:
            self.model.add_constraint(
                self.dvs["aux"]["int"]["track"]
                == self.dvs["x"]["bin"]["deploy"].sum(axis=0),
                name="mcs_track",
            )
            mcs_ports = np.array(self.set["supply"]["port"], dtype=int)
            deployed_mcs_ports = (
                mcs_ports.reshape(-1, 1) * self.dvs["x"]["bin"]["deploy"]
            ).sum(axis=0)
            self._setup_mcs_charge_event_constraints(
                self.dvs["aux"]["int"].get("mcs_charge_event"),
                self.dvs["aux"]["int"]["overflow"],
                self.dvs["aux"]["cont"]["node_unit_energy"],
                mcs_port_capacity=deployed_mcs_ports,
                name="mcs",
            )
            fcs_ports = np.array(self.set["node"]["fcs_port"])
            self.model.add_constraint(
                self.dvs["aux"]["int"]["port"]
                == fcs_ports
                + (mcs_ports.reshape(-1, 1) * self.dvs["x"]["bin"]["deploy"]).sum(
                    axis=0
                ),
                name="port_count",
            )
            lhs, rhs = (
                self.dvs["aux"]["int"]["demand"],
                self.dvs["aux"]["int"]["port"].reshape(-1, 1),
            )
            self.model.add_constraint(lhs <= rhs, name="port_limit")
            lhs, rhs = (
                self.dvs["aux"]["int"]["overflow"],
                self.dvs["aux"]["int"]["port"].reshape(-1, 1) - self._fcs_port_column(),
            )
            self.model.add_constraint(lhs <= rhs, name="mcs_port_limit")
            lhs, rhs = (
                self.dvs["aux"]["cont"]["mcs_unit_energy"].sum(axis=0),
                self.dvs["aux"]["cont"]["node_unit_energy"],
            )
            self.model.add_constraint(lhs == rhs, name="mcs_unit_energy_demand")
            self.model.add_constraint(
                self.dvs["aux"]["cont"]["mcs_unit_energy"]
                <= self.mcs_unit_energy_bounds[:, np.newaxis, np.newaxis]
                * self.dvs["x"]["bin"]["deploy"][:, :, np.newaxis],
                name="link_mcs_energy_to_deployment",
            )
            mcs_batteries_reshaped = np.array(
                self.set["supply"]["battery"], dtype=float
            ).reshape(-1, 1)
            delta_soc = (
                self.dvs["aux"]["cont"]["mcs_unit_energy"].sum(axis=1)
                / mcs_batteries_reshaped
            )
            self.model.add_constraint(
                self.dvs["aux"]["cont"]["mcs_soc"][:, 1:]
                == self.dvs["aux"]["cont"]["mcs_soc"][:, :-1] - delta_soc,
                name="mcs_soc_dynamics",
            )
            initial_socs = np.array(self.set["supply"]["soc"], dtype=float)
            self.model.add_constraint(
                self.dvs["aux"]["cont"]["mcs_soc"][:, 0] == initial_socs,
                name="mcs_soc_initial",
            )
            mcs_limit = np.array(self.set["node"]["mcs_limit"])
            self.model.add_constraint(
                self.dvs["aux"]["int"]["track"] <= mcs_limit,
                name="mcs_deployment_site_limit",
            )
            self.model.add_constraint(
                self.dvs["x"]["bin"]["deploy"].sum(axis=1) == 1,
                name="mcs_realistic_deployment",
            )
        else:
            self.model.add_constraint(
                self.dvs["aux"]["int"]["port"] == fcs_ports, name="port_count"
            )
            lhs, rhs = (
                self.dvs["aux"]["int"]["demand"],
                self.dvs["aux"]["int"]["port"].reshape(-1, 1),
            )
            self.model.add_constraint(lhs <= rhs, name="port_limit")
            self.model.add_constraint(
                self.dvs["aux"]["int"]["overflow"] == 0, name="overflow_no_mcs"
            )
            self.model.add_constraint(
                self.dvs["aux"]["cont"]["node_unit_energy"]
                == np.zeros((self.dim["node"]["real"], self.dim["time"])),
                name="node_unit_energy_no_mcs",
            )
        self.telemetry["time_mcs_constraints"] = (
            time.perf_counter() - deployment_constraints_start
        )
