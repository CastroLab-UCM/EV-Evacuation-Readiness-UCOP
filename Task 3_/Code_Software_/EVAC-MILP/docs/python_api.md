# Python API

Install the package as described in the [README](../README.md), then run this example from the project folder:

```python
from evac import (
    load_scenario, MILP, RunCase, run,
)
from evac.reporting import from_result

scenario = load_scenario("data/scenarios/mariposa.yaml")
case = RunCase(
    scenario,
    MILP(time_limit_seconds=180, threads=1, seed=7),
    objective="mean", mcs_count=1, initial_soc=0.20,
)
result = run(case, output_dir="results/my_run")
views = from_result(result)
print(views.table("summary").to_dataframe())
print(views.table("mcs_deployments").to_dataframe())
print(views.table("schedule").to_dataframe().head())
views.save_csv("results/my_run/tables")
```

## Workflow

Import workflow functions from `evac`, tables from `evac.reporting`, and plots from `evac.visualization`.

| Call | Returns |
| --- | --- |
| `load_scenario(path)` | Scenario with relative references to its inputs |
| `prepare(scenario, objective="mean", mcs_count=1, initial_soc=0.20)` | Validated vehicles, paths, chargers, departure windows, and objective |
| `probe_gurobi(runtime=True)` | Gurobi installation and license startup status |
| `inspect_run_case(RunCase(...))` | Vehicle/path counts and MILP dimensions without solving |
| `plan(prepared, MILP(...))` | Solver status, objective, bound, gap, model details, and optional plan |
| `inspect_plan(prepared, planner.plan)` | Structural and physical plan checks; `.valid` indicates success |
| `simulate(prepared, planner.plan)` | Simulation events for the returned plan |
| `evaluate(prepared, simulation)` | Simulation feasibility and metrics, including both mean and max |
| `run(RunCase(...), output_dir=path)` | Complete workflow, stage timings, and saved input/result files |
| `from_prepared(prepared)` | Map and candidate-path tables |
| `from_result(result)` | Summary, deployment, schedule, event, and completion tables |

## Settings

`objective` accepts `"mean"` or `"max"` and changes the MILP objective before solving. `mcs_count` selects units from the supplied inventory. Omit `initial_soc` to retain each demand cohort's SOC; supply a value to override all cohorts before preparation. Each MCS remains at its assigned site throughout the evacuation. Invalid SOC bounds, references, or malformed inputs raise `ValidationError`.

`MILP` accepts `time_limit_seconds`, `threads`, `seed`, `mip_gap`, and `mip_gap_absolute`; omit them to use Gurobi defaults. Absolute gap applies to the raw formulation objective, which is a sum for mean. Reported mean objective and bound are divided by vehicle count. Use those named arguments for Gurobi's `TimeLimit`, `Threads`, `Seed`, `MIPGap`, and `MIPGapAbs`. `time_limit_seconds` is one budget across the objective tiers. Pass additional native options through `MILP(parameters={"MIPFocus": 1})`; Gurobi validates their names and values. See the [Gurobi parameter interface](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#setParam).

## Results and files

Check `planner.plan is not None` before simulation. Without a plan, `result.simulation` and `result.evaluation` are `None`. See the [result definitions](../README.md#read-the-results) for statuses, feasibility, time origins, and units.

`ReportingViews.table(name).to_dataframe()` returns a pandas table; `.save_csv(directory)` saves all tables. The schedule has one row per vehicle. `MCSDeployment` records a unit and its assigned site for the whole evacuation. `RunResult.plan` reads the plan from `RunResult.planner`; there is one stored plan.

Use these pairs to save and reload objects:

| Object | Save / load functions |
| --- | --- |
| Full result, including accepted solver values, plan, simulation, metrics, and timings | `save_result` / `load_result` |
| Prepared inputs needed to simulate the matching plan | `save_prepared_scenario` / `load_prepared_scenario` |
| Plan alone | `save_plan` / `load_plan` |

Each function takes the object and path when saving, or the path when loading. Keep the matching prepared inputs with a saved plan. Reusing a filename replaces that file; the notebook creates a new output folder for each run.

## Reload a saved run

Replace the example path with the baseline folder printed by the notebook:

```python
from pathlib import Path
from evac import load_prepared_scenario, load_result, simulate, evaluate
from evac.reporting import from_result

saved = Path("results/mariposa_YOUR_RUN/baseline")  # Use your saved baseline folder.
prepared = load_prepared_scenario(saved / "prepared.yaml")
result = load_result(saved / "result.yaml")
print(from_result(result).table("summary").to_dataframe())

# A saved run can lack a plan if optimization ended without an incumbent.
if result.plan is not None:
    simulation = simulate(prepared, result.plan)
    evaluation = evaluate(prepared, simulation)
    print(evaluation.metrics)
```

Loading reads existing files without rerunning optimization. The last block runs a new simulation of the saved plan.

## Model assumptions

The MILP requires common per-port charging power and charging efficiencies across eligible providers. It conservatively groups vehicles and uses a finite candidate-path library. The supplied Mariposa case has homogeneous vehicles; regression tests also cover distinct SOC cohorts. Traffic inputs accept a positive constant multiplier or a piecewise linear profile starting at second zero. Travel integrates the reciprocal multiplier over the trip; after the final profile point, the last multiplier remains in effect. Preparation, MILP timing, and simulation use this same rule. Simulation uses the case's event-time tolerance in seconds. MCS energy accounts for charging and discharge losses.

## Show the planning geography

`from_prepared(prepared).table("map_nodes")` includes fixed charging ports, eligible MCS capacity, and the number of vehicles starting or finishing at each node. `map_svg` uses these fields for distinct origin, destination, charger, and candidate-site markers.

```python
from IPython.display import SVG, display
from evac.reporting import from_prepared
from evac.visualization import map_svg

map_views = from_prepared(prepared)
display(SVG(map_svg(
    map_views.table("map_nodes"),
    map_views.table("map_edges"),
    title="Evacuation network",
)))
```

Optional `origin_label` and `destination_label` arguments describe the scenario's assumed roles. They are display labels and do not introduce hazard boundaries or constraints. The opening illustration in the Mariposa notebook depicts the supplied case; the post-solve map is generated from the current prepared inputs.
