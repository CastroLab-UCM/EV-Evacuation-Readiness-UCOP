# EV evacuation planning with mobile charging stations

Plan EV departures, charging, and mobile charging station (MCS) locations on a road network. Each MCS stays at its assigned site for the whole evacuation. The package builds a mixed-integer linear program (MILP), solves it, and simulates the returned plan. You can minimize **mean** or **maximum** completion time and inspect both measures for the same plan.

Follow the [Mariposa notebook](example/mariposa_planning.ipynb) from setup through a saved evacuation plan. The steps below cover installation and use with limited Python experience.

## Start here

Use Python **3.11**, the tested version, and a terminal with internet access for installation. Keep the complete project folder together; all example data is included.

### 1. Create an environment

Open a terminal in this folder. On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

On Windows, use Command Prompt in the project folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

The `-e` installation makes the local `evac` Python package available to the notebook. Keep this folder in place while using it. These commands install Gurobi's Python interface and solver. License activation tools such as `grbgetkey` are a separate download; they are not included in the pip package. [Gurobi Python installation guide](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python).

### 2. Choose a Gurobi license

The supplied Mariposa case requires an unrestricted Gurobi license. Installing the solver alone does not provide one.

| Your situation | Path |
| --- | --- |
| Student, faculty, or staff at an eligible degree-granting institution, doing coursework, teaching, or academic research | Request a free academic license. Academic Named-User and Academic WLS licenses support models of this size. |
| Non-production use with smaller custom models | The pip installation includes a free size-limited license. The supplied Mariposa case exceeds its limits. |
| Non-academic user who needs the larger case | Use an existing suitable license or investigate Gurobi's evaluation license. Check the terms applicable to your use. |

**Academic Named-User:** create a Gurobi account with your university affiliation, request the license in the [academic portal](https://www.gurobi.com/academics), and follow its activation instructions. Download Gurobi's standalone license tools or full installer, then run the exact `grbgetkey` command shown for your license. Initial validation requires a recognized university network. This license is tied to one user and computer; after activation it can work offline. [Academic eligibility and license choices](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license).

**Academic WLS:** request the academic WLS license, then use the Web License Manager to create an API key and download `gurobi.lic`. WLS requires internet access during use. Follow the portal's academic-network validation and renewal instructions. Keep the license file outside this project and never put its contents in a notebook. [WLS setup](https://support.gurobi.com/hc/en-us/articles/13232844297489-How-do-I-set-up-a-Web-License-Service-WLS-license), [academic WLS restrictions](https://support.gurobi.com/hc/en-us/articles/34672988479633-What-are-the-restrictions-on-using-an-academic-WLS-license).

Gurobi can find `gurobi.lic` in its documented default location, including your home directory. For another location, set `GRB_LICENSE_FILE` to the **file's full path** before starting Jupyter. For example, replace the placeholder below with your actual path:

```bash
export GRB_LICENSE_FILE="/absolute/path/to/gurobi.lic"
```

In Windows Command Prompt:

```bat
set "GRB_LICENSE_FILE=C:\path\to\gurobi.lic"
```

Restart Jupyter after changing license configuration. [License options](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license), [evaluation and other licensing paths](https://www.gurobi.com/product/pricing-and-licensing).

The bundled restricted license permits **2,000 variables and 2,000 linear constraints** for non-production use. Models with quadratic terms have a **200-variable** limit. This package builds linear models. The actual supplied model counts are:

| Case and objective | Variables | Linear constraints after the second objective tier |
| --- | ---: | ---: |
| Mariposa baseline, mean, one MCS | 63,319 | 49,523 |
| Mariposa baseline, max, one MCS | 82,520 | 107,123 |

**The supplied Mariposa model exceeds those limits.** Execution checks used an unrestricted Gurobi license. Changing inputs can change model size. `inspect_run_case(case)` reports the compiled counts; the second objective tier adds one linear constraint. Licensing guidance was checked against official documentation on September 28, 2026. [Exact restricted-license limits](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error).

### 3. Register the kernel and open Jupyter

In the same activated environment:

```bash
python -m ipykernel install --sys-prefix --name evac --display-name "EVac"
python -m jupyter lab
```

Open [example/mariposa_planning.ipynb](example/mariposa_planning.ipynb) and select the **EVac** kernel. Run cells in order with Shift+Enter, or use **Kernel → Restart Kernel and Run All Cells**. The notebook checks Gurobi before preparing the inputs. The first simulation may take longer while numerical kernels compile.

Sections 1–6 produce and save a baseline plan. Section 7 is optional: set `RUN_COMPARISONS = True` there to solve three additional cases and plot their completion curves. The default run solves only the baseline. After changing baseline settings, restart the kernel and run the cells in order. Restarting clears Python variables but keeps saved files.

## What you can change

The Mariposa settings cell exposes these decisions:

| Setting | Meaning | Supplied baseline |
| --- | --- | --- |
| `MCS_COUNT` | Available units from the supplied inventory; integer 0–15 | 1 |
| `INITIAL_SOC` | Initial EV battery fraction, 0–1 | 0.20 |
| `OBJECTIVE` | `"mean"` or `"max"` completion time, selected **before** optimization | `"mean"` |
| `TIME_LIMIT_SECONDS` | Positive optimization time limit; `None` removes it | 180 s |
| `THREADS` | Positive solver worker count | 1 |
| `SEED` | Nonnegative solver seed | 7 |

When enabled, notebook section 7 compares three MCS units, SOC 0.25, and the max objective separately against the baseline. Edit `comparison_settings` for other comparisons. For a different evacuation, supply inputs and assumptions appropriate to that case.

## Python API

The [API guide](docs/python_api.md) provides a complete Python example and documents inputs, return values, and saving or loading results. `run` combines `prepare`, `plan`, `simulate`, and `evaluate`; the Mariposa notebook shows each stage separately.

## Read the results

All times below are in **seconds**, except plot axes explicitly labeled in minutes.

| Result | Interpretation |
| --- | --- |
| `formulation_objective_seconds` | Mean or max completion represented by the finite MILP. The mean is normalized by vehicle count. |
| `formulation_bound_seconds`, `relative_gap` | Solver certificate for the selected primary objective. A bound is not a simulated result. |
| `simulated_objective_seconds` | Selected objective evaluated from the returned plan's replay. |
| `mean_completion_seconds`, `max_completion_seconds` | Completion measured from the common evacuation start at time zero. Includes time waiting to depart. |
| `mean_evacuation_duration_seconds`, `max_evacuation_duration_seconds` | Completion minus each vehicle's actual departure time. |
| `preparation_seconds`, `build_seconds`, `solve_seconds` | Computer time for input preparation, model construction, and optimization. |
| `simulation_seconds`, `evaluation_seconds` | Computer time for replay and calculating its metrics. |

`optimal` means the solver proved optimality to its numerical tolerances for the finite model and its objective hierarchy. `feasible` means an incumbent was returned without that complete proof. `infeasible` certifies no solution for the selected finite inputs. `no_incumbent` means no plan was returned; a short time limit is one possible cause. It does not prove infeasibility. Check `terminal_cause`, `has_plan`, and `proven_optimal` with the objective.

Check simulation feasibility separately from solver status. Without a plan or complete simulation, completion metrics are unavailable; missing values are not zero. Integer decisions are extracted within the declared numerical tolerance and validated before simulation. Saved results also retain the exact accepted solver values, variable blocks, and objective certificates.

The primary objective is completion time. A lower-priority objective reduces MCS service energy among schedules with the same primary value within the declared numerical tolerance. Different schedules or deployments can have the same objective.

The notebook creates a new timestamped folder under `results/`. Its `baseline/` folder contains `prepared.yaml`, `result.yaml`, and CSV tables for the complete schedule, deployment, timelines, and metrics. Enabling comparisons adds one folder per variant, `comparison.csv`, and `completion_curves.png`. See the [reload example](docs/python_api.md#reload-a-saved-run) to reopen a saved plan. The notebook is distributed without execution output.

## Mariposa inputs and interpretation

The Mariposa example has 600 EVs on a network of 12 nodes and 34 directed roads. See [the input guide](data/README.md#mariposa) for charging infrastructure, vehicle parameters, departure windows, and path limits.

The physical parameters follow the homogeneous case in the TTE preprint listed below. The departure horizon and finite path limits specify this packaged example. These examples **do not claim to reproduce the paper's full numerical study**, its heterogeneous cases, or its parameter sweeps. Finite routes, discrete departure and charging times, and conservative grouped constraints can restrict the schedules considered. Optimality applies to that finite model.

In the verified runs, three MCS units reduced simulated mean completion from **5,409.6 s** to **4,969.6 s**. Raising initial SOC to 0.25 reduced it to **5,214.6 s**. Minimizing the maximum reduced the last completion from **9,140.5 s** to **8,038.7 s**, while the returned plan's mean increased. These results illustrate resource and objective tradeoffs for this case. Other inputs, alternate optima, and solver settings can produce different results.

The checks used Python 3.11 on macOS and Gurobi 13.0.3. The four Mariposa cases were exercised with Gurobi, including plan simulation, saving, and loading. Gurobi regression tests cover both objectives with synthetic test fixtures. Windows/Linux installation instructions use standard environment commands but were not executed on those platforms. Solve times depend on hardware, licenses, solver versions, and settings; 180 seconds does not guarantee an incumbent or optimality on every computer.

## Troubleshooting

- **`No module named evac`:** activate the environment, repeat the editable installation from the project root, and select the EVac kernel. Installing into one environment does not install into every Jupyter kernel.
- **EVac kernel is missing:** run the kernel registration and Jupyter commands from the same environment. `--sys-prefix` installs the kernel in that environment.
- **`Model too large for size-limited Gurobi license`:** Mariposa needs a suitable unrestricted license. Confirm Gurobi can find that license file, then restart Jupyter. Passing the startup check does not prove that the license supports this model size.
- **License or WLS connection error:** follow the official license instructions above; check network access and the file location. Do not paste license credentials into the notebook.
- **No plan after the time limit:** record `no_incumbent` and the bound. You may raise the time limit and rerun. Do not substitute zero for a missing objective.
- **Infeasible inputs:** inspect SOC, available charging sites, energy, charging-path limits, and departure windows. A model with a restricted path library may be infeasible even if a different route library could work.
- **A cell seems slow:** preparation, model building, and first-use compilation add time outside the solve limit. Check the notebook's stage timings.
- **Data cannot be found:** start Jupyter from the project root and retain the provided relative folder layout.

## Files and development

```text
src/evac/       Public Python package, MILP, simulation, reporting
example/       Mariposa planning notebook
data/          Network, vehicle, charger, demand, and scenario inputs
docs/          Python API guide and optional participant worksheet
tests/         Regression tests and synthetic fixtures
```

The distribution is named `evac-planning`; the Python import is `evac`. The source archive includes the example, input data, documentation, and tests. The wheel installs the Python library; supply your own input files or use the data from the source archive. Inputs and code are provided by the authors under the [MIT license](LICENSE). See [data/README.md](data/README.md) for data organization. Solver products have their own licenses.

The optional [participant worksheet](docs/participant_worksheet.docx) contains student exercises based on the notebook.

To run the tests:

```bash
python -m pip install -e ".[test]"
python -m pytest tests
```

## References

These papers describe the research context. The two conference papers use earlier case and charging assumptions; their numerical settings should not be substituted for the supplied Mariposa inputs.

- Xuchang Tang, Simon Kuang, Shuang Feng, Joseph Moyalan, Ricardo de Castro, Qijian Gan, Scott Moura, and Xinfan Lin. *Coordinated Optimization of Electric Vehicle Evacuation with Mobile Charging Stations: A MILP-Based Scheduling Framework*. Author-provided IEEE Transactions on Transportation Electrification preprint. This is the main method and Mariposa parameter reference; no published DOI is asserted here.
- *Enhancing Large-Scale Evacuations of Electric Vehicles through Integration of Mobile Charging Stations*. IEEE ITSC, 2024, pp. 1494–1501. [DOI: 10.1109/ITSC58415.2024.10919547](https://doi.org/10.1109/ITSC58415.2024.10919547).
- *Optimization of Electric Vehicle Evacuation Integrating Mobile Charging Stations and Considering Vehicle Diversity*. American Control Conference, 2025, pp. 3028–3034. [DOI: 10.23919/ACC63710.2025.11107473](https://doi.org/10.23919/ACC63710.2025.11107473).
