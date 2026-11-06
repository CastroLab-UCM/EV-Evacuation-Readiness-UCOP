# EV evacuation planning with mobile charging stations

Plan EV departures, charging, and mobile charging station (MCS) locations on a road network. Each MCS stays at its assigned site for the whole evacuation. The package builds a mixed-integer linear program (MILP), solves it, and simulates the returned plan. You can minimize **mean** or **maximum** completion time and inspect both measures for the same plan.

The [Mariposa notebook](example/mariposa_planning.ipynb) walks through one case from setup to a saved evacuation plan. The steps below assume little Python experience.

## Start here

You need Jupyter with Python 3.11 or newer, internet access for the first run, and a Gurobi license. Keep the project folder together; all example data is included.

### 1. Get Python and Jupyter

**Anaconda (simplest).** Install [Anaconda Distribution](https://www.anaconda.com/download) if you do not already have it. No other installation is needed: the notebook's first cell installs the remaining packages, including Gurobi's Python interface, into Anaconda's Python.

**Separate environment (optional).** To keep this project's packages apart from other work, open a terminal in this folder. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
```

On Windows, in Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e ".[notebook]"
```

### 2. Get a Gurobi license

The free license that comes with Gurobi's pip package is limited to 2,000 variables and constraints. The Mariposa model has about 63,000 variables, so it needs a full license. [Size-limited license details](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error).

- **Students, faculty, and staff at a degree-granting institution** can get a free academic license from the [academic portal](https://www.gurobi.com/academics). [Eligibility and license choices](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license).
  - **Named-User:** for one computer. Download Gurobi's license tools, then run the `grbgetkey` command shown on your license page while connected to your university network. It works offline afterwards.
  - **WLS:** a downloadable `gurobi.lic` file that needs internet access while solving. [WLS setup](https://support.gurobi.com/hc/en-us/articles/13232844297489-How-do-I-set-up-a-Web-License-Service-WLS-license).
- **Others** can use an existing Gurobi license or request an [evaluation license](https://www.gurobi.com/product/pricing-and-licensing).

Gurobi finds `gurobi.lic` in your home folder automatically. If you keep it elsewhere, set `GRB_LICENSE_FILE` to the file's full path before starting Jupyter:

```bash
export GRB_LICENSE_FILE="/absolute/path/to/gurobi.lic"
```

In Windows Command Prompt:

```bat
set "GRB_LICENSE_FILE=C:\path\to\gurobi.lic"
```

Restart Jupyter after changing the license.

### 3. Open and run the notebook

Start JupyterLab from Anaconda Navigator, or run `python -m jupyter lab` in your activated environment. Browse to this folder and open [example/mariposa_planning.ipynb](example/mariposa_planning.ipynb). Run cells in order with Shift+Enter, or use **Kernel → Restart Kernel and Run All Cells**. The first code cell installs any missing packages; the notebook then checks Gurobi before preparing the inputs. The first simulation may take longer while numerical kernels compile.

Sections 1–6 produce and save a baseline plan. Section 7 is optional: set `RUN_COMPARISONS = True` there to solve three more cases and plot their completion curves. After changing settings, restart the kernel and run the cells in order again.

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

When enabled, notebook section 7 compares three MCS units, SOC 0.25, and the max objective separately against the baseline. Edit `comparison_settings` for other comparisons. To model a different evacuation, see [Use different inputs](data/README.md#use-different-inputs).

## Python API

The notebook loads the package from this folder by itself. To `import evac` in your own scripts or notebooks, install it once by running `python -m pip install -e .` in this folder (from Anaconda Prompt on Windows, or your activated environment). The `-e` option links the installation to this folder, so keep the folder in place.

The [API guide](docs/python_api.md) provides a complete Python example and documents inputs, return values, and saving or loading results. `run` combines `prepare`, `plan`, `simulate`, and `evaluate`; the Mariposa notebook shows each stage separately.

## Read the results

All times are in seconds.

| Result | Meaning |
| --- | --- |
| `formulation_objective_seconds` | The objective (mean or max completion) as computed by the optimization model |
| `formulation_bound_seconds`, `relative_gap` | The solver's proven lower bound and the remaining gap to it |
| `simulated_objective_seconds` | The same objective, measured by replaying the plan in the simulation |
| `mean_completion_seconds`, `max_completion_seconds` | Time from the evacuation start until vehicles arrive, including waiting to depart |
| `mean_evacuation_duration_seconds`, `max_evacuation_duration_seconds` | Time from each vehicle's own departure until it arrives |
| `preparation_seconds`, `build_seconds`, `solve_seconds`, `simulation_seconds`, `evaluation_seconds` | Computer time for each stage |

The solver status is one of:

- `optimal`: the plan is proven best for the model.
- `feasible`: a plan was found, but not proven best before the time limit.
- `infeasible`: no plan satisfies the model's constraints.
- `no_incumbent`: no plan was found, usually because the time limit was too short.

The simulation replays the returned plan independently, so check `simulation_feasible` as well. When there is no plan, completion metrics are empty.

Among plans with the same completion time, the solver prefers the one that uses the least MCS energy.

The notebook creates a new timestamped folder under `results/`. Its `baseline/` folder contains `prepared.yaml`, `result.yaml`, and CSV tables for the complete schedule, deployment, timelines, and metrics. Enabling comparisons adds one folder per variant, `comparison.csv`, and `completion_curves.png`. See the [reload example](docs/python_api.md#reload-a-saved-run) to reopen a saved plan.

## Mariposa inputs and interpretation

The Mariposa example has 600 EVs on a network of 12 nodes and 34 directed roads. See [the input guide](data/README.md#mariposa) for charging infrastructure, vehicle parameters, departure windows, and path limits.

The physical parameters follow the homogeneous case in the journal paper listed under [References](#references); the departure horizon and path limits are specific to this example. The model uses a finite set of routes and discrete departure and charging times, so "optimal" means best within that model.

In our runs, three MCS units reduced simulated mean completion from **5,409.6 s** to **4,969.6 s**. Raising initial SOC to 0.25 reduced it to **5,214.6 s**. Minimizing the maximum reduced the last completion from **9,140.5 s** to **8,038.7 s**, while that plan's mean increased.

These numbers come from Gurobi 13.0.3 with Python 3.11 and 3.13 on macOS. Solve times vary by computer; if 180 seconds is not enough, increase `TIME_LIMIT_SECONDS`.

## Troubleshooting

- **Python 3.11 or newer required:** update Anaconda, or create the separate environment with a newer Python.
- **Package installation failed in the first cell:** check internet access and rerun the cell. On a computer where you cannot install packages, use the separate environment option.
- **`No module named evac` outside the notebook:** install the package as described in [Python API](#python-api).
- **`Model too large for size-limited Gurobi license`:** Gurobi is using the free license. Install a full license (step 2), then restart Jupyter.
- **License or WLS connection error:** follow the license instructions above; check network access and the file location.
- **No plan after the time limit:** increase `TIME_LIMIT_SECONDS` and rerun.
- **Infeasible:** check SOC, charging sites, path limits, and departure windows in your inputs.
- **A cell seems slow:** preparing inputs, building the model, and first-time compilation take time beyond the solve limit.
- **Data cannot be found:** open the notebook from its place inside this project folder and keep the provided folder layout.

## Files and development

```text
src/evac/       Public Python package, MILP, simulation, reporting
example/       Mariposa planning notebook
data/          Network, vehicle, charger, demand, and scenario inputs
docs/          Python API guide and optional participant worksheet
tests/         Regression tests and synthetic fixtures
```

Code and input data are released under the [MIT license](LICENSE); Gurobi is licensed separately. See [data/README.md](data/README.md) for the input file formats.

The optional [participant worksheet](docs/participant_worksheet.docx) contains student exercises based on the notebook.

To run the tests:

```bash
python -m pip install -e ".[test]"
python -m pytest tests
```

## References

If you use this code, please cite the journal paper; [CITATION.cff](CITATION.cff) provides the entry, also available from GitHub's "Cite this repository" button. The two conference papers are earlier work with different case assumptions.

- Xuchang Tang, Simon Kuang, Shuang Feng, Joseph Moyalan, Ricardo de Castro, Qijian Gan, Scott Moura, and Xinfan Lin. *Coordinated Optimization of Electric Vehicle Evacuation with Mobile Charging Stations: A MILP-Based Scheduling Framework*. IEEE Transactions on Transportation Electrification, preprint.
- *Enhancing Large-Scale Evacuations of Electric Vehicles through Integration of Mobile Charging Stations*. IEEE ITSC, 2024, pp. 1494–1501. [DOI: 10.1109/ITSC58415.2024.10919547](https://doi.org/10.1109/ITSC58415.2024.10919547).
- *Optimization of Electric Vehicle Evacuation Integrating Mobile Charging Stations and Considering Vehicle Diversity*. American Control Conference, 2025, pp. 3028–3034. [DOI: 10.23919/ACC63710.2025.11107473](https://doi.org/10.23919/ACC63710.2025.11107473).
