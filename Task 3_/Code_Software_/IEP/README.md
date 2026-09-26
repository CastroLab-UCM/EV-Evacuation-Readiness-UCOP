# EV Evacuation & Charging Infrastructure Optimization

An optimization framework for electric vehicle (EV) evacuation planning and multi-layer charging station allocation. This repository models multi-commodity traffic flow, battery consumption constraints, and charging infrastructure deployment using **Pyomo**, **Gurobi**, and **iGraph**, with support for microscopic traffic simulation via **SUMO** (Simulation of Urban MObility).

---

## Technical Overview

The framework provides multi-tiered layers to balance computational complexity with modeling detail:

* **Layer 0 (Baseline / UEP):** Heuristic baseline routing paths without optimized charging placement.
* **Layer 1:** Feasibility check for single Origin-Destination (OD) pairs under vehicle battery constraints.
* **Layer 2:** Strategic optimization layer determining global placement of mobile/fixed charging stations.
* **Layer 3 (IEP - Iterative Evacuation Plan):** Sequential OD pair routing with capacity constraints, queueing delays (BPR + M/M/c queue approximations), and pre-assigned station tracking.
* **Layer 4 (JEP - Joint Evacuation Plan):** Integrated optimization combining route choice and charging deployment decisions simultaneously.
---

## Directory Structure

```text
├── configs.py          # Platform paths, layer selection, and simulation parameters
├── constants.py        # Physical constants (vehicle range, battery spec, hazard locations)
├── rules.py            # Pyomo objective functions and mathematical constraints
├── layer_lib.py        # Core optimization solver routines and data processing
├── sumo_lib.py         # Network XML generation and SUMO co-simulation runner
├── main.py             # Primary entry point for multi-layer optimization runs
├── main_baseline.py    # Dedicated runner for baseline heuristic executions
├── layer_lib_test.py   # Unit tests for route sorting and data structures
└── sumo_lib_test.py    # Unit tests for SUMO network generation logic
└── mariposa_small.csv  # Map
```

---

## Requirements & Dependencies

### Core Requirements
* **Python 3.10+**
* **Gurobi Optimizer** (with valid license) https://www.gurobi.com/downloads/
* **SUMO** (optional, required only for micro-simulation runs) https://sumo.dlr.de/docs/Installing/index.html

### Python Libraries
Install required dependencies via `pip`:

```bash
pip install pyomo igraph numpy pandas openpyxl pynverse matplotlib
```

---

## Configuration (`configs.py`)

Key execution modes and parameters can be adjusted inside `configs.py`:

```python
# Mode Selection: 'BASELINE', 'OPT' (Iterative), or 'SINGLE' (Joint Layer 4)
LAYER = 'OPT'

# Case Study / Network Target
CASE_NAME = 'mariposa_small'

# Output Parameters
SAVE_RESULTS = True
MAX_CHARGER_NUMBER = 1
CHARGER_PROT_NUM = 5  # Number of chargers per station
```

---

## Quick Start

### 1. Running the Main Optimization Pipeline
To run the active layer mode configured in `configs.py`:

```bash
python main.py
```

### 2. Running Baseline Comparison Mode
To execute the fast heuristic baseline directly:

```bash
python main_baseline.py
```

### 3. Running Unit Tests
To verify graph processing and routing logic:

```bash
python -m unittest discover -p "*_test.py"
```

---

## Input Data Requirements

Each case study folder under `CASE_NAME` requires the following CSV and XML definitions:

1. `<CASE_NAME>.csv` – Graph edge definitions, lengths, and road capacities.
2. `<CASE_NAME>_od_demand.csv` – Origin-Destination matrices and demand sizes.
3. `<CASE_NAME>_xy.csv` – Node spatial coordinates.
4. `<CASE_NAME>.nodes.xml` & `<CASE_NAME>.edges.xml` – SUMO map definitions.

---

## Output Artifacts

Execution outputs are saved to the configured base path (`BASE_DIR`):

* **Excel Reports (`*.xlsx`):** Aggregated metrics for travel times, BPR delays, station wait times, flow allocations, and OD route logs.
* **Solver Logs (`*.log`):** Detailed MIP optimization logs from Gurobi.
* **SUMO Files (`*.trips.xml`, `*.add.xml`, `*.net.xml`):** Generated simulation networks for visualization in SUMO-GUI.
