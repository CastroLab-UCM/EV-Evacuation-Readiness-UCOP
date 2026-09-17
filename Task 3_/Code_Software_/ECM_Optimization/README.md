## Overview
This repository contains the source code associated with the paper “Equivalent Circuit Model–based Electric Vehicle Evacuation with Mobile Charging Stations.” The proposed framework uses an Equivalent Circuit Model (ECM) to jointly optimize evacuation routing, EV charging decisions, and the deployment of Mobile Charging Stations (MCSs). The model accounts for vehicle driving-range limitations, road-capacity constraints, Fixed Charging Station (FCS) availability, and MCS deployment and charging-capacity constraints.

The repository includes the optimization codes and supporting files used to generate the numerical results and case studies presented in the paper.

### 1. OS Compatibility
This guide describes an installation and development workflow for Windows operating systems (Windows 10 and 11). It has not been tested on macOS or Linux, although it should work with minor or no modifications if the required dependencies are installed correctly.

### 2. Project Architecture
```text
ECM_optimization/
├── Optimization_code.py                   # Main ECM-based evacuation optimization
├── MILP_solve_tmax_unidirectional.py      # Maximum evacuation-time objective
├── MILP_solve_t_avg_unidirectional.py     # Average evacuation-time objective
├── MILP_solve_t_avg_sd_unidirectional.py  # Average/deviation-based objective
├── Optimization_code_MCS_utility_plot.py  # MCS utilization analysis and plotting
├── Additional_Plots.py                    # Additional result visualization
├── plot_map_networkx_unidirectional.py     # Transportation-network visualization
├── Mariposa_analysis.xlsx                 # Analysis and simulation results
├── Mariposa_plots.xlsx                    # Data used for generating plots
│
├── GeoJson/                               # Mariposa transportation-network data
│   ├── centroids.geojson
│   ├── centroid_connections.geojson
│   ├── nodes.geojson
│   ├── sections.geojson
│   ├── turnings.geojson
│   └── LinkFlowStatistics_NormalFlow_v20250916a.txt
│
└── Uncoordinated evac planning/           # Baseline evacuation-planning approach
    ├── baseline_algorithm_MCS_plotting.py
    └── config.py
```

