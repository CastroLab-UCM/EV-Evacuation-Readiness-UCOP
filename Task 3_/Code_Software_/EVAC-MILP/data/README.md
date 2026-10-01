# Input data

The authors provide all inputs under the project MIT license. Network geometry uses local Cartesian coordinates in metres. Loading a scenario uses local files and does not download data.

| Folder | Contents |
| --- | --- |
| `maps` | Network declarations and GeoJSON nodes and roads |
| `assets/vehicles` | Battery sizes and distance per kWh |
| `assets/mobile_chargers` | MCS battery, port power, and port count |
| `demands` | Vehicle cohorts: type, speed, SOC, origin, destination, and count |
| `supplies` | Available MCS units and initial SOC |
| `traffic` | Exogenous travel-time profiles; the supplied profile uses a constant multiplier of 1 |
| `evaluations` | Default completion-time objective |
| `scenario_overlays` | Charging physics, SOC bounds, path limits, and departure windows |
| `scenarios` | Relative references joining each case's inputs |

## Mariposa

`mariposa.yaml` defines the network and inputs below. The authors' geometry is retained, with node identifiers numbered 1–12 to match the case-study notation.

| Input | Supplied value |
| --- | --- |
| Network | 12 nodes and 34 directed roads |
| Demand | 300 EVs from node 4 to node 1; 300 from node 4 to node 10 |
| EV battery and initial SOC | 30 kWh; 0.20 |
| EV speed and efficiency | 72.4 km/h; 5,000 m/kWh (20 kWh/100 km) |
| Fixed charging ports | 20 at node 4, 20 at node 6, 40 at node 9 |
| MCS inventory | 15 units; the notebook baseline uses 1 |
| Each MCS | 420 kWh; five independent 40 kW ports, totaling 200 kW |
| Charging and MCS discharge efficiency | 0.9 each |
| Charging increment | 300 s, delivering 3 kWh to an EV |
| Departure windows | 48 windows of 150 s; last departure at 7,050 s |
| Path limits per demand group | Up to 100 geographical paths and 200 charging paths; at most two charging stops |

Completion can occur after the last departure window. MCS sites and capacities are declared in the map; inspect them with `from_prepared(prepared).table("map_nodes")`. Each MCS stays at its assigned site for the entire evacuation. Port power stays fixed even when other ports are idle. Charging queues are forbidden in this case.

The [root README](../README.md#mariposa-inputs-and-interpretation) explains the example's research relationship and validation limits.

## Use different inputs

Copy the relevant data files within this project, update their scenario references, and load the new scenario. Keep the supplied case for comparison. Follow each file's units: SOC is a fraction, and vehicle count is a cohort size rather than a flow rate.

### Traffic profiles

Traffic files use `schema: evac/traffic/v1`. Set `elapsed_exogenous_profile` to
`{kind: constant, multiplier: 1.0}` or a piecewise linear profile such as:

```yaml
elapsed_exogenous_profile:
  kind: piecewise_linear
  points:
    - {elapsed_seconds: 0, multiplier: 1.0}
    - {elapsed_seconds: 900, multiplier: 1.5}
```

Times are seconds from the evacuation start and must increase strictly. All
multipliers must be positive. A multiplier of 1 is free-flow travel; larger
values slow travel. Travel progress is integrated across profile changes, and
the final multiplier continues after the last point. The profile is an input
assumption and does not depend on the optimized vehicle flows.
