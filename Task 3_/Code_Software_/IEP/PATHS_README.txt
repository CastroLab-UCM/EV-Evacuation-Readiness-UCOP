Portable paths: All bundled input files retain their original names.

The following input files are read from IEP/mariposa_small/:
  mariposa_small.csv
  mariposa_small_od_demand.csv
  mariposa_small_xy.csv
  mariposa_small.nodes.xml
  mariposa_small.edges.xml

configs.py now resolves them relative to its own location, regardless of where VS Code is launched.
Outputs remain under IEP/results/<timestamp>/ with the original generated filenames.
main.py now copies configs.py and constants.py to the results directory without overwriting one another.
The absent optional viewSettings.xml is no longer passed to SUMO.
netconvert still needs to be available from SUMO on your PATH and Python dependencies must be installed.
No optimization or simulation model logic was intentionally modified.
