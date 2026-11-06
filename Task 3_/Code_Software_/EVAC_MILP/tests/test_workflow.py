"""Public planning behavior, persistence, and charging boundaries."""

from pathlib import Path
from dataclasses import replace
import pytest
from evac import (
    load_scenario,
    prepare,
    MILP,
    plan,
    RunCase,
    run,
    load_result,
    load_prepared_scenario,
    inspect_plan,
    ValidationError,
)
from evac.reporting import from_result, from_prepared
from evac.visualization import map_svg

FIXTURE_DATA = Path(__file__).resolve().parent / "fixtures" / "data"


@pytest.fixture
def scenario():
    return load_scenario(FIXTURE_DATA / "scenarios/trellis.yaml")


@pytest.mark.parametrize("objective", ["mean", "max"])
@pytest.mark.parametrize("mcs_count", [0, 1])
def test_charge_solve_replay_and_persist(scenario, tmp_path, objective, mcs_count):
    result = run(
        RunCase(
            scenario,
            MILP(threads=1),
            objective,
            mcs_count=mcs_count,
            initial_soc=0.105,
        ),
        output_dir=tmp_path,
    )
    assert result.plan is not None
    assert result.evaluation.feasible
    assert result.evaluation.metrics["finished_vehicle_count"] == 2
    assert result.evaluation.metrics["charging_session_count"] == 2
    assert result.evaluation.metrics["mean_completion_seconds"] == pytest.approx(160)
    assert result.evaluation.metrics["max_completion_seconds"] == pytest.approx(160)
    assert result.planner.formulation_objective == pytest.approx(160)
    prepared = load_prepared_scenario(tmp_path / "prepared.yaml")
    restored = load_result(tmp_path / "result.yaml")
    assert restored.plan == result.plan
    assert restored.evaluation.metrics == result.evaluation.metrics
    assert restored.planner.metadata["raw_variable_values"] == list(
        result.planner.metadata["raw_variable_values"]
    )
    assert inspect_plan(prepared, result.plan).valid
    assert len(result.plan.mcs_deployments) == mcs_count
    assert from_result(restored).table("schedule").to_dataframe().shape[0] == 2


def test_objective_is_applied_before_optimization(scenario):
    mean = prepare(scenario, objective="mean")
    maximum = prepare(scenario, objective="max")
    assert mean.evaluation.objectives[0].norm == "mean"
    assert maximum.evaluation.objectives[0].norm == "max"
    assert mean.identity != maximum.identity
    with pytest.raises(ValidationError):
        prepare(scenario, objective="median")


def test_settings_checked_at_public_boundary(scenario):
    with pytest.raises(ValidationError):
        prepare(scenario, mcs_count=2)
    with pytest.raises(ValidationError):
        prepare(scenario, initial_soc=1.1)


def test_no_incumbent_is_presentable(scenario, tmp_path):
    result = run(
        RunCase(scenario, MILP(time_limit_seconds=1e-12)),
        output_dir=tmp_path,
    )
    assert result.plan is None
    assert result.simulation is None
    assert result.evaluation is None
    assert result.planner.terminal_cause == "time_limit"
    assert from_result(result).table("schedule").rows == ()
    assert load_result(tmp_path / "result.yaml").plan is None


def test_mcs_deployments_are_complete_and_unique(scenario):
    from evac import simulate
    from evac.domain import DeploymentId

    prepared = prepare(scenario)
    result = plan(prepared, MILP(threads=1))
    deployment = result.plan.mcs_deployments[0]
    duplicate = replace(deployment, id=DeploymentId("duplicate"))
    for deployments in [(), (deployment, duplicate)]:
        invalid = replace(result.plan, mcs_deployments=deployments)
        assert not inspect_plan(prepared, invalid).valid
        with pytest.raises(ValidationError, match="exactly one deployment"):
            simulate(prepared, invalid)


def test_map_uses_bounded_display_coordinates(scenario):
    views = from_prepared(prepare(scenario))
    svg = map_svg(views.table("map_nodes"), views.table("map_edges"))
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)
    for circle in root.findall(".//{http://www.w3.org/2000/svg}circle"):
        assert 0 <= float(circle.attrib["cx"]) <= float(root.attrib["width"])
        assert 0 <= float(circle.attrib["cy"]) <= float(root.attrib["height"])


@pytest.fixture
def editable_case(tmp_path):
    import shutil
    import yaml

    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    return data, yaml


def test_fixed_port_power_and_simultaneous_sessions(editable_case):
    data, yaml = editable_case
    overlay_path = data / "scenario_overlays/trellis.yaml"
    overlay = yaml.safe_load(overlay_path.read_text())
    overlay["departure_windows"] = {"count": 6, "size_seconds": 60}
    overlay["fixed_charger_site_overrides"] = {"2": {"ports": 0}, "3": {"ports": 0}}
    overlay_path.write_text(yaml.safe_dump(overlay))
    demand = data / "demands/trellis.csv"
    demand.write_text(demand.read_text().replace(",1,4,2", ",1,4,6"))
    scenario = load_scenario(data / "scenarios/trellis.yaml")
    result = run(
        RunCase(
            scenario,
            MILP(threads=1),
            objective="mean",
            mcs_count=1,
            initial_soc=0.105,
        )
    )
    assert result.evaluation.feasible
    metrics = result.evaluation.metrics
    assert metrics["finished_vehicle_count"] == 6
    assert metrics["mcs_charging_session_count"] == 6
    assert metrics["mcs_delivered_energy_kwh"] == pytest.approx(4.5)
    assert metrics["mean_completion_seconds"] == pytest.approx(280)
    assert metrics["max_completion_seconds"] == pytest.approx(400)
    assert metrics["mcs_total_charging_duration_seconds"] == pytest.approx(360)


def test_infeasible_case_returns_no_plan(editable_case):
    data, yaml = editable_case
    demand = data / "demands/trellis.csv"
    demand.write_text(demand.read_text().replace(",1,4,2", ",1,4,100"))
    scenario = load_scenario(data / "scenarios/trellis.yaml")
    result = run(
        RunCase(
            scenario,
            MILP(threads=1),
            objective="mean",
            mcs_count=0,
            initial_soc=0.105,
        )
    )
    assert result.planner.termination.value == "infeasible"
    assert result.plan is None and result.simulation is None
    assert from_result(result).table("schedule").rows == ()


def test_distinct_vehicle_soc_is_retained(editable_case):
    data, yaml = editable_case
    demand = data / "demands/trellis.csv"
    demand.write_text(
        demand.read_text().splitlines()[0] + "\n"
        "low,example_ev,20.0,0.105,1,4,1\n"
        "high,example_ev,20.0,0.5,1,4,1\n"
    )
    scenario = load_scenario(data / "scenarios/trellis.yaml")
    prepared = prepare(scenario, objective="max")
    assert {vehicle.initial_soc for vehicle in prepared.vehicles} == {0.105, 0.5}
    result = run(RunCase(scenario, MILP(threads=1), objective="max"))
    assert result.evaluation.feasible
    assert result.evaluation.metrics["finished_vehicle_count"] == 2
    assert (
        result.evaluation.metrics["max_completion_seconds"]
        >= result.evaluation.metrics["mean_completion_seconds"]
    )


def test_map_reports_demand_and_effective_charging_capacity(editable_case):
    data, yaml = editable_case
    overlay_path = data / "scenario_overlays/trellis.yaml"
    overlay = yaml.safe_load(overlay_path.read_text())
    overlay["fixed_charger_site_overrides"] = {"2": {"ports": 0}, "3": {"ports": 0}}
    overlay_path.write_text(yaml.safe_dump(overlay))
    prepared = prepare(load_scenario(data / "scenarios/trellis.yaml"))
    views = from_prepared(prepared)
    nodes = {r["node_id"]: r for r in views.table("map_nodes").records()}
    assert nodes["1"]["origin_vehicle_count"] == 2
    assert nodes["4"]["destination_vehicle_count"] == 2
    assert all(r["fixed_charger_ports"] == 0 for r in nodes.values())
    svg = map_svg(views.table("map_nodes"), views.table("map_edges"),
                  origin_label="Threatened origin", destination_label="Safe destination")
    assert "Threatened origin: 2 EVs" in svg
    assert "Safe destination: 2 EVs" in svg
    assert "Fixed charging station" not in svg
    assert "Eligible MCS site" in svg


def test_map_keeps_minimal_tables_and_geometry_usable():
    from evac.reporting import ReportColumn, ReportTable
    import xml.etree.ElementTree as ET

    nodes = ReportTable("nodes", tuple(ReportColumn(c) for c in ["node_id", "x", "y"]),
                        (("a", 0, 0), ("b", 10, 0), ("c", 0, 20)))
    edges = ReportTable("edges", tuple(ReportColumn(c) for c in
                        ["edge_id", "source_node_id", "target_node_id"]),
                        (("ab", "a", "b"), ("ac", "a", "c")))
    svg = map_svg(nodes, edges, title="A < B & C")
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    points = [g.find("s:circle", ns) for g in root.findall("s:g", ns)]
    a, b, c = [tuple(float(p.attrib[k]) for k in ["cx", "cy"]) for p in points]
    assert (b[0] - a[0]) / 10 == pytest.approx((a[1] - c[1]) / 20)
    assert root.find("s:title", ns).text == "A < B & C"
    assert "Fixed charging station" not in svg
    assert "Threatened" not in svg


@pytest.mark.parametrize("objective", ["mean", "max"])
@pytest.mark.parametrize("multipliers", [(2.0,), (1.0, 2.0), (2.0, 1.0)])
def test_traffic_profile_solve_replay_and_reload(editable_case, objective, multipliers):
    from evac import simulate, evaluate

    data, yaml = editable_case
    profile = (
        {"kind": "constant", "multiplier": multipliers[0]}
        if len(multipliers) == 1
        else {
            "kind": "piecewise_linear",
            "points": [
                {"elapsed_seconds": 0, "multiplier": multipliers[0]},
                {"elapsed_seconds": 60, "multiplier": multipliers[1]},
            ],
        }
    )
    (data / "traffic/trellis.yaml").write_text(yaml.safe_dump({
        "schema": "evac/traffic/v1", "elapsed_exogenous_profile": profile,
    }))
    scenario = load_scenario(data / "scenarios/trellis.yaml")
    prepared = prepare(scenario)
    # After the final profile point, the last multiplier stays in effect.
    assert prepared.traffic.exit_time(
        entry_time_seconds=180, free_flow_duration_seconds=100,
    ) == pytest.approx(180 + 100 * multipliers[-1])
    destination = data.parent / "run"
    result = run(
        RunCase(scenario, MILP(threads=1), objective,
                mcs_count=1, initial_soc=0.105),
        output_dir=destination,
    )
    assert result.plan is not None and result.evaluation.feasible
    assert result.evaluation.metrics["finished_vehicle_count"] == 2
    assert result.evaluation.metrics["charging_session_count"] == 2
    assert result.planner.formulation_objective == pytest.approx(
        result.evaluation.metrics[f"{objective}_completion_seconds"], abs=1e-6,
    )
    restored = load_result(destination / "result.yaml")
    inputs = load_prepared_scenario(destination / "prepared.yaml")
    replay = evaluate(inputs, simulate(inputs, restored.plan))
    assert replay.feasible
    assert replay.metrics == result.evaluation.metrics


def test_scenario_save_load_roundtrip(scenario, tmp_path):
    from evac import save_scenario

    destination = tmp_path / "scenario.yaml"
    save_scenario(scenario, destination)
    restored = load_scenario(destination)
    assert restored == scenario
    assert prepare(restored).identity == prepare(scenario).identity


def test_result_has_one_plan_through_save_and_load(scenario, tmp_path):
    from evac import RunResult, save_result

    planner = plan(prepare(scenario), MILP(threads=1))
    result = RunResult(planner, None, None)
    assert result.plan is planner.plan
    assert dict(from_result(result).table("summary").rows)["has_plan"] is True
    save_result(result, tmp_path / "result.yaml")
    restored = load_result(tmp_path / "result.yaml")
    assert restored.plan is restored.planner.plan
    assert restored.plan == result.plan
    assert dict(from_result(restored).table("summary").rows)["has_plan"] is True


def test_native_parameters_and_objective_tiers(scenario):
    parameters = {"MIPFocus": 1, "Presolve": 0}
    planner = MILP(threads=1, seed=7, parameters=parameters)
    parameters["MIPFocus"] = 2
    assert planner.parameters["MIPFocus"] == 1
    result = run(RunCase(scenario, planner, initial_soc=0.105))
    assert result.plan is not None and result.evaluation.feasible
    assert result.planner.metadata["backend"] == "gurobi"
    assert result.planner.metadata["provider_version"]
    tiers = result.planner.metadata["solver_telemetry"]["tiers"]
    assert len(tiers) == 2
    assert tiers[1]["warm_start_columns"] > 0
    assert result.planner.metadata["raw_variable_values"] is not None


@pytest.mark.parametrize("parameters", [{"MIPFocus": 99}, {"InvalidParameter": 1}])
def test_gurobi_rejects_invalid_native_parameters(scenario, parameters):
    import gurobipy as gp

    with pytest.raises(gp.GurobiError, match=next(iter(parameters))):
        plan(prepare(scenario), MILP(parameters=parameters))


def test_common_controls_have_one_configuration_entry(scenario):
    with pytest.raises(ValidationError, match="MILP.time_limit_seconds"):
        plan(prepare(scenario), MILP(parameters={"time_limit": 10}))


def test_inspection_does_not_start_gurobi(scenario, monkeypatch):
    import builtins
    from evac import inspect_run_case, probe_gurobi

    native_import = builtins.__import__

    def without_solver(name, *args, **kwargs):
        if name == "gurobipy" or name.startswith("gurobipy."):
            raise AssertionError("Read-only inspection started the solver runtime")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_solver)
    assert probe_gurobi().dependency_installed
    inspection = inspect_run_case(RunCase(scenario, MILP()))
    assert inspection.vehicle_count == 2
    assert inspection.variables > 0 and inspection.objective_tiers == 2
