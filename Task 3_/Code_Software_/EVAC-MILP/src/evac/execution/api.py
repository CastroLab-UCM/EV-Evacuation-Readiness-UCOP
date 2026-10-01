"""Prepare, optimize, simulate, evaluate, and optionally save a run."""

from pathlib import Path
from time import perf_counter
from evac.domain import ArtifactLocations, RunResult
from evac.scenario.preparation import prepare
from evac.planning import plan
from evac.simulation import simulate
from evac.evaluation import evaluate
from .run_case import RunCase


def run(run_case: RunCase, *, output_dir: str | Path | None = None) -> RunResult:
    start = perf_counter()
    prepared = prepare(
        run_case.scenario,
        objective=run_case.objective,
        mcs_count=run_case.mcs_count,
        initial_soc=run_case.initial_soc,
    )
    preparation_seconds = perf_counter() - start
    planner = plan(prepared, run_case.planner)
    simulation = None
    evaluation = None
    simulation_seconds = 0.0
    evaluation_seconds = 0.0
    if planner.plan is not None:
        start = perf_counter()
        simulation = simulate(prepared, planner.plan)
        simulation_seconds = perf_counter() - start
        start = perf_counter()
        evaluation = evaluate(prepared, simulation)
        evaluation_seconds = perf_counter() - start
    destination = None if output_dir is None else Path(output_dir)
    paths = (
        {}
        if destination is None
        else {
            "result": destination / "result.yaml",
            "prepared": destination / "prepared.yaml",
        }
    )
    result = RunResult(
        planner,
        simulation,
        evaluation,
        ArtifactLocations(destination, paths),
        {
            "preparation_seconds": preparation_seconds,
            "build_seconds": planner.metadata["build_seconds"],
            "solve_seconds": planner.metadata["solve_seconds"],
            "simulation_seconds": simulation_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
    )
    if destination is not None:
        from evac.artifacts.documents import save_result, save_prepared_scenario

        save_prepared_scenario(prepared, paths["prepared"])
        save_result(result, paths["result"])
    return result
