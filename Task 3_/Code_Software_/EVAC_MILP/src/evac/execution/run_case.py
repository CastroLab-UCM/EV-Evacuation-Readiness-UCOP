"""Inputs for one MCS planning and simulation run."""

from dataclasses import dataclass
from evac.scenario import Scenario
from evac.planning.selections import MILP


@dataclass(frozen=True, slots=True)
class RunCase:
    scenario: Scenario
    planner: MILP
    objective: str | None = None

    mcs_count: int | None = None
    initial_soc: float | None = None
