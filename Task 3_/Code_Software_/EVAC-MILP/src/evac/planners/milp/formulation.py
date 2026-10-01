"""Build and materialize the MCS formulation."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from evac.planners.milp._source.optimization.formulation import MILPSolver
from evac.physics.units import CANONICAL_UNIT_SYSTEM
from evac.planners.milp.problem import MilpProblem


@dataclass(slots=True)
class FormulationContext:
    problem: MilpProblem
    _source: MILPSolver

    def materialize_source_solution(self, values: Sequence[float]):
        accepted = tuple(values)
        self.problem.validate_values(accepted)
        self._source.model.assign_solution(accepted)
        solution = self._source._extract(self._source.dvs)
        solution["unit_system"] = dict(CANONICAL_UNIT_SYSTEM)
        self._source.solution = solution
        return self._source.report(return_all=True)


def build_formulation(pack: Mapping[str, Any]) -> FormulationContext:
    source = MILPSolver(pack)
    return FormulationContext(source.build_problem(), source)
