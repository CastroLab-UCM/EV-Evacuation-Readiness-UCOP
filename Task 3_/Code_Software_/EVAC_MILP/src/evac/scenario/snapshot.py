"""Immutable scenario inputs and their identity."""

from __future__ import annotations
from dataclasses import dataclass
from evac.domain import SemanticIdentity
from evac.errors import ValidationError
from evac.scenario.authored import AuthoredScenarioResources
from evac.scenario.codecs import load_scenario_resources
from evac.scenario.identity import scenario_identity_from_resources
from evac.scenario.models import Scenario


@dataclass(frozen=True, slots=True)
class ResolvedScenarioSnapshot:
    scenario: Scenario
    resources: AuthoredScenarioResources
    identity: SemanticIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, Scenario):
            raise ValidationError("ResolvedScenarioSnapshot.scenario must be typed.")
        if not isinstance(self.resources, AuthoredScenarioResources):
            raise ValidationError("ResolvedScenarioSnapshot.resources must be typed.")
        if self.identity.scope != "scenario":
            raise ValidationError(
                "ResolvedScenarioSnapshot.identity must be a Scenario identity."
            )


def resolve_scenario_snapshot(scenario: Scenario) -> ResolvedScenarioSnapshot:
    """Load and validate every authored resource exactly once for this snapshot."""
    if not isinstance(scenario, Scenario):
        raise ValidationError("resolve_scenario_snapshot requires a typed Scenario.")
    resources = load_scenario_resources(scenario)
    return ResolvedScenarioSnapshot(
        scenario, resources, scenario_identity_from_resources(scenario, resources)
    )


__all__ = ["ResolvedScenarioSnapshot", "resolve_scenario_snapshot"]
