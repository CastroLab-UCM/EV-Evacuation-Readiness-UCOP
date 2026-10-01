"""Resolve scenario input files by resource role."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from evac.errors import ValidationError
from evac.scenario.models import ResourceRef, Scenario

_ROLE_PATHS: dict[str, tuple[str, str]] = {
    "map": ("maps", "map.yaml"),
    "vehicles": ("assets/vehicles", ".yaml"),
    "mobile_chargers": ("assets/mobile_chargers", ".yaml"),
    "demand": ("demands", ".csv"),
    "supply": ("supplies", ".csv"),
    "traffic": ("traffic", ".yaml"),
    "evaluation": ("evaluations", ".yaml"),
    "overlay": ("scenario_overlays", ".yaml"),
}


@dataclass(frozen=True, slots=True)
class ResolvedResource:
    reference: ResourceRef
    path: Path


@dataclass(frozen=True, slots=True)
class ResourceResolver:
    dataset_root: Path

    def __post_init__(self) -> None:
        root = Path(self.dataset_root).resolve()
        if not root.is_dir():
            raise ValidationError(
                f"Dataset root does not exist or is not a directory: {root}."
            )
        object.__setattr__(self, "dataset_root", root)

    def resolve(self, reference: ResourceRef) -> ResolvedResource:
        base, suffix = _ROLE_PATHS[reference.role]
        key_path = Path(*PurePosixPath(reference.key).parts)
        if reference.role == "map":
            candidate = self.dataset_root / base / key_path / suffix
        else:
            candidate = self.dataset_root / base / key_path
            candidate = candidate.with_name(candidate.name + suffix)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValidationError(
                f"Resource {reference.role}:{reference.key} escapes the dataset root."
            ) from exc
        if not resolved.is_file():
            raise ValidationError(
                f"Resource {reference.role}:{reference.key} does not resolve to a file: {resolved}."
            )
        return ResolvedResource(reference=reference, path=resolved)

    def resolve_scenario(self, scenario: Scenario) -> tuple[ResolvedResource, ...]:
        return tuple(
            (self.resolve(reference) for reference in scenario.resource_references())
        )


def resolver_for_scenario(scenario: Scenario) -> ResourceResolver:
    if scenario.dataset_root is None:
        raise ValidationError(
            "Scenario has no dataset-root binding; provide one before resolving resources."
        )
    return ResourceResolver(scenario.dataset_root)
