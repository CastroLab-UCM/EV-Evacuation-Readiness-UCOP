from __future__ import annotations
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from evac.domain.identifiers import ScenarioId
from evac.errors import ValidationError

_KEY_SEGMENT = re.compile("^[a-z0-9][a-z0-9_-]*$")
RESOURCE_ROLES = frozenset(
    {
        "map",
        "vehicles",
        "mobile_chargers",
        "demand",
        "supply",
        "traffic",
        "evaluation",
        "overlay",
    }
)


@dataclass(frozen=True, slots=True)
class ResourceRef:
    role: str
    key: str

    def __post_init__(self) -> None:
        if self.role not in RESOURCE_ROLES:
            raise ValidationError(
                f"Unsupported resource role {self.role!r}; expected one of {sorted(RESOURCE_ROLES)!r}."
            )
        if not isinstance(self.key, str) or not self.key:
            raise ValidationError("Resource keys must be non-empty strings.")
        if self.key.startswith("/") or "\\" in self.key:
            raise ValidationError(
                f"Resource key must be a portable logical key: {self.key!r}."
            )
        segments = self.key.split("/")
        if any(
            (
                segment in {"", ".", ".."} or not _KEY_SEGMENT.fullmatch(segment)
                for segment in segments
            )
        ):
            raise ValidationError(
                f"Resource keys must contain lowercase logical segments made of letters, digits, '_' or '-'; got {self.key!r}."
            )


@dataclass(frozen=True, slots=True)
class Scenario:
    id: ScenarioId
    map: ResourceRef
    vehicles: ResourceRef
    mobile_chargers: ResourceRef
    demand: ResourceRef
    supply: ResourceRef
    traffic: ResourceRef
    evaluation: ResourceRef
    overlays: tuple[ResourceRef, ...] = ()
    _dataset_root: Path | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _document_path: Path | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        expected_roles = {
            "map": self.map,
            "vehicles": self.vehicles,
            "mobile_chargers": self.mobile_chargers,
            "demand": self.demand,
            "supply": self.supply,
            "traffic": self.traffic,
            "evaluation": self.evaluation,
        }
        for role, reference in expected_roles.items():
            if not isinstance(reference, ResourceRef) or reference.role != role:
                raise ValidationError(
                    f"Scenario.{role} must be a ResourceRef with role {role!r}."
                )
        for index, overlay in enumerate(self.overlays):
            if not isinstance(overlay, ResourceRef) or overlay.role != "overlay":
                raise ValidationError(
                    f"Scenario.overlays[{index}] must be a ResourceRef with role 'overlay'."
                )

    @property
    def dataset_root(self) -> Path | None:
        return self._dataset_root

    @property
    def document_path(self) -> Path | None:
        return self._document_path

    def resource_references(self) -> tuple[ResourceRef, ...]:
        return (
            self.map,
            self.vehicles,
            self.mobile_chargers,
            self.demand,
            self.supply,
            self.traffic,
            self.evaluation,
            *self.overlays,
        )

    def bind_resources(self, dataset_root: str | Path) -> Scenario:
        from evac.scenario.resources import ResourceResolver

        bound = self._with_resource_binding(dataset_root)
        ResourceResolver(bound.dataset_root).resolve_scenario(bound)
        return bound

    def _with_resource_binding(
        self, dataset_root: str | Path, *, document_path: str | Path | None = None
    ) -> Scenario:
        bound = replace(self)
        object.__setattr__(bound, "_dataset_root", Path(dataset_root).resolve())
        source = (
            self._document_path
            if document_path is None
            else Path(document_path).resolve()
        )
        object.__setattr__(bound, "_document_path", source)
        return bound
