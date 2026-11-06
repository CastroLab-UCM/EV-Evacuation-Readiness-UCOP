from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from evac.artifacts.yaml_io import dump_yaml, load_yaml
from evac.domain import ScenarioId
from evac.errors import ValidationError
from evac.scenario.models import ResourceRef, Scenario
from evac.scenario.codecs import load_scenario_resources

SCENARIO_SCHEMA = "evac/scenario/v1"
_TOP_LEVEL_FIELDS = {
    "schema",
    "id",
    "resources",
    "map",
    "assets",
    "demand",
    "supply",
    "traffic",
    "evaluation",
    "overlays",
}
_REQUIRED_FIELDS = _TOP_LEVEL_FIELDS - {"overlays"}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping.")
    if any((not isinstance(key, str) for key in value)):
        raise ValidationError(f"{label} keys must be strings.")
    return value


def _exact_fields(
    value: Mapping[str, Any], *, required: set[str], allowed: set[str], label: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise ValidationError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}."
        )


def _logical_key(value: object, *, role: str) -> ResourceRef:
    if not isinstance(value, str):
        raise ValidationError(
            f"Scenario {role} reference must be a string logical key."
        )
    return ResourceRef(role, value)


def scenario_from_data(
    value: object,
    *,
    document_path: Path,
    dataset_root: Path | None = None,
    validate_resources: bool = True,
) -> Scenario:
    data = _mapping(value, label="Scenario")
    _exact_fields(
        data, required=_REQUIRED_FIELDS, allowed=_TOP_LEVEL_FIELDS, label="Scenario"
    )
    if data["schema"] != SCENARIO_SCHEMA:
        raise ValidationError(
            f"Unsupported Scenario schema {data['schema']!r}; expected {SCENARIO_SCHEMA!r}."
        )
    resources = _mapping(data["resources"], label="Scenario.resources")
    _exact_fields(
        resources, required=set(), allowed={"root"}, label="Scenario.resources"
    )
    document = Path(document_path).resolve()
    if dataset_root is None:
        serialized_root = resources.get("root")
        if serialized_root is None:
            resolved_root = document.parent
        else:
            if not isinstance(serialized_root, str) or not serialized_root:
                raise ValidationError(
                    "Scenario.resources.root must be a non-empty string."
                )
            serialized_path = Path(serialized_root)
            if serialized_path.is_absolute():
                raise ValidationError(
                    "Scenario.resources.root must not be an absolute path."
                )
            resolved_root = (document.parent / serialized_path).resolve()
    else:
        resolved_root = Path(dataset_root).resolve()
    assets = _mapping(data["assets"], label="Scenario.assets")
    _exact_fields(
        assets,
        required={"vehicles", "mobile_chargers"},
        allowed={"vehicles", "mobile_chargers"},
        label="Scenario.assets",
    )
    raw_overlays = data.get("overlays", [])
    if not isinstance(raw_overlays, list):
        raise ValidationError("Scenario.overlays must be a list.")
    scenario = Scenario(
        id=ScenarioId(data["id"]),
        map=_logical_key(data["map"], role="map"),
        vehicles=_logical_key(assets["vehicles"], role="vehicles"),
        mobile_chargers=_logical_key(assets["mobile_chargers"], role="mobile_chargers"),
        demand=_logical_key(data["demand"], role="demand"),
        supply=_logical_key(data["supply"], role="supply"),
        traffic=_logical_key(data["traffic"], role="traffic"),
        evaluation=_logical_key(data["evaluation"], role="evaluation"),
        overlays=tuple((_logical_key(item, role="overlay") for item in raw_overlays)),
    )._with_resource_binding(resolved_root, document_path=document)
    if validate_resources:
        load_scenario_resources(scenario)
    return scenario


def scenario_to_data(scenario: Scenario, *, document_path: Path) -> dict[str, Any]:
    if not isinstance(scenario, Scenario):
        raise ValidationError("save_scenario requires a typed Scenario.")
    if scenario.dataset_root is None:
        raise ValidationError(
            "Scenario has no resolved dataset root; a portable resource binding cannot be written."
        )
    destination = Path(document_path).resolve()
    try:
        root_text = os.path.relpath(scenario.dataset_root, destination.parent)
    except ValueError as exc:
        raise ValidationError(
            "Scenario dataset root cannot be represented relative to the target document."
        ) from exc
    if Path(root_text).is_absolute():
        raise ValidationError("Scenario writer refuses an absolute resource root.")
    load_scenario_resources(scenario)
    value: dict[str, Any] = {
        "schema": SCENARIO_SCHEMA,
        "id": scenario.id.value,
        "resources": {"root": root_text},
        "map": scenario.map.key,
        "assets": {
            "vehicles": scenario.vehicles.key,
            "mobile_chargers": scenario.mobile_chargers.key,
        },
        "demand": scenario.demand.key,
        "supply": scenario.supply.key,
        "traffic": scenario.traffic.key,
        "evaluation": scenario.evaluation.key,
    }
    if scenario.overlays:
        value["overlays"] = [overlay.key for overlay in scenario.overlays]
    return value


def load_scenario(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    validate_resources: bool = True,
) -> Scenario:
    source = Path(path).resolve()
    value = load_yaml(source)
    return scenario_from_data(
        value,
        document_path=source,
        dataset_root=None if dataset_root is None else Path(dataset_root),
        validate_resources=validate_resources,
    )


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    destination = Path(path).resolve()
    dump_yaml(scenario_to_data(scenario, document_path=destination), destination)
