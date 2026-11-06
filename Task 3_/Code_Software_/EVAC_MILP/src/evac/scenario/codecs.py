from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Mapping
from evac.artifacts.yaml_io import load_yaml
from evac.domain.identifiers import (
    AssetTypeId,
    CohortId,
    EdgeId,
    MCSUnitId,
    MapId,
    NodeId,
)
from evac.errors import ValidationError
from evac.physics import ExogenousProfile, TrafficSpec
from evac.scenario.authored import (
    AuthoredDemandCohort,
    AuthoredMap,
    AuthoredMapNode,
    AuthoredObjective,
    AuthoredRoadRecord,
    AuthoredScenarioResources,
    AuthoredSupplyUnit,
    EvaluationDeclaration,
    MapDirectionality,
    MobileChargerCatalog,
    MobileChargerType,
    ScenarioOverlay,
    VehicleCatalog,
    VehicleType,
)
from evac.scenario.models import Scenario
from evac.scenario.resources import ResolvedResource, ResourceResolver


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        (not isinstance(key, str) for key in value)
    ):
        raise ValidationError(f"{label} must be a string-keyed mapping.")
    return value


def _fields(
    value: Mapping[str, Any], *, required: set[str], allowed: set[str], label: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise ValidationError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}."
        )


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot load JSON resource {path}: {exc}.") from exc


def load_map(path: str | Path) -> AuthoredMap:
    metadata_path = Path(path).resolve()
    data = _mapping(load_yaml(metadata_path), label="Map metadata")
    required = {
        "map_id",
        "display_name",
        "network",
        "coordinate_system",
        "coordinate_units",
        "directionality",
        "charger_absence",
        "units",
    }
    _fields(data, required=required, allowed=required, label="Map metadata")
    if data["charger_absence"] != "no_charger":
        raise ValidationError("Map charger_absence must be 'no_charger'.")
    units = _mapping(data["units"], label="Map units")
    if dict(units) != {"distance": "m", "free_flow_time": "s", "power": "kW"}:
        raise ValidationError(
            "Map units must use the canonical distance/time/power declarations."
        )
    network_name = data["network"]
    if not isinstance(network_name, str) or Path(network_name).name != network_name:
        raise ValidationError("Map network must be a sibling filename.")
    network_data = _mapping(
        _load_json(metadata_path.parent / network_name), label="GeoJSON"
    )
    _fields(
        network_data,
        required={"type", "features"},
        allowed={"type", "features"},
        label="GeoJSON",
    )
    if network_data["type"] != "FeatureCollection" or not isinstance(
        network_data["features"], list
    ):
        raise ValidationError("Map network must be a GeoJSON FeatureCollection.")
    try:
        directionality = MapDirectionality(data["directionality"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("Unsupported Map directionality declaration.") from exc
    nodes: list[AuthoredMapNode] = []
    raw_roads: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, item in enumerate(network_data["features"]):
        feature = _mapping(item, label=f"GeoJSON feature {index}")
        _fields(
            feature,
            required={"type", "geometry", "properties"},
            allowed={"type", "geometry", "properties"},
            label=f"GeoJSON feature {index}",
        )
        if feature["type"] != "Feature":
            raise ValidationError(f"GeoJSON feature {index} has invalid type.")
        geometry = _mapping(feature["geometry"], label=f"GeoJSON geometry {index}")
        _fields(
            geometry,
            required={"type", "coordinates"},
            allowed={"type", "coordinates"},
            label=f"GeoJSON geometry {index}",
        )
        properties = _mapping(
            feature["properties"], label=f"GeoJSON properties {index}"
        )
        if geometry["type"] == "Point":
            _fields(
                properties,
                required={"node_id"},
                allowed={"node_id", "name", "infrastructure"},
                label=f"Map node {index}",
            )
            coordinates = geometry["coordinates"]
            if not isinstance(coordinates, list) or len(coordinates) != 2:
                raise ValidationError(f"Map node {index} requires a two-value Point.")
            infrastructure = _mapping(
                properties.get("infrastructure", {}),
                label=f"Map node {index} infrastructure",
            )
            _fields(
                infrastructure,
                required=set(),
                allowed={
                    "has_charger",
                    "ports",
                    "power",
                    "mobile_charger_limit",
                    "charger_type",
                    "notes",
                },
                label=f"Map node {index} infrastructure",
            )
            has_charger = infrastructure.get("has_charger", False)
            if not isinstance(has_charger, bool):
                raise ValidationError("Map infrastructure.has_charger must be boolean.")
            ports = infrastructure.get("ports", 0)
            power = infrastructure.get("power")
            if not has_charger and (ports != 0 or power is not None):
                raise ValidationError(
                    "Map fixed-charger fields contradict has_charger=false."
                )
            nodes.append(
                AuthoredMapNode(
                    id=NodeId(properties["node_id"]),
                    coordinate=tuple(coordinates),
                    fixed_charger_ports=ports,
                    fixed_charger_power_kw=power,
                    mcs_limit=infrastructure.get("mobile_charger_limit", 0),
                    charger_type_label=infrastructure.get("charger_type"),
                    notes=infrastructure.get("notes"),
                )
            )
        elif geometry["type"] == "LineString":
            raw_roads.append((geometry, properties))
        else:
            raise ValidationError(
                f"Unsupported GeoJSON geometry type {geometry['type']!r}."
            )
    endpoint_counts: dict[
        tuple[type[str] | type[int], str | int, type[str] | type[int], str | int], int
    ] = {}
    for _, properties in raw_roads:
        if "source" not in properties or "target" not in properties:
            raise ValidationError("Road properties require source and target IDs.")
        key = (
            type(properties["source"]),
            properties["source"],
            type(properties["target"]),
            properties["target"],
        )
        endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
    roads: list[AuthoredRoadRecord] = []
    for index, (geometry, properties) in enumerate(raw_roads):
        required_road = {"source", "target", "length_m"}
        _fields(
            properties,
            required=required_road,
            allowed=required_road
            | {"edge_id", "alpha", "beta", "free_flow_time", "source_speed_mps"},
            label=f"Map road {index}",
        )
        if "free_flow_time" not in properties and "source_speed_mps" not in properties:
            raise ValidationError(
                f"Map road {index} requires free_flow_time or source_speed_mps."
            )
        coordinates = geometry["coordinates"]
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValidationError(
                f"Map road {index} requires a LineString with at least two points."
            )
        source = properties["source"]
        target = properties["target"]
        edge_value = properties.get("edge_id")
        if edge_value is None:
            key = (type(source), source, type(target), target)
            if endpoint_counts[key] != 1:
                raise ValidationError(
                    "Parallel authored roads require explicit edge_id values."
                )
            separator = (
                "<->"
                if directionality is MapDirectionality.BIDIRECTIONAL_PHYSICAL_LINKS
                else "->"
            )
            edge_value = f"{source}{separator}{target}"
        roads.append(
            AuthoredRoadRecord(
                id=EdgeId(edge_value),
                source=NodeId(source),
                target=NodeId(target),
                distance_m=properties["length_m"],
                free_flow_duration_seconds=properties.get("free_flow_time"),
                source_speed_m_per_second=properties.get("source_speed_mps"),
            )
        )
    return AuthoredMap(
        id=MapId(data["map_id"]),
        display_name=data["display_name"],
        coordinate_system=data["coordinate_system"],
        coordinate_units=data["coordinate_units"],
        directionality=directionality,
        nodes=tuple(nodes),
        roads=tuple(roads),
    )


def load_vehicle_catalog(path: str | Path) -> VehicleCatalog:
    data = _mapping(load_yaml(Path(path)), label="Vehicle catalog")
    _fields(
        data,
        required={"schema", "units", "types"},
        allowed={"schema", "units", "types"},
        label="Vehicle catalog",
    )
    if data["schema"] != "evac/vehicle-catalog/v1":
        raise ValidationError("Unsupported vehicle-catalog schema.")
    if dict(_mapping(data["units"], label="Vehicle units")) != {
        "battery": "kWh",
        "efficiency": "m/kWh",
    }:
        raise ValidationError("Vehicle catalog units are not canonical.")
    types = _mapping(data["types"], label="Vehicle types")
    result: list[VehicleType] = []
    for key, raw in types.items():
        item = _mapping(raw, label=f"Vehicle type {key!r}")
        _fields(
            item,
            required={"battery_kwh", "efficiency_m_per_kwh"},
            allowed={"battery_kwh", "efficiency_m_per_kwh"},
            label=f"Vehicle type {key!r}",
        )
        result.append(
            VehicleType(
                AssetTypeId(key), item["battery_kwh"], item["efficiency_m_per_kwh"]
            )
        )
    return VehicleCatalog(tuple(result))


def load_mobile_charger_catalog(path: str | Path) -> MobileChargerCatalog:
    data = _mapping(load_yaml(Path(path)), label="Mobile-charger catalog")
    _fields(
        data,
        required={"schema", "units", "types"},
        allowed={"schema", "units", "types"},
        label="Mobile-charger catalog",
    )
    if data["schema"] != "evac/mobile-charger-catalog/v1":
        raise ValidationError("Unsupported mobile-charger-catalog schema.")
    if dict(_mapping(data["units"], label="Mobile-charger units")) != {
        "battery": "kWh",
        "power": "kW",
    }:
        raise ValidationError("Mobile-charger catalog units are not canonical.")
    types = _mapping(data["types"], label="Mobile-charger types")
    result: list[MobileChargerType] = []
    for key, raw in types.items():
        item = _mapping(raw, label=f"Mobile-charger type {key!r}")
        _fields(
            item,
            required={"battery_kwh", "power_kw", "port_count"},
            allowed={"battery_kwh", "power_kw", "port_count", "power_scope"},
            label=f"Mobile-charger type {key!r}",
        )
        power_scope = item.get("power_scope", "unit_total")
        if power_scope not in {"unit_total", "per_port"}:
            raise ValidationError(
                f"Mobile-charger type {key!r} power_scope must be 'unit_total' or 'per_port'."
            )
        result.append(
            MobileChargerType(
                AssetTypeId(key),
                item["battery_kwh"],
                item["power_kw"],
                item["port_count"],
                power_scope == "unit_total",
            )
        )
    return MobileChargerCatalog(tuple(result))


def _csv_rows(
    path: Path, *, required: set[str], optional: set[str]
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)):
                raise ValidationError(f"CSV {path} has missing or duplicate headers.")
            if set(fields) != required and set(fields) != required | optional:
                raise ValidationError(
                    f"CSV {path} fields mismatch: expected {sorted(required)!r} with optional {sorted(optional)!r}; got {fields!r}."
                )
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValidationError(f"Cannot load CSV resource {path}: {exc}.") from exc


def _integer_text(value: str, *, label: str, positive: bool = False) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be an integer; got {value!r}.") from exc
    if str(parsed) != value.strip() or (positive and parsed <= 0):
        qualifier = "positive integer" if positive else "integer"
        raise ValidationError(
            f"{label} must be a canonical {qualifier}; got {value!r}."
        )
    return parsed


def _float_text(value: str, *, label: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValidationError(f"{label} must be numeric; got {value!r}.") from exc


def load_demand(path: str | Path) -> tuple[AuthoredDemandCohort, ...]:
    source = Path(path)
    required = {
        "cohort_id",
        "vehicle_type_id",
        "speed_m_per_second",
        "initial_soc",
        "origin_node_id",
        "destination_node_id",
        "size",
    }
    rows = _csv_rows(source, required=required, optional={"source_name"})
    if not rows:
        raise ValidationError("Demand CSV requires at least one cohort.")
    result = tuple(
        (
            AuthoredDemandCohort(
                id=CohortId(row["cohort_id"]),
                vehicle_type_id=AssetTypeId(row["vehicle_type_id"]),
                speed_m_per_second=_float_text(
                    row["speed_m_per_second"], label="demand speed"
                ),
                initial_soc=_float_text(row["initial_soc"], label="demand SOC"),
                origin=NodeId(row["origin_node_id"]),
                destination=NodeId(row["destination_node_id"]),
                size=_integer_text(row["size"], label="demand size", positive=True),
                source_name=row.get("source_name"),
            )
            for row in rows
        )
    )
    if len({item.id for item in result}) != len(result):
        raise ValidationError("Demand CSV contains duplicate cohort IDs.")
    return result


def load_supply(path: str | Path) -> tuple[AuthoredSupplyUnit, ...]:
    rows = _csv_rows(
        Path(path),
        required={"mcs_unit_id", "mcs_type_id", "initial_soc"},
        optional=set(),
    )
    result = tuple(
        (
            AuthoredSupplyUnit(
                id=MCSUnitId(row["mcs_unit_id"]),
                mcs_type_id=AssetTypeId(row["mcs_type_id"]),
                initial_soc=_float_text(row["initial_soc"], label="supply SOC"),
            )
            for row in rows
        )
    )
    if len({item.id for item in result}) != len(result):
        raise ValidationError("Supply CSV contains duplicate MCS unit IDs.")
    return result


def load_traffic(path: str | Path) -> TrafficSpec:
    data = _mapping(load_yaml(Path(path)), label="Traffic")
    _fields(
        data,
        required={"schema", "elapsed_exogenous_profile"},
        allowed={"schema", "elapsed_exogenous_profile"},
        label="Traffic",
    )
    if data["schema"] != "evac/traffic/v1":
        raise ValidationError("Unsupported traffic schema.")
    return TrafficSpec(ExogenousProfile.from_data(data["elapsed_exogenous_profile"]))


def load_evaluation(path: str | Path) -> EvaluationDeclaration:
    data = _mapping(load_yaml(Path(path)), label="Evaluation")
    common = {"schema"}
    if data.get("schema") != "evac/evaluation/v2":
        raise ValidationError("Unsupported evaluation schema.")
    _fields(
        data,
        required=common | {"objective"},
        allowed=common | {"objective"},
        label="Evaluation",
    )
    objective = _mapping(data["objective"], label="Evaluation objective")
    fields = {"name", "aggregation", "unit"}
    _fields(objective, required=fields, allowed=fields, label="Evaluation objective")
    if objective["name"] != "completion_time":
        raise ValidationError("Evaluation objective.name must be 'completion_time'.")
    if objective["aggregation"] not in {"mean", "max"}:
        raise ValidationError("Evaluation aggregation must be 'mean' or 'max'.")
    if objective["unit"] != "s":
        raise ValidationError("completion_time unit must be seconds ('s').")
    objectives = [
        AuthoredObjective(
            name=objective["name"],
            aggregation=objective["aggregation"],
            unit=objective["unit"],
        )
    ]
    return EvaluationDeclaration(tuple(objectives))


_OVERLAY_FIELDS = {
    "schema",
    "soc_bounds",
    "grouping",
    "numerical_policy",
    "route_selection",
    "departure_windows",
    "mcs_site_limits",
    "fixed_charger_site_overrides",
    "charging_physics",
    "charging_contention",
}


def load_overlay(path: str | Path) -> ScenarioOverlay:
    data = _mapping(load_yaml(Path(path)), label="Scenario overlay")
    _fields(
        data, required={"schema"}, allowed=_OVERLAY_FIELDS, label="Scenario overlay"
    )
    if data["schema"] != "evac/scenario-overlay/v1":
        raise ValidationError("Unsupported Scenario-overlay schema.")
    return ScenarioOverlay(
        {key: value for key, value in data.items() if key != "schema"}
    )


def load_scenario_resources(scenario: Scenario) -> AuthoredScenarioResources:
    if scenario.dataset_root is None:
        raise ValidationError(
            "Scenario requires an explicit dataset binding before resource loading."
        )
    resolved = ResourceResolver(scenario.dataset_root).resolve_scenario(scenario)
    by_role: dict[str, list[ResolvedResource]] = {}
    for item in resolved:
        by_role.setdefault(item.reference.role, []).append(item)
    resources = AuthoredScenarioResources(
        map=load_map(by_role["map"][0].path),
        vehicles=load_vehicle_catalog(by_role["vehicles"][0].path),
        mobile_chargers=load_mobile_charger_catalog(by_role["mobile_chargers"][0].path),
        demand=load_demand(by_role["demand"][0].path),
        supply=load_supply(by_role["supply"][0].path),
        traffic=load_traffic(by_role["traffic"][0].path),
        evaluation=load_evaluation(by_role["evaluation"][0].path),
        overlays=tuple(
            (load_overlay(item.path) for item in by_role.get("overlay", []))
        ),
    )
    node_ids = {node.id for node in resources.map.nodes}
    vehicle_type_ids = {item.id for item in resources.vehicles.types}
    mcs_type_ids = {item.id for item in resources.mobile_chargers.types}
    for cohort in resources.demand:
        if cohort.vehicle_type_id not in vehicle_type_ids:
            raise ValidationError(
                f"Demand cohort {cohort.id} references an unknown vehicle type."
            )
        if cohort.origin not in node_ids or cohort.destination not in node_ids:
            raise ValidationError(
                f"Demand cohort {cohort.id} references an unknown Map node."
            )
    for unit in resources.supply:
        if unit.mcs_type_id not in mcs_type_ids:
            raise ValidationError(
                f"Supply unit {unit.id} references an unknown MCS type."
            )
    return resources
