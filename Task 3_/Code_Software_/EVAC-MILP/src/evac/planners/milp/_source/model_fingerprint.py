"""Deterministic model fingerprints from normalized inputs and dependencies."""

from __future__ import annotations
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import numbers
from typing import Any, Mapping
import copy
import networkx as nx
import numpy as np
import pandas as pd

NORMALIZED_INPUT_SCHEMA = "normalized_mathematical_input_v1"
GROUP_CERTIFICATION_SCHEMA = "universal_group_path_certificate_v1"
PATH_LIBRARY_SCHEMA = "certified_free_flow_path_library_v1"
COMPONENT_NAMES = ("input", "group", "path")


class FingerprintError(ValueError):
    """Raised when mathematical data cannot be fingerprinted unambiguously."""


def _canonical_number(value: numbers.Real, *, context: str) -> int | float:
    if isinstance(value, bool):
        raise FingerprintError(
            f"{context} contains a boolean where a number identity is required."
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    result = float(value)
    if not math.isfinite(result):
        raise FingerprintError(f"{context} contains non-finite numeric value {result}.")
    if result == 0.0:
        return 0.0
    return result


def _typed_identity(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, bool):
        raise FingerprintError(f"{context} boolean identity {value!r} is unsupported.")
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, numbers.Integral):
        return {"type": "int", "value": int(value)}
    raise FingerprintError(
        f"{context} identity must be a string or integer; got {type(value).__name__}."
    )


def _identity_collision_key(value: Any, *, context: str) -> str:
    typed = _typed_identity(value, context=context)
    return str(typed["value"])


def _canonical_graph(graph: nx.Graph, *, context: str) -> dict[str, Any]:
    if graph.is_multigraph():
        raise FingerprintError(
            f"{context} multigraphs require an explicit edge-key contract."
        )
    identity_by_display: dict[str, dict[str, Any]] = {}
    node_records: list[dict[str, Any]] = []
    for node, attrs in graph.nodes(data=True):
        identity = _typed_identity(node, context=f"{context} node")
        display = _identity_collision_key(node, context=f"{context} node")
        previous = identity_by_display.get(display)
        if previous is not None and previous != identity:
            raise FingerprintError(
                f"{context} contains ambiguous numeric/string node identities for {display!r}."
            )
        identity_by_display[display] = identity
        node_records.append(
            {
                "id": identity,
                "attributes": canonicalize(
                    attrs, context=f"{context} node {node!r} attributes"
                ),
            }
        )
    node_records.sort(key=lambda record: canonical_json(record["id"]))
    edge_records: list[dict[str, Any]] = []
    for source, target, attrs in graph.edges(data=True):
        source_id = _typed_identity(source, context=f"{context} edge source")
        target_id = _typed_identity(target, context=f"{context} edge target")
        if not graph.is_directed() and canonical_json(source_id) > canonical_json(
            target_id
        ):
            source_id, target_id = (target_id, source_id)
        edge_records.append(
            {
                "source": source_id,
                "target": target_id,
                "attributes": canonicalize(
                    attrs, context=f"{context} edge {source!r}->{target!r} attributes"
                ),
            }
        )
    edge_records.sort(key=canonical_json)
    return {
        "graph_type": "directed" if graph.is_directed() else "undirected",
        "graph_attributes": canonicalize(
            graph.graph, context=f"{context} graph attributes"
        ),
        "nodes": node_records,
        "edges": edge_records,
    }


def _canonical_dataframe(frame: pd.DataFrame, *, context: str) -> dict[str, Any]:
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise FingerprintError(f"{context} contains duplicate columns {duplicates!r}.")
    columns = sorted((str(column) for column in frame.columns))
    if len(columns) != len(set(columns)):
        raise FingerprintError(
            f"{context} contains columns that collide after normalization."
        )
    normalized = frame.rename(columns={column: str(column) for column in frame.columns})
    records = [
        canonicalize(record, context=f"{context} row {index}")
        for index, record in enumerate(
            normalized.loc[:, columns].to_dict(orient="records")
        )
    ]
    if "id" in columns:
        seen: set[str] = set()
        for record in records:
            identity = canonical_json(record["id"])
            if identity in seen:
                raise FingerprintError(
                    f"{context} contains duplicate normalized id {record['id']!r}."
                )
            seen.add(identity)
        records.sort(key=lambda record: canonical_json(record["id"]))
    return {"columns": columns, "records": records}


def _canonical_mapping(value: Mapping[Any, Any], *, context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    display_types: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, bool):
            raise FingerprintError(
                f"{context} contains unsupported boolean mapping key {key!r}."
            )
        if isinstance(key, str):
            normalized_key = key
            display = key
            key_type = "str"
        elif isinstance(key, numbers.Integral):
            display = str(int(key))
            normalized_key = f"#int:{display}"
            key_type = "int"
        else:
            raise FingerprintError(
                f"{context} mapping key must be a string or integer; got {type(key).__name__}."
            )
        previous_type = display_types.get(display)
        if previous_type is not None and previous_type != key_type:
            raise FingerprintError(
                f"{context} contains ambiguous numeric/string mapping key {display!r}."
            )
        display_types[display] = key_type
        if normalized_key in result:
            raise FingerprintError(
                f"{context} contains duplicate normalized key {normalized_key!r}."
            )
        result[normalized_key] = canonicalize(item, context=f"{context}.{display}")
    return {key: result[key] for key in sorted(result)}


def canonicalize(value: Any, *, context: str = "payload") -> Any:
    """Return a deterministic JSON-compatible representation or fail explicitly."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Real):
        return _canonical_number(value, context=context)
    if isinstance(value, np.generic):
        return canonicalize(value.item(), context=context)
    if isinstance(value, nx.Graph):
        return _canonical_graph(value, context=context)
    if isinstance(value, pd.DataFrame):
        return _canonical_dataframe(value, context=context)
    if isinstance(value, pd.Series):
        return {
            "name": canonicalize(value.name, context=f"{context}.name"),
            "values": canonicalize(value.tolist(), context=f"{context}.values"),
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": canonicalize(value.tolist(), context=f"{context}.values"),
        }
    if is_dataclass(value):
        return canonicalize(asdict(value), context=context)
    if isinstance(value, Mapping):
        return _canonical_mapping(value, context=context)
    if isinstance(value, (list, tuple)):
        return [
            canonicalize(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [
            canonicalize(item, context=f"{context} set item") for item in value
        ]
        return sorted(normalized, key=canonical_json)
    raise FingerprintError(
        f"{context} contains unsupported object type {type(value).__name__}; object repr values are not a mathematical identity."
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComponentFingerprint:
    name: str
    schema: str
    fingerprint: str
    payload: Any

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "fingerprint": self.fingerprint,
        }


def fingerprint_component(name: str, schema: str, payload: Any) -> ComponentFingerprint:
    if name not in COMPONENT_NAMES:
        raise FingerprintError(f"Unknown mathematical fingerprint component {name!r}.")
    if not isinstance(schema, str) or not schema:
        raise FingerprintError(
            f"Fingerprint component {name!r} requires a non-empty schema."
        )
    normalized = canonicalize(payload, context=f"{name} fingerprint payload")
    envelope = {"component": name, "schema": schema, "payload": normalized}
    return ComponentFingerprint(
        name=name,
        schema=schema,
        fingerprint=sha256_fingerprint(envelope),
        payload=normalized,
    )


def normalized_input_payload(
    input_data: Mapping[str, Any], *, base_graph: nx.Graph | None = None
) -> dict[str, Any]:
    if not isinstance(input_data, Mapping):
        raise FingerprintError("input_data must be a mapping.")
    required = ("unit_system", "demand", "supply")
    missing = [key for key in required if key not in input_data]
    if missing:
        raise FingerprintError(
            f"input_data is missing required mathematical fields {missing}."
        )
    graph = base_graph if base_graph is not None else input_data.get("map")
    if graph is None:
        raise FingerprintError(
            "normalized input fingerprint requires the base map or graph."
        )
    return {
        "schema": NORMALIZED_INPUT_SCHEMA,
        "unit_system": input_data["unit_system"],
        "map": graph,
        "demand": input_data["demand"],
        "supply": input_data["supply"],
    }


def path_library_payload(library: Mapping[Any, Any]) -> dict[str, Any]:
    stable_library = json_stable_path_library(library)
    groups = [
        {"group_id": {"type": "int", "value": int(group_id)}, "paths": paths}
        for group_id, paths in stable_library.items()
    ]
    groups.sort(key=lambda item: canonical_json(item["group_id"]))
    return {"schema": PATH_LIBRARY_SCHEMA, "groups": groups}


def json_stable_path_library(library: Mapping[Any, Any]) -> dict[str, Any]:
    """Normalize path-group IDs to collision-free JSON object keys.

    Preprocessing creates nonnegative integer group IDs, while JSON object keys
    reload as strings. This domain-specific boundary makes those two storage
    representations one semantic identity without changing global typed IDs.
    """
    if not isinstance(library, Mapping):
        raise FingerprintError("path library must be a mapping by group ID.")
    normalized: dict[str, Any] = {}
    for raw_group_id, paths in library.items():
        if isinstance(raw_group_id, bool):
            raise FingerprintError("path group boolean identity is unsupported.")
        if isinstance(raw_group_id, numbers.Integral):
            group_id = int(raw_group_id)
        elif isinstance(raw_group_id, str):
            try:
                group_id = int(raw_group_id)
            except ValueError as exc:
                raise FingerprintError(
                    f"path group ID {raw_group_id!r} must be a canonical nonnegative integer."
                ) from exc
            if raw_group_id != str(group_id):
                raise FingerprintError(
                    f"path group ID {raw_group_id!r} must use canonical integer spelling."
                )
        else:
            raise FingerprintError(
                f"path group identity must be an integer or its canonical JSON string; got {type(raw_group_id).__name__}."
            )
        if group_id < 0:
            raise FingerprintError(
                f"path group ID must be nonnegative; got {group_id}."
            )
        key = str(group_id)
        if key in normalized:
            raise FingerprintError(
                f"path library contains duplicate semantic group ID {group_id}."
            )
        normalized[key] = paths
    return {key: normalized[key] for key in sorted(normalized, key=int)}


def group_certification_payload(grouping: Mapping[str, Any]) -> dict[str, Any]:
    """Return mathematical grouping metadata without measured runtime fields."""
    if not isinstance(grouping, Mapping):
        raise FingerprintError("grouping metadata must be a mapping.")
    payload = copy.deepcopy(dict(grouping))
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise FingerprintError("grouping metadata must contain a groups list.")
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise FingerprintError(
                f"grouping metadata group {index} must be a mapping."
            )
        group.pop("certification_runtime_s", None)
    return payload
