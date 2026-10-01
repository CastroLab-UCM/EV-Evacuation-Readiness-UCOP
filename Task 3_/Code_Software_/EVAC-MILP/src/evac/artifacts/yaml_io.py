"""YAML reading and writing for scenario, plan, and result files."""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Any
import yaml
from evac.errors import ArtifactError, ValidationError


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValidationError(f"YAML document contains duplicate key {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml(path: Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Cannot read YAML artifact {source}: {exc}.") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"Invalid YAML artifact {source}: {exc}.") from exc


def dump_yaml(value: Any, path: Path) -> None:
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(
            value, allow_unicode=True, default_flow_style=False, sort_keys=False
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except OSError as exc:
        raise ArtifactError(
            f"Cannot write YAML artifact {destination}: {exc}."
        ) from exc
