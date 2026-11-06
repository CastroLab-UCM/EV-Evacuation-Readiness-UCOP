"""Planning configuration identities."""

from __future__ import annotations
import json
from collections.abc import Mapping
from evac.domain import PreparedScenario, SemanticIdentity


def model_identity(
    prepared: PreparedScenario, planner_kind: str, controls: Mapping[str, object]
) -> SemanticIdentity:
    return SemanticIdentity(
        "planner_model",
        json.dumps(
            {
                "preparation_identity": prepared.identity.value,
                "planner_kind": planner_kind,
                "controls": controls,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


__all__ = ["model_identity"]
