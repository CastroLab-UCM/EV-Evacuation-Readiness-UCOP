"""Stable ordering for typed identifiers."""

from __future__ import annotations
from typing import Any


def identifier_key(value: Any) -> tuple[str, str]:
    """Return a stable ordering key across identifier types."""
    raw = value.value
    return ("int" if isinstance(raw, int) else "str", str(raw))


__all__ = ["identifier_key"]
