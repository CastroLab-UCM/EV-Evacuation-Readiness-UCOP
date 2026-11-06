"""Exceptions raised by the public API."""

from __future__ import annotations


class EVacError(Exception):
    """Base class for EVac validation and execution failures."""


class ValidationError(EVacError):
    """An external value does not satisfy its canonical typed contract."""


class UnsupportedCapabilityError(EVacError):
    """The MILP cannot represent a requested scenario assumption."""


class ArtifactError(EVacError):
    """A persisted artifact is missing, corrupt, unsupported, or incompatible."""


class ExecutionError(EVacError):
    """Execution failed outside the ordinary scientific outcome model."""


class MILPFormulationError(ExecutionError):
    """The authoritative MILP could not be built or materialized."""


class InvariantError(EVacError):
    """An established internal invariant was violated."""
