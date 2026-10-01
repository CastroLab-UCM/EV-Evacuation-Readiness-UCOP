"""Plan inspection results and issues."""

from __future__ import annotations
from dataclasses import dataclass
from evac.domain import SemanticIdentity
from evac.errors import ValidationError


@dataclass(frozen=True, slots=True)
class PlanInspectionIssue:
    code: str
    detail: str
    subject_namespace: str | None = None
    subject_value: str | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValidationError("PlanInspectionIssue.code must be non-empty.")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValidationError("PlanInspectionIssue.detail must be non-empty.")
        if self.subject_namespace is not None and (
            not isinstance(self.subject_namespace, str) or not self.subject_namespace
        ):
            raise ValidationError(
                "PlanInspectionIssue.subject_namespace must be non-empty when set."
            )
        if self.subject_value is not None and (
            not isinstance(self.subject_value, (str, int))
        ):
            raise ValidationError(
                "PlanInspectionIssue.subject_value must be a string or integer when set."
            )


@dataclass(frozen=True, slots=True)
class PlanInspection:
    preparation_identity: SemanticIdentity
    plan_identity: SemanticIdentity
    issues: tuple[PlanInspectionIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.preparation_identity, SemanticIdentity):
            raise ValidationError("PlanInspection.preparation_identity must be typed.")
        if not isinstance(self.plan_identity, SemanticIdentity):
            raise ValidationError("PlanInspection.plan_identity must be typed.")
        if any((not isinstance(issue, PlanInspectionIssue) for issue in self.issues)):
            raise ValidationError("PlanInspection.issues must contain typed issues.")

    @property
    def valid(self) -> bool:
        return not self.issues


__all__ = ["PlanInspection", "PlanInspectionIssue"]
