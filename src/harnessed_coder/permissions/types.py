"""Permission value types and extension protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol


class PermissionAction(str, Enum):
    """Possible execution decisions for a tool call."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionReviewAction(str, Enum):
    """Possible recommendations from an automatic permission reviewer."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class PermissionReview:
    """Automatic review recommendation for an ``ask`` decision."""

    action: PermissionReviewAction
    reason: str


@dataclass(frozen=True, slots=True)
class PermissionReviewNotice:
    """Host-visible lifecycle update for one automatic permission review."""

    stage: str
    tool_name: str
    tool_round: int
    call_index: int
    total_calls: int
    action: PermissionReviewAction | None = None
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """A concrete permission decision and the reason for it."""

    action: PermissionAction
    reason: str
    initial_action: PermissionAction | None = None
    resolution: str | None = None
    resolution_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for trace events."""
        payload = asdict(self)
        payload["action"] = self.action.value
        if self.initial_action is None:
            payload.pop("initial_action")
        else:
            payload["initial_action"] = self.initial_action.value
        if self.resolution is None:
            payload.pop("resolution")
        if self.resolution_reason is None:
            payload.pop("resolution_reason")
        return payload


@dataclass(frozen=True, slots=True)
class PermissionApprovalRequest:
    """One tool call requiring automatic review or host approval."""

    tool_name: str
    arguments: dict[str, Any]
    decision: PermissionDecision
    task_context: str = ""
    review: PermissionReview | None = None


class PermissionReviewer(Protocol):
    """Automatic reviewer for an ``ask`` permission decision."""

    def __call__(self, request: PermissionApprovalRequest) -> PermissionReview: ...


class PermissionApprover(Protocol):
    """Host callback for resolving an ``ask`` permission decision."""

    def __call__(self, request: PermissionApprovalRequest) -> bool: ...


class PermissionPolicy(Protocol):
    """Policy contract used by the tool registry before dispatch."""

    def evaluate(self, tool: Any, arguments: dict[str, Any]) -> PermissionDecision: ...
