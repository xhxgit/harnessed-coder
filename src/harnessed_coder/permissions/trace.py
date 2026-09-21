"""Structured trace payloads for permission decisions."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from ..llm import ToolCall
from .types import PermissionDecision


def permission_trace_payload(
    tool_call: ToolCall,
    decision: PermissionDecision,
) -> dict[str, Any]:
    """Describe the complete decision chain for one concrete tool call."""
    resolution = decision.resolution
    mechanical_action = decision.initial_action or decision.action
    payload: dict[str, Any] = {
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        "input": _input_summary(tool_call),
        "mechanical_policy": {
            "action": mechanical_action.value,
            "reason": decision.reason,
        },
    }

    if resolution in {"auto_approved", "auto_denied"}:
        payload["automatic_review"] = {
            "action": "allow" if resolution == "auto_approved" else "deny",
            "reason": decision.resolution_reason,
        }
    elif decision.resolution_reason is not None:
        payload["automatic_review"] = {
            "action": "abstain",
            "reason": decision.resolution_reason,
        }

    if resolution is not None and resolution.startswith("user_"):
        payload["human_approval"] = {
            "action": "allow" if resolution.startswith("user_approved") else "deny",
        }
    elif resolution in {"approval_interrupted", "approval_error"}:
        payload["human_approval"] = {"action": resolution.removeprefix("approval_")}
    elif resolution in {"approval_unavailable", "auto_abstained"}:
        payload["human_approval"] = {"action": "unavailable"}

    payload["final"] = {
        "action": decision.action.value,
        "source": _decision_source(decision),
        "resolution": resolution or "mechanical",
    }
    return payload


def _input_summary(tool_call: ToolCall) -> dict[str, Any]:
    serialized = json.dumps(
        tool_call.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    summary: dict[str, Any] = {
        "argument_keys": sorted(tool_call.arguments),
        "arguments_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
    }
    command = tool_call.arguments.get("command")
    if isinstance(command, str):
        one_line = " ".join(command.split())
        summary.update(
            {
                "command_summary": (
                    one_line if len(one_line) <= 160 else f"{one_line[:160]}..."
                ),
                "command_chars": len(command),
                "command_sha256": sha256(command.encode("utf-8")).hexdigest(),
            }
        )
    return summary


def _decision_source(decision: PermissionDecision) -> str:
    if decision.initial_action is None:
        return "mechanical_policy"
    if decision.resolution in {"auto_approved", "auto_denied"}:
        return "automatic_reviewer"
    if decision.resolution is not None and decision.resolution.startswith("user_"):
        return "user"
    return "fail_closed"
