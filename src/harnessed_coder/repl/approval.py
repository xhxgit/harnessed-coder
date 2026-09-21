"""Interactive permission approval for the REPL host."""

from __future__ import annotations

import json
from typing import TextIO

from ..agent import Agent
from ..permissions import PermissionApprovalRequest
from .input import CallableReplInput, PromptToolkitReplInput


def bind_permission_approver(
    agent: Agent,
    *,
    repl_input: CallableReplInput | PromptToolkitReplInput,
    output: TextIO,
) -> None:
    """Bind an interactive allow-once prompt to an Agent."""

    def approve(request: PermissionApprovalRequest) -> bool:
        output.write(f"\n! Permission required · {request.tool_name}\n")
        output.write(f"  Policy: {request.decision.reason}\n")
        if request.review is not None:
            output.write(f"  Review: {request.review.action.value}\n")
            output.write(f"  Review reason: {request.review.reason}\n")
        detail = _permission_request_detail(request)
        if detail:
            output.write(f"  {detail}\n")
        output.flush()
        try:
            answer = repl_input.read("Allow once? [y/N] ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            output.write("\n")
            output.flush()
            return False
        return answer in {"y", "yes"}

    agent.set_permission_approver(approve)


def _permission_request_detail(request: PermissionApprovalRequest) -> str:
    for key, label in (("command", "Command"), ("path", "Path")):
        value = request.arguments.get(key)
        if isinstance(value, str):
            return f"{label}: {_truncate_permission_text(value)}"
    if not request.arguments:
        return ""
    rendered = json.dumps(request.arguments, ensure_ascii=False, default=str)
    return f"Arguments: {_truncate_permission_text(rendered)}"


def _truncate_permission_text(value: str, *, limit: int = 500) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
