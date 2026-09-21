"""Repair Agent history that stopped mid tool-call turn."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..session.message_metadata import SESSION_RECOVERY_CONTENT_SOURCE

INTERRUPTED_TOOL_RESULT_MESSAGE = (
    "Error: the previous session was interrupted before this tool result was "
    "recorded. The execution state is unknown: the tool may have started or "
    "completed side effects before the interruption. Do not assume it did not run "
    "and do not retry it blindly. Inspect current files or external state first. "
    "Retry only after confirming the action did not occur, or when the action is "
    "known to be safe and idempotent."
)

INTERRUPTED_TURN_CLOSURE_MESSAGE = (
    "Previous turn was interrupted before I could send a final response. "
    "The recorded tool results above are partial progress only, and any interrupted "
    "tool without a recorded result has unknown execution state. I will use the "
    "user's next message to decide how to continue."
)


@dataclass(frozen=True)
class SessionRecoveryResult:
    messages: list[dict[str, Any]]
    changed: bool


def close_interrupted_tool_turn(messages: list[dict[str, Any]]) -> SessionRecoveryResult:
    """Close a saved turn that stopped after tool calls because the process died.

    Providers generally require every assistant tool call to have a matching
    tool message before later conversation continues. Even when that structure
    is present, a history that ends in tool results has no assistant response
    consuming them. Closing it locally lets a resumed session accept a natural
    "continue" message without pretending the interrupted model completed work.
    """
    copied_messages = [deepcopy(message) for message in messages]
    last_user_index = _last_message_index(copied_messages, role="user")
    if last_user_index is None:
        return SessionRecoveryResult(copied_messages, changed=False)

    turn_messages = copied_messages[last_user_index + 1 :]
    last_tool_call_assistant_index = _last_assistant_tool_call_index(turn_messages)
    if last_tool_call_assistant_index is None:
        return SessionRecoveryResult(copied_messages, changed=False)

    later_assistant_index = _last_message_index(
        turn_messages[last_tool_call_assistant_index + 1 :],
        role="assistant",
    )
    if later_assistant_index is not None:
        return SessionRecoveryResult(copied_messages, changed=False)

    repaired = [*copied_messages]
    assistant_message = turn_messages[last_tool_call_assistant_index]
    expected_tool_call_ids = _tool_call_ids(assistant_message)
    existing_tool_call_ids = {
        str(message.get("tool_call_id"))
        for message in turn_messages[last_tool_call_assistant_index + 1 :]
        if message.get("role") == "tool" and message.get("tool_call_id") is not None
    }
    for tool_call_id in expected_tool_call_ids:
        if tool_call_id not in existing_tool_call_ids:
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": INTERRUPTED_TOOL_RESULT_MESSAGE,
                }
            )

    repaired.append(
        {
            "role": "assistant",
            "content": INTERRUPTED_TURN_CLOSURE_MESSAGE,
            "content_source": SESSION_RECOVERY_CONTENT_SOURCE,
        }
    )
    return SessionRecoveryResult(repaired, changed=True)


def _last_message_index(messages: list[dict[str, Any]], *, role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == role:
            return index
    return None


def _last_assistant_tool_call_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and _tool_call_ids(message):
            return index
    return None


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            ids.append(tool_call_id)
    return ids
