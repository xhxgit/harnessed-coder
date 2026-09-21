"""Validate provider-facing chat message sequence invariants."""

from __future__ import annotations

from typing import Any

from ..session import conversation_turns


class MessageSequenceError(ValueError):
    """Raised when chat history cannot be sent as a valid tool-call sequence."""


def validate_message_sequence(messages: list[dict[str, Any]]) -> None:
    """Require every assistant tool call to have one adjacent matching result.

    Results may be in any order because parallel-safe calls can finish
    concurrently, but no non-tool message may split a tool-result group.
    Tool-call IDs are unique across the request so results are unambiguous.
    """
    turns = conversation_turns(messages)
    for turn in turns[:-1]:
        if turn.status == "in_progress":
            raise MessageSequenceError(
                f"user turn {turn.number} is incomplete before user turn "
                f"{turn.number + 1} starts"
            )

    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    pending_call_ids: set[str] = set()
    pending_started_at: int | None = None

    for index, message in enumerate(messages):
        role = message.get("role")

        if pending_call_ids and role != "tool":
            raise MessageSequenceError(
                f"message {index} has role {role!r} before tool results from "
                f"assistant message {pending_started_at} were complete; "
                f"missing: {_formatted_ids(pending_call_ids)}"
            )

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise MessageSequenceError(
                    f"tool message {index} must have a non-empty string tool_call_id"
                )
            if tool_call_id in seen_result_ids:
                raise MessageSequenceError(
                    f"tool message {index} duplicates the result for tool_call_id "
                    f"{tool_call_id!r}"
                )
            if not pending_call_ids:
                raise MessageSequenceError(
                    f"tool message {index} is orphaned; no assistant tool call is "
                    f"awaiting tool_call_id {tool_call_id!r}"
                )
            if tool_call_id not in pending_call_ids:
                raise MessageSequenceError(
                    f"tool message {index} references unknown tool_call_id "
                    f"{tool_call_id!r}; awaiting: {_formatted_ids(pending_call_ids)}"
                )
            pending_call_ids.remove(tool_call_id)
            seen_result_ids.add(tool_call_id)
            if not pending_call_ids:
                pending_started_at = None
            continue

        if role != "assistant" or "tool_calls" not in message:
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise MessageSequenceError(
                f"assistant message {index} must have a non-empty tool_calls list"
            )

        declared_ids: set[str] = set()
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise MessageSequenceError(
                    f"assistant message {index} tool call {call_index} must be an object"
                )
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise MessageSequenceError(
                    f"assistant message {index} tool call {call_index} must have a "
                    "non-empty string id"
                )
            if tool_call_id in declared_ids or tool_call_id in seen_call_ids:
                raise MessageSequenceError(
                    f"assistant message {index} duplicates tool call id {tool_call_id!r}"
                )
            declared_ids.add(tool_call_id)

        seen_call_ids.update(declared_ids)
        pending_call_ids = declared_ids
        pending_started_at = index

    if pending_call_ids:
        raise MessageSequenceError(
            f"assistant message {pending_started_at} has tool calls without results; "
            f"missing: {_formatted_ids(pending_call_ids)}"
        )


def _formatted_ids(tool_call_ids: set[str]) -> str:
    return ", ".join(repr(tool_call_id) for tool_call_id in sorted(tool_call_ids))
