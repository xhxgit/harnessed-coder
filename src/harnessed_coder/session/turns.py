"""Canonical user-turn projection shared across conversation subsystems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .message_metadata import SESSION_RECOVERY_CONTENT_SOURCE


_TurnStatus = Literal["in_progress", "completed", "aborted"]


@dataclass(frozen=True)
class ConversationTurn:
    """One user-message block projected from a specific input message list.

    ``number`` is one-based within that input only. It is a canonical turn
    number only when the caller supplied canonical conversation history.
    """

    number: int
    start_index: int
    end_index: int
    messages: list[dict[str, Any]]
    status: _TurnStatus


def conversation_turns(messages: list[dict[str, Any]]) -> list[ConversationTurn]:
    """Project any message list into turns, grouping adjacent user messages.

    This pure helper does not identify whether input is canonical, compressed,
    or provider-facing. Callers may interpret ``ConversationTurn.number`` as a
    canonical turn ID only when ``messages`` is canonical history (or a strict
    one-to-one role/order projection of it).
    """
    boundaries = _turn_boundaries(messages)
    return [
        ConversationTurn(
            number=number,
            start_index=start,
            end_index=end,
            messages=messages[start:end],
            status=_turn_status(messages[start:end]),
        )
        for number, (start, end) in enumerate(boundaries, start=1)
    ]


def _turn_boundaries(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    starts: list[int] = []
    previous_role: object = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user" and previous_role != "user":
            starts.append(index)
        previous_role = role
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
        for index, start in enumerate(starts)
    ]


def _turn_status(messages: list[dict[str, Any]]) -> _TurnStatus:
    for message in reversed(messages):
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if message.get("content_source") == SESSION_RECOVERY_CONTENT_SOURCE:
            return "aborted"
        return "completed"
    return "in_progress"
