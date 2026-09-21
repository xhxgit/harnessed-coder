"""Deterministic transcript projection for canonical session messages."""

from __future__ import annotations

import json
from typing import Any

from ...session import ConversationTurn as SessionTurn

TOOL_RESULT_PREVIEW_CHARS = 2_000
TOOL_ARGUMENT_PREVIEW_CHARS = 2_000
SEARCH_SNIPPET_CHARS = 600


def render_turn(
    turn: SessionTurn,
    *,
    tool_batch_summary: str | None = None,
    summary_through_batch: int | None = None,
) -> str:
    """Render a complete turn while previewing large tool results."""
    sections = [f"Turn {turn.number}"]
    for message in turn.messages:
        role = str(message.get("role", "unknown"))
        content = visible_message_text(message)
        if role == "assistant" and message.get("tool_calls"):
            if content:
                sections.append(f"assistant:\n{content}")
            for tool_call in _tool_calls(message):
                sections.append(_render_tool_request(tool_call))
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            preview = tool_result_preview(content, tool_call_id=tool_call_id)
            sections.append(
                f"tool result:\n"
                f"tool_call_id: {tool_call_id}\n"
                f"characters: {len(content)}\n"
                f"{preview}"
            )
            continue
        sections.append(f"{role}:\n{content}")
    if tool_batch_summary is not None:
        sections.append(
            "Tool Batch Summaries "
            f"(derived, through batch {summary_through_batch}):\n"
            f"{tool_batch_summary}"
        )
    return "\n\n".join(sections)


def visible_message_text(message: dict[str, Any]) -> str:
    """Extract visible content without exposing persisted reasoning fields."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def search_snippet(text: str, match_start: int) -> str:
    """Return a bounded one-line snippet centered around the first match."""
    match_start = len(" ".join(text[:match_start].split()))
    collapsed = " ".join(text.split())
    if len(collapsed) <= SEARCH_SNIPPET_CHARS:
        return collapsed
    start = max(0, match_start - SEARCH_SNIPPET_CHARS // 2)
    end = min(len(collapsed), start + SEARCH_SNIPPET_CHARS)
    start = max(0, end - SEARCH_SNIPPET_CHARS)
    prefix = "... " if start else ""
    suffix = " ..." if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end]}{suffix}"


def tool_result_preview(text: str, *, tool_call_id: Any = None) -> str:
    """Keep the head and tail of large tool results in a transcript."""
    if len(text) <= TOOL_RESULT_PREVIEW_CHARS:
        return text
    retrieval_hint = (
        "Retrieve the complete result in one session_read_tool_result call with "
        f"tool_call_id={tool_call_id!r}, char_offset=0, char_count={len(text)}."
        if len(text) <= 30_000
        else (
            "Retrieve the complete result with session_read_tool_result using "
            f"tool_call_id={tool_call_id!r}; paginate because it exceeds 30000 "
            "characters."
        )
    )
    marker = (
        f"\n... [{len(text) - TOOL_RESULT_PREVIEW_CHARS} characters omitted. "
        f"{retrieval_hint}] ...\n"
    )
    head_chars = TOOL_RESULT_PREVIEW_CHARS // 2
    tail_chars = TOOL_RESULT_PREVIEW_CHARS - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _render_tool_request(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name", tool_call.get("name", "unknown"))
    arguments = function.get("arguments", tool_call.get("arguments", "{}"))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
    arguments_preview = _tool_argument_preview(arguments)
    return (
        "assistant tool request:\n"
        f"id: {tool_call.get('id')}\n"
        f"tool: {name}\n"
        f"argument characters: {len(arguments)}\n"
        f"arguments: {arguments_preview}"
    )


def _tool_argument_preview(arguments: str) -> str:
    if len(arguments) <= TOOL_ARGUMENT_PREVIEW_CHARS:
        return arguments
    omitted = len(arguments) - TOOL_ARGUMENT_PREVIEW_CHARS
    marker = (
        f"\n... [{omitted} characters omitted from tool arguments; "
        "inspect the session trace for the complete request] ...\n"
    )
    head_chars = TOOL_ARGUMENT_PREVIEW_CHARS // 2
    tail_chars = TOOL_ARGUMENT_PREVIEW_CHARS - head_chars
    return f"{arguments[:head_chars]}{marker}{arguments[-tail_chars:]}"
