"""Read one saved tool result by its existing tool-call ID."""

from __future__ import annotations

from typing import Any, ClassVar

from ._shared import run_history_read
from ._tool_base import SessionHistoryToolBase


class SessionReadToolResultTool(SessionHistoryToolBase):
    """Read a character page from one complete canonical tool result."""

    name: ClassVar[str] = "session_read_tool_result"
    description: ClassVar[str] = (
        "Read complete saved tool-result text by its globally searched tool_call_id. "
        "Use an id and total character count shown by session_read or session_search. "
        "For a result of at most 30000 characters, retrieve it in one call with "
        "char_offset=0 and char_count equal to the reported total. Paginate only "
        "results larger than 30000 characters."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "tool_call_id": {
                "type": "string",
                "description": "Exact tool request id stored in any saved session.",
            },
            "char_offset": {
                "type": "integer",
                "description": "Zero-based character offset. Defaults to 0.",
            },
            "char_count": {
                "type": "integer",
                "description": "Maximum characters to return, 1-30000.",
            },
        },
        "required": ["tool_call_id"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        expected = {
            "tool_call_id",
            "char_offset",
            "char_count",
        }
        unexpected = set(kwargs) - expected
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        tool_call_id = kwargs.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return "Error: tool_call_id must be a non-empty string"
        return run_history_read(
            self.name,
            lambda: self.history.read_tool_result(
                tool_call_id=tool_call_id,
                char_offset=kwargs.get("char_offset", 0),
                char_count=kwargs.get("char_count", 20_000),
            ),
        )
