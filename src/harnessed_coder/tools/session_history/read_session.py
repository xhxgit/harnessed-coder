"""Read saved session messages."""

from __future__ import annotations

from typing import Any, ClassVar

from ._shared import run_history_read
from ._tool_base import SessionHistoryToolBase


class SessionReadTool(SessionHistoryToolBase):
    """Read a turn-oriented transcript from one saved session."""

    name: ClassVar[str] = "session_read"
    description: ClassVar[str] = (
        "Read canonical saved messages as a turn-oriented transcript. Large tool "
        "arguments and results are previewed; results can be read completely with "
        "session_read_tool_result using the displayed tool_call_id and character "
        "count. Retrieve results of at most 30000 characters in one call; paginate "
        "only larger results. Optionally append each turn's latest Tool Batch "
        "Summary."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Workspace data slot id.",
            },
            "session": {
                "type": "string",
                "description": "Session name without .json, such as default.",
            },
            "start_turn": {
                "type": "integer",
                "description": "One-based first user turn. Defaults to 1.",
            },
            "turn_count": {
                "type": "integer",
                "description": "Maximum complete turns to return, 1-20.",
            },
            "include_tool_batch_summary": {
                "type": "boolean",
                "description": (
                    "Append the source-valid Tool Batch Summaries for each returned "
                    "turn. Defaults to false."
                ),
            },
        },
        "required": ["workspace_id", "session"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {
            "workspace_id",
            "session",
            "start_turn",
            "turn_count",
            "include_tool_batch_summary",
        }
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        workspace_id = kwargs.get("workspace_id")
        session = kwargs.get("session")
        if not isinstance(workspace_id, str):
            return "Error: workspace_id must be a string"
        if not isinstance(session, str):
            return "Error: session must be a string"
        return run_history_read(
            self.name,
            lambda: self.history.read_session(
                workspace_id=workspace_id,
                session=session,
                start_turn=kwargs.get("start_turn", 1),
                turn_count=kwargs.get("turn_count", 5),
                include_tool_batch_summary=kwargs.get(
                    "include_tool_batch_summary", False
                ),
            ),
        )
