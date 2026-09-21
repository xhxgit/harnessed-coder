"""List saved sessions."""

from __future__ import annotations

from typing import Any, ClassVar

from ._shared import run_history_read
from ._tool_base import SessionHistoryToolBase


class SessionListTool(SessionHistoryToolBase):
    """List saved sessions across one or all workspace data slots."""

    name: ClassVar[str] = "session_list"
    description: ClassVar[str] = (
        "List saved harnessed-coder sessions. Optionally restrict to one "
        "workspace_id from session_list_workspaces. Returns turn and message counts."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Optional workspace data slot id.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum sessions to return, 1-100.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based offset for paginated results.",
            },
        },
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"workspace_id", "limit", "offset"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        workspace_id = kwargs.get("workspace_id")
        if workspace_id is not None and not isinstance(workspace_id, str):
            return "Error: workspace_id must be a string"
        return run_history_read(
            self.name,
            lambda: self.history.list_sessions(
                workspace_id=workspace_id,
                limit=kwargs.get("limit", 20),
                offset=kwargs.get("offset", 0),
            ),
        )
