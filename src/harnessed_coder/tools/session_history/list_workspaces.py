"""List saved workspace data slots."""

from __future__ import annotations

from typing import Any, ClassVar

from ._shared import run_history_read
from ._tool_base import SessionHistoryToolBase


class SessionListWorkspacesTool(SessionHistoryToolBase):
    """List saved harnessed-coder workspace data slots."""

    name: ClassVar[str] = "session_list_workspaces"
    description: ClassVar[str] = (
        "List saved harnessed-coder workspace data slots from the configured "
        "user data directory. Returns workspace ids, optional names, roots, and "
        "session counts."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum workspaces to return, 1-100.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based offset for paginated results.",
            },
        },
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"limit", "offset"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        return run_history_read(
            self.name,
            lambda: self.history.list_workspaces(
                limit=kwargs.get("limit", 20),
                offset=kwargs.get("offset", 0),
            ),
        )
