"""Search saved session text."""

from __future__ import annotations

from typing import Any, ClassVar

from ._shared import run_history_read
from ._tool_base import SessionHistoryToolBase


class SessionSearchTool(SessionHistoryToolBase):
    """Search visible text across saved sessions."""

    name: ClassVar[str] = "session_search"
    description: ClassVar[str] = (
        "Search saved user/assistant content and each user turn's latest rolling "
        "Tool Batch Summary with a case-insensitive regular expression, then "
        "return bounded match snippets. "
        "Optionally restrict by workspace_id and session."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Case-insensitive Python regular expression.",
            },
            "workspace_id": {
                "type": "string",
                "description": "Optional workspace data slot id.",
            },
            "session": {
                "type": "string",
                "description": "Optional session name without .json.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum matches to return, 1-100.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based offset for paginated results.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"query", "workspace_id", "session", "limit", "offset"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        query = kwargs.get("query")
        workspace_id = kwargs.get("workspace_id")
        session = kwargs.get("session")
        if not isinstance(query, str) or not query.strip():
            return "Error: query must be a non-empty string"
        if workspace_id is not None and not isinstance(workspace_id, str):
            return "Error: workspace_id must be a string"
        if session is not None and not isinstance(session, str):
            return "Error: session must be a string"
        return run_history_read(
            self.name,
            lambda: self.history.search_sessions(
                query=query,
                workspace_id=workspace_id,
                session=session,
                limit=kwargs.get("limit", 20),
                offset=kwargs.get("offset", 0),
            ),
        )
