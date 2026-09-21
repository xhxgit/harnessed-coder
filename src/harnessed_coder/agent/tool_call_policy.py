"""Model-facing policies that gate tool-call execution."""

from __future__ import annotations


MISSING_FIRST_TOOL_CALL_CONTENT_STATUS = "missing_first_tool_call_content"
MISSING_FIRST_TOOL_CALL_CONTENT_MESSAGE = (
    "Error: the first tool-call batch in each user turn must include a concise, "
    "non-empty assistant content note before tools can run. No tools from this "
    "batch were executed. Retry the tool calls with a brief visible explanation "
    "of the immediate objective in assistant content."
)
