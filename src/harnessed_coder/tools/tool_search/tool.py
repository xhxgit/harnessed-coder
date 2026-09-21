"""Model-visible tool for discovering deferred tools."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..base import Tool, ToolMetadata
from ..tool_registry import ToolRegistry
from .matcher import ToolMatcher


logger = logging.getLogger(__name__)


class ToolSearchTool(Tool):
    """Match and expose registered-but-hidden tools."""

    name: ClassVar[str] = "tool_search"
    description: ClassVar[str] = (
        "Find deferred tools by describing the required capability and expose "
        "the matched tools for subsequent model calls in the current user turn. "
        "Use this when the current visible tool list does not include the needed "
        "capability. Newly exposed tools are unavailable to other tool calls in "
        "the same parallel batch; use them only after the next model response."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=True,
        is_parallel_safe=False,
        skip_permission_review=True,
        result_size_hint="small",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "Precise description of the required capability, including "
                    "the action, object, and important constraints. Matching is "
                    "performed by a model and returns every plausibly useful "
                    "tool, whose "
                    "complete schemas become available in the next model request. "
                    "Resolve known ambiguities in this description whenever "
                    "possible. "
                    "To improve precision, explicitly name tools to select when "
                    "known, or name tools to exclude when they are not needed."
                ),
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        registry: ToolRegistry,
        matcher: ToolMatcher,
    ) -> None:
        self._registry = registry
        self._matcher = matcher

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"description"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"

        description = kwargs.get("description")
        if not isinstance(description, str) or not description.strip():
            return "Error: description must be a non-empty string"

        deferred_tools = self._registry.deferred_tools()
        if not deferred_tools:
            return "No deferred tools are currently available."

        try:
            matched_names = self._matcher.match(description.strip(), deferred_tools)
        except Exception as exc:
            logger.exception("Semantic tool search failed")
            return f"Error: semantic tool search failed: {type(exc).__name__}: {exc}"
        if not matched_names:
            return "No deferred tools matched the requested capability."

        newly_visible = set(self._registry.expose_for_turn(matched_names))
        lines = [
            f"Exposed {len(matched_names)} tool(s) for later model calls in "
            "the current user turn:",
            "Use newly exposed tools only after the next model response, not in "
            "the current parallel tool-call batch.",
        ]
        for name in matched_names:
            status = "new" if name in newly_visible else "already visible"
            lines.append(f"- {name} ({status})")
        return "\n".join(lines)
