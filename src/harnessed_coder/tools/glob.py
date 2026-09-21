"""Glob workspace files tool."""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, ClassVar

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata


class GlobTool(WorkspaceTool, Tool):
    """List workspace paths matching a glob pattern."""

    name: ClassVar[str] = "glob"
    description: ClassVar[str] = "Find files in the workspace matching a glob pattern."
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=True,
        is_parallel_safe=True,
        skip_permission_review=True,
        result_size_hint="paged",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Relative glob pattern, for example src/**/*.py.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum number of matching files to return.",
                "default": 200,
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of sorted matching files to skip before returning results.",
                "default": 0,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"pattern", "limit", "offset"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        pattern = kwargs.get("pattern")
        limit = kwargs.get("limit", 200)
        offset = kwargs.get("offset", 0)
        if not isinstance(pattern, str) or not pattern.strip():
            return "Error: pattern must be a non-empty string"
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            return "Error: limit must be an integer between 1 and 1000"
        if not isinstance(offset, int) or offset < 0:
            return "Error: offset must be a non-negative integer"

        normalized = pattern.replace("\\", "/")
        pure_pattern = PurePath(normalized)
        if Path(normalized).is_absolute() or ".." in pure_pattern.parts:
            return "Error: pattern must stay within the workspace"

        try:
            matches = sorted(
                self._relative_name(path.resolve())
                for path in self.root.glob(normalized)
                if path.is_file() and path.resolve().is_relative_to(self.root)
            )
        except (OSError, ValueError) as exc:
            return f"Error: {exc}"

        if not matches:
            return "No files matched."
        page = matches[offset : offset + limit]
        if not page:
            return f"No files matched at offset {offset}. Total matches: {len(matches)}."

        displayed = list(page)
        if offset:
            displayed.insert(0, f"... ({min(offset, len(matches))} earlier matches)")
        remaining = len(matches) - offset - len(page)
        if remaining > 0:
            displayed.append(f"... ({remaining} more matches)")
        return "\n".join(displayed)
