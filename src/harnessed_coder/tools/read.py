"""Read file tool."""

from __future__ import annotations

from itertools import islice
from typing import Any, ClassVar

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata
from ._file_content import (
    UnsupportedTextFileError,
    normalize_text_encoding,
    read_text_file,
    unsupported_text_file_error,
)


_MAX_FILE_BYTES = 10_000_000
_MAX_LINE_CHARS = 4_000


class ReadFileTool(WorkspaceTool, Tool):
    """Return a numbered page of text from a workspace file."""

    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read text from a file in the workspace, returning numbered lines."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=True,
        is_parallel_safe=True,
        skip_permission_review=True,
        result_size_hint="paged",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace root.",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "First line to return, starting at 1.",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum number of lines to return.",
                "default": 200,
            },
            "encoding": {
                "type": "string",
                "description": (
                    "Optional Python text codec name. When omitted, encoding "
                    "is detected automatically."
                ),
            },
            "column_offset": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Zero-based character offset within the selected line. "
                    "Only valid when limit is 1."
                ),
                "default": 0,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {
            "path",
            "offset",
            "limit",
            "encoding",
            "column_offset",
        }
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        path = kwargs.get("path")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit", 200)
        encoding = kwargs.get("encoding")
        column_offset = kwargs.get("column_offset", 0)
        if not isinstance(path, str) or not path.strip():
            return "Error: path must be a non-empty string"
        if not isinstance(offset, int) or offset < 1:
            return "Error: offset must be an integer greater than or equal to 1"
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            return "Error: limit must be an integer between 1 and 1000"
        if (
            not isinstance(column_offset, int)
            or isinstance(column_offset, bool)
            or column_offset < 0
        ):
            return "Error: column_offset must be a non-negative integer"
        if "column_offset" in kwargs and limit != 1:
            return "Error: column_offset is only supported when limit is 1"
        if encoding is not None and (
            not isinstance(encoding, str) or not encoding.strip()
        ):
            return "Error: encoding must be a non-empty string when provided"
        try:
            normalized_encoding = (
                normalize_text_encoding(encoding)
                if isinstance(encoding, str)
                else None
            )
        except ValueError as exc:
            return f"Error: {exc}"

        relative_name = path
        try:
            target = self._resolve_path(path, allow_root=False)
            if not target.is_file():
                return f"Error: file not found: {path}"
            relative_name = self._relative_name(target)
            file_size = target.stat().st_size
            if file_size > _MAX_FILE_BYTES:
                return (
                    f"Error: file is larger than {_MAX_FILE_BYTES} bytes: "
                    f"{relative_name}"
                )
            text_file = read_text_file(target, encoding=normalized_encoding)
            page_with_marker = list(
                islice(
                    text_file.text.splitlines(keepends=True),
                    offset - 1,
                    offset + limit,
                )
            )
        except UnsupportedTextFileError as exc:
            return unsupported_text_file_error(relative_name, str(exc))
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

        has_more = len(page_with_marker) > limit
        page = page_with_marker[:limit]
        if not page:
            return f"{self._relative_name(target)}: no lines at offset {offset}"

        result: list[str] = []
        for line_number, line in enumerate(page, offset):
            content = line.rstrip("\r\n")
            if column_offset >= len(content) and column_offset > 0:
                return (
                    f"{self._relative_name(target)}: no characters at "
                    f"column_offset {column_offset} on line {line_number}; "
                    f"line has {len(content)} characters"
                )
            rendered = _render_line(
                line_number,
                content,
                column_offset=column_offset,
            )
            result.append(rendered)
            if (
                limit != 1
                and column_offset + _MAX_LINE_CHARS < len(content)
            ):
                result.append(
                    f"... (line {line_number} exceeded the "
                    f"{_MAX_LINE_CHARS}-character limit; the rest of this "
                    "line and all subsequent lines were omitted. Re-read "
                    f"this line with offset={line_number}, limit=1.)"
                )
                return "\n".join(result)
        if has_more:
            result.append("... (more lines available)")
        return "\n".join(result)


def _render_line(
    line_number: int,
    content: str,
    *,
    column_offset: int,
) -> str:
    end = min(len(content), column_offset + _MAX_LINE_CHARS)
    visible = content[column_offset:end]
    rendered = f"{line_number}: {visible}"
    if column_offset > 0 or end < len(content):
        shown_start = column_offset + 1 if visible else column_offset
        rendered += (
            " ... "
            f"(line window; characters {shown_start}-{end} of {len(content)}"
        )
        if end < len(content):
            rendered += (
                f"; continue with offset={line_number}, limit=1, "
                f"column_offset={end}"
            )
        rendered += ")"
    return rendered
