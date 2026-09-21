"""Write text to a workspace file."""

from __future__ import annotations

from typing import Any, ClassVar

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata
from ._file_content import (
    UnsupportedTextFileError,
    normalize_text_encoding,
    read_non_binary_file_bytes,
    read_text_file,
    unsupported_text_file_error,
)


class WriteFileTool(WorkspaceTool, Tool):
    """Create or overwrite a workspace file with text content."""

    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create or overwrite a text file in the workspace."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=True,
        result_size_hint="small",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace root.",
            },
            "content": {
                "type": "string",
                "description": "Complete text content to write to the file.",
            },
            "encoding": {
                "type": "string",
                "description": (
                    "Optional Python text codec name. New files otherwise use "
                    "UTF-8; existing files otherwise preserve detected encoding."
                ),
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"path", "content", "encoding"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"

        path = kwargs.get("path")
        content = kwargs.get("content")
        encoding = kwargs.get("encoding")
        if not isinstance(path, str) or not path.strip():
            return "Error: path must be a non-empty string"
        if not isinstance(content, str):
            return "Error: content must be a string"
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
        created_parent_names: list[str] = []
        try:
            target = self._resolve_path(path, allow_root=False)
            relative_name = self._relative_name(target)
            if target.exists() and not target.is_file():
                return f"Error: path is not a file: {path}"
            candidate = target.parent
            while candidate != self.root and not candidate.exists():
                created_parent_names.append(self._relative_name(candidate))
                candidate = candidate.parent
            created_parent_names.reverse()
            target.parent.mkdir(parents=True, exist_ok=True)
            output_encoding = normalized_encoding or "utf-8"
            if target.is_file():
                if normalized_encoding is None:
                    existing = read_text_file(target)
                    output_encoding = existing.encoding
                else:
                    read_non_binary_file_bytes(target)
            encoded = content.encode(output_encoding)
            target.write_bytes(encoded)
        except UnsupportedTextFileError as exc:
            return unsupported_text_file_error(relative_name, str(exc))
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

        result = f"Wrote {self._relative_name(target)}."
        if created_parent_names:
            result += f" Created parent directories: {', '.join(created_parent_names)}."
        return result
