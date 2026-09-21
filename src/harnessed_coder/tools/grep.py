"""Search text in workspace files tool."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, ClassVar, Iterator

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata
from ._file_content import (
    UnsupportedTextFileError,
    normalize_text_encodings,
    read_text_file,
    read_text_file_with_encodings,
)


_IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__"}
_MAX_SEARCH_FILE_BYTES = 1_000_000
_MAX_MATCH_LINE_CHARS = 2_000


class GrepTool(WorkspaceTool, Tool):
    """Search workspace text files with a regular expression."""

    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search text files in the workspace with a regular expression and return matches."
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
            "pattern": {
                "type": "string",
                "description": "Python regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search, relative to the workspace.",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Optional glob for included files, such as *.py.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether matching is case-sensitive.",
                "default": True,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum number of matching lines to return.",
                "default": 100,
            },
            "encodings": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
                "uniqueItems": True,
                "description": (
                    "Optional ordered text codec fallbacks applied to every "
                    "candidate file. When omitted, encoding is auto-detected."
                ),
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {
            "pattern",
            "path",
            "include",
            "case_sensitive",
            "limit",
            "encodings",
        }
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        pattern = kwargs.get("pattern")
        path = kwargs.get("path", ".")
        include = kwargs.get("include")
        case_sensitive = kwargs.get("case_sensitive", True)
        limit = kwargs.get("limit", 100)
        encodings = kwargs.get("encodings")
        if not isinstance(pattern, str) or not pattern:
            return "Error: pattern must be a non-empty string"
        if not isinstance(path, str) or not path.strip():
            return "Error: path must be a non-empty string"
        if not isinstance(case_sensitive, bool):
            return "Error: case_sensitive must be a boolean"
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            return "Error: limit must be an integer between 1 and 1000"
        if include is not None and (not isinstance(include, str) or not include):
            return "Error: include must be a non-empty string when provided"
        if encodings is not None:
            if (
                not isinstance(encodings, list)
                or not 1 <= len(encodings) <= 8
                or any(
                    not isinstance(encoding, str) or not encoding.strip()
                    for encoding in encodings
                )
            ):
                return (
                    "Error: encodings must be an array of 1 to 8 "
                    "non-empty strings"
                )
            try:
                normalized_encodings = normalize_text_encodings(encodings)
            except ValueError as exc:
                return f"Error: {exc}"
        else:
            normalized_encodings = None

        try:
            expression = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            target = self._resolve_path(path)
            files = self._files_to_search(target, include)
        except (OSError, ValueError, re.error) as exc:
            return f"Error: {exc}"

        matches: list[str] = []
        skipped_non_text = 0
        skipped_large = 0
        try:
            # Filter files before decoding. This keeps search predictable on
            # large repositories and avoids dragging binary blobs into the
            # model-facing result.
            for file_path in files:
                file_size = file_path.stat().st_size
                if file_size > _MAX_SEARCH_FILE_BYTES:
                    skipped_large += 1
                    continue
                try:
                    text_file = (
                        read_text_file(file_path)
                        if normalized_encodings is None
                        else read_text_file_with_encodings(
                            file_path,
                            normalized_encodings,
                        )
                    )
                    lines = text_file.text.splitlines()
                except UnsupportedTextFileError:
                    skipped_non_text += 1
                    continue
                for line_number, line in enumerate(lines, 1):
                    match = expression.search(line)
                    if match is not None:
                        rendered = _render_match(
                            self._relative_name(file_path),
                            line_number,
                            line,
                            match,
                        )
                        matches.append(rendered)
                        if len(matches) >= limit:
                            matches.append("... (result limit reached)")
                            matches.extend(_skipped_summary(skipped_non_text, skipped_large))
                            return "\n".join(matches)
        except (OSError, ValueError) as exc:
            return f"Error: {exc}"

        summary = _skipped_summary(skipped_non_text, skipped_large)
        if not matches:
            return "\n".join(["No matches found.", *summary])
        matches.extend(summary)
        return "\n".join(matches)

    def _files_to_search(self, target: Path, include: str | None) -> Iterator[Path]:
        if not target.exists():
            raise ValueError(f"path not found: {self._relative_name(target)}")
        if target.is_file():
            if include is None or target.match(include):
                yield target
            return

        for candidate in sorted(target.rglob("*")):
            if any(part in _IGNORED_DIRECTORIES for part in candidate.parts):
                continue
            if candidate.is_file() and (include is None or candidate.match(include)):
                yield candidate


def _skipped_summary(skipped_non_text: int, skipped_large: int) -> list[str]:
    summary = []
    if skipped_non_text:
        summary.append(
            f"... (skipped {skipped_non_text} binary or undecodable files)"
        )
    if skipped_large:
        summary.append(
            f"... (skipped {skipped_large} files larger than {_format_bytes(_MAX_SEARCH_FILE_BYTES)})"
        )
    return summary


def _format_bytes(size: int) -> str:
    return f"{size} bytes"


def _render_match(
    relative_name: str,
    line_number: int,
    line: str,
    match: re.Match[str],
) -> str:
    if len(line) <= _MAX_MATCH_LINE_CHARS:
        return f"{relative_name}:{line_number}: {line}"

    match_length = match.end() - match.start()
    if match_length >= _MAX_MATCH_LINE_CHARS:
        start = match.start()
    else:
        surrounding = _MAX_MATCH_LINE_CHARS - match_length
        start = max(0, match.start() - surrounding // 2)
        start = min(start, len(line) - _MAX_MATCH_LINE_CHARS)
    end = min(len(line), start + _MAX_MATCH_LINE_CHARS)
    excerpt = line[start:end]
    if start > 0:
        excerpt = f"...{excerpt}"
    if end < len(line):
        excerpt = f"{excerpt}..."
    return (
        f"{relative_name}:{line_number}:{start + 1}-{end}: {excerpt} "
        f"(line truncated; total {len(line)} characters)"
    )
