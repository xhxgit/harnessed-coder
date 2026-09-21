"""Edit an existing workspace file by replacing one exact text occurrence."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata
from ._file_content import (
    UnsupportedTextFileError,
    normalize_text_encoding,
    read_text_file,
    unsupported_text_file_error,
)


class EditFileTool(WorkspaceTool, Tool):
    """Replace one uniquely matching text segment in an existing file."""

    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace exactly one occurrence of text in an existing workspace file."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=True,
        result_size_hint="diff",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace root.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find. It must occur exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "Optional 1-based inclusive line where matching starts. "
                    "When set, old_text only needs to be unique inside the selected range."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "Optional 1-based inclusive line where matching ends. "
                    "When set, old_text only needs to be unique inside the selected range."
                ),
            },
            "encoding": {
                "type": "string",
                "description": (
                    "Optional Python text codec name. When omitted, encoding "
                    "is detected automatically and preserved."
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {
            "path",
            "old_text",
            "new_text",
            "start_line",
            "end_line",
            "encoding",
        }
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"

        path = kwargs.get("path")
        old_text = kwargs.get("old_text")
        new_text = kwargs.get("new_text")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        encoding = kwargs.get("encoding")
        if not isinstance(path, str) or not path.strip():
            return "Error: path must be a non-empty string"
        if not isinstance(old_text, str) or not old_text:
            return "Error: old_text must be a non-empty string"
        if not isinstance(new_text, str):
            return "Error: new_text must be a string"
        if old_text == new_text:
            return "Error: replacement would not change the file"
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
        line_range_error = _validate_line_range_args(start_line, end_line)
        if line_range_error is not None:
            return line_range_error

        relative_name = path
        try:
            target = self._resolve_path(path, allow_root=False)
            if not target.is_file():
                return f"Error: file not found: {path}"
            relative_name = self._relative_name(target)
            text_file = read_text_file(target, encoding=normalized_encoding)
            content = text_file.text
        except UnsupportedTextFileError as exc:
            return unsupported_text_file_error(relative_name, str(exc))
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

        try:
            prefix, searchable, suffix = _select_searchable_text(
                content,
                start_line=start_line,
                end_line=end_line,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        matched_old_text, matched_new_text, occurrences = _match_with_newline_fallback(
            searchable,
            old_text,
            new_text,
        )
        if occurrences == 0:
            return _not_found_message(start_line, end_line)
        if occurrences > 1:
            return _not_unique_message(occurrences, start_line, end_line)

        # The selected range is the only searchable/mutable segment. Prefix and
        # suffix are stitched back unchanged so range-limited edits cannot alter
        # matching text elsewhere in the file.
        updated_searchable = searchable.replace(matched_old_text, matched_new_text, 1)
        updated = f"{prefix}{updated_searchable}{suffix}"
        if updated == content:
            return "Error: replacement would not change the file"

        try:
            _atomic_write_text(
                target,
                updated,
                encoding=text_file.encoding,
                expected_current=text_file.raw_bytes,
            )
        except _ConcurrentModificationError as exc:
            return f"Error: {exc}"
        except (OSError, UnicodeError) as exc:
            return f"Error: {exc}"

        return f"Updated {self._relative_name(target)}.\n{_short_diff(content, updated)}"


class _ConcurrentModificationError(Exception):
    pass


def _validate_line_range_args(start_line: object, end_line: object) -> str | None:
    for name, value in (("start_line", start_line), ("end_line", end_line)):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return f"Error: {name} must be a positive integer"
    if isinstance(start_line, int) and isinstance(end_line, int) and start_line > end_line:
        return "Error: start_line must be less than or equal to end_line"
    return None


def _select_searchable_text(
    content: str,
    *,
    start_line: object,
    end_line: object,
) -> tuple[str, str, str]:
    if start_line is None and end_line is None:
        return "", content, ""

    lines = content.splitlines(keepends=True)
    if not lines:
        raise ValueError("line range is outside the file")

    start = start_line if isinstance(start_line, int) else 1
    end = end_line if isinstance(end_line, int) else len(lines)
    if start > len(lines) or end > len(lines):
        raise ValueError(f"line range is outside the file; file has {len(lines)} lines")

    start_offset = sum(len(line) for line in lines[: start - 1])
    end_offset = sum(len(line) for line in lines[:end])
    return content[:start_offset], content[start_offset:end_offset], content[end_offset:]


def _not_found_message(start_line: object, end_line: object) -> str:
    if start_line is None and end_line is None:
        return "Error: old_text was not found in the file"
    return "Error: old_text was not found in the selected line range"


def _not_unique_message(occurrences: int, start_line: object, end_line: object) -> str:
    if start_line is None and end_line is None:
        return f"Error: old_text occurs {occurrences} times; it must be unique"
    return (
        f"Error: old_text occurs {occurrences} times in the selected line range; "
        "it must be unique"
    )


def _match_with_newline_fallback(
    searchable: str,
    old_text: str,
    new_text: str,
) -> tuple[str, str, int]:
    """Match exact text first, then adapt LF-only model text to file newlines."""
    occurrences = searchable.count(old_text)
    if occurrences or "\n" not in old_text or "\r" in old_text:
        return old_text, new_text, occurrences

    for newline in ("\r\n", "\r"):
        candidate_old = old_text.replace("\n", newline)
        candidate_occurrences = searchable.count(candidate_old)
        if candidate_occurrences:
            candidate_new = (
                new_text.replace("\n", newline) if "\r" not in new_text else new_text
            )
            return candidate_old, candidate_new, candidate_occurrences
    return old_text, new_text, 0


def _atomic_write_text(
    target: Path,
    content: str,
    *,
    encoding: str,
    expected_current: bytes,
) -> None:
    temp_path: str | None = None
    try:
        encoded = content.encode(encoding)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(encoded)
        # Re-read immediately before replace so a concurrent editor or tool
        # cannot have its changes silently overwritten by this edit.
        if _read_current_bytes(target) != expected_current:
            raise _ConcurrentModificationError(
                "file changed since it was read; retry with fresh content"
            )
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _read_current_bytes(target: Path) -> bytes:
    return target.read_bytes()


def _short_diff(before: str, after: str, *, limit: int = 20) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if not diff_lines:
        return (
            "Changed lines:\n"
            f"content changed without line-level diff "
            f"(chars {len(before)} -> {len(after)})"
        )
    shown = diff_lines[:limit]
    if len(diff_lines) > limit:
        shown.append(f"... diff truncated after {limit} lines")
    return "Changed lines:\n" + "\n".join(_annotate_diff_line(line) for line in shown)


def _annotate_diff_line(line: str) -> str:
    original_length = len(line) - 1
    line = _truncate_diff_line(line)
    if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
        return f"{line} (len={original_length})"
    return line


_MAX_DIFF_LINE_CHARS = 2_000


def _truncate_diff_line(line: str) -> str:
    if line.startswith(("+++", "---", "@@")) or len(line) <= _MAX_DIFF_LINE_CHARS + 1:
        return line
    prefix = line[0] if line.startswith(("+", "-", " ")) else ""
    content = line[1:] if prefix else line
    if len(content) <= _MAX_DIFF_LINE_CHARS:
        return line
    half = _MAX_DIFF_LINE_CHARS // 2
    omitted = len(content) - (half * 2)
    return (
        f"{prefix}{content[:half]}"
        f"<... {omitted} characters omitted ...>"
        f"{content[-half:]}"
    )
