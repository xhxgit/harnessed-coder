"""Shared helpers for reading saved harnessed-coder session history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from harnessed_coder.constants import WORKSPACE_METADATA_FILE_NAME
from harnessed_coder.session import (
    SESSION_VERSION,
    conversation_turns as session_turns,
    resolve_data_dir,
)
from harnessed_coder.context.tool_batch_summary import (
    ToolBatchSummary,
    tool_batch_source_fingerprint,
)

from .transcript import (
    render_turn,
    search_snippet,
    visible_message_text,
)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_DEFAULT_TURN_COUNT = 5
_MAX_TURN_COUNT = 20
_TRANSCRIPT_MAX_CHARS = 32_000
_DEFAULT_TOOL_RESULT_CHAR_COUNT = 20_000
_MAX_TOOL_RESULT_CHAR_COUNT = 30_000


class SessionHistoryReader:
    """Read saved workspace/session data from the configured user data dir."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = resolve_data_dir(data_dir)
        self.workspaces_dir = self.data_dir / "workspaces"

    def list_workspaces(self, *, limit: Any = _DEFAULT_LIMIT, offset: Any = 0) -> str:
        normalized_limit = normalize_limit(limit)
        normalized_offset = normalize_offset(offset)
        workspaces = self._workspace_dirs()
        page = workspaces[normalized_offset : normalized_offset + normalized_limit]
        if not page:
            return (
                f"No workspaces found at offset {normalized_offset}. "
                f"Total workspaces: {len(workspaces)}."
            )

        lines = pagination_prefix(normalized_offset, len(workspaces))
        for workspace_dir in page:
            metadata = read_json_object(workspace_dir / WORKSPACE_METADATA_FILE_NAME)
            session_count = len(session_files(workspace_dir))
            root = metadata.get("workspace_root") or "(unknown root)"
            name = metadata.get("workspace_name") or ""
            name_part = f" name={name}" if name else ""
            lines.append(
                f"{workspace_dir.name}{name_part} root={root} sessions={session_count}"
            )
        lines.extend(pagination_suffix(normalized_offset, normalized_limit, len(workspaces)))
        return "\n".join(lines)

    def list_sessions(
        self,
        *,
        workspace_id: str | None = None,
        limit: Any = _DEFAULT_LIMIT,
        offset: Any = 0,
    ) -> str:
        normalized_limit = normalize_limit(limit)
        normalized_offset = normalize_offset(offset)
        entries: list[tuple[str, Path]] = []
        workspace_dirs = (
            [self.resolve_workspace_dir(workspace_id)]
            if workspace_id
            else self._workspace_dirs()
        )
        for workspace_dir in workspace_dirs:
            for session_file in session_files(workspace_dir):
                entries.append((workspace_dir.name, session_file))

        entries.sort(key=lambda item: (item[0], item[1].stem))
        page = entries[normalized_offset : normalized_offset + normalized_limit]
        if not page:
            return (
                f"No sessions found at offset {normalized_offset}. "
                f"Total sessions: {len(entries)}."
            )

        lines = pagination_prefix(normalized_offset, len(entries))
        for entry_workspace_id, session_file in page:
            data = read_json_object(session_file)
            messages = messages_from_data(data)
            turns = session_turns(messages)
            modified = session_file.stat().st_mtime
            lines.append(
                f"{entry_workspace_id}/{session_file.stem} "
                f"turns={len(turns)} messages={len(messages)} "
                f"modified_epoch={modified:.0f}"
            )
        lines.extend(pagination_suffix(normalized_offset, normalized_limit, len(entries)))
        return "\n".join(lines)

    def read_session(
        self,
        *,
        workspace_id: str,
        session: str,
        start_turn: Any = 1,
        turn_count: Any = _DEFAULT_TURN_COUNT,
        include_tool_batch_summary: Any = False,
    ) -> str:
        normalized_start = normalize_start_turn(start_turn)
        normalized_count = normalize_turn_count(turn_count)
        if not isinstance(include_tool_batch_summary, bool):
            raise ValueError("include_tool_batch_summary must be a boolean")
        session_file = self.resolve_session_file(workspace_id, session)
        data = read_json_object(session_file)
        messages = messages_from_data(data)
        turns = session_turns(messages)
        latest_summaries = (
            latest_tool_batch_summaries(data, messages)
            if include_tool_batch_summary
            else {}
        )
        first_index = normalized_start - 1
        candidates = turns[first_index : first_index + normalized_count]
        if not candidates:
            return (
                f"No turns found at start_turn {normalized_start}. "
                f"Total turns: {len(turns)}."
            )

        lines = [
            f"Session: {workspace_id}/{session_file.stem}",
            f"Turns: {len(turns)}",
            f"Messages: {len(messages)}",
        ]
        if normalized_start > 1:
            lines.append(f"... ({min(first_index, len(turns))} earlier turns)")

        rendered_count = 0
        for turn in candidates:
            checkpoint = latest_summaries.get(turn.number)
            rendered = render_turn(
                turn,
                tool_batch_summary=(checkpoint.summary if checkpoint is not None else None),
                summary_through_batch=(
                    checkpoint.batch_index if checkpoint is not None else None
                ),
            )
            candidate_text = "\n\n".join([*lines, rendered])
            if rendered_count and len(candidate_text) > _TRANSCRIPT_MAX_CHARS:
                break
            lines.append(rendered)
            rendered_count += 1

        next_turn = normalized_start + rendered_count
        if next_turn <= len(turns):
            lines.append(
                f"... ({len(turns) - next_turn + 1} more turns; "
                f"next start_turn={next_turn})"
            )
        return "\n".join(lines)

    def read_tool_result(
        self,
        *,
        tool_call_id: str,
        char_offset: Any = 0,
        char_count: Any = _DEFAULT_TOOL_RESULT_CHAR_COUNT,
    ) -> str:
        normalized_offset = normalize_offset(char_offset, name="char_offset")
        normalized_count = normalize_char_count(char_count)
        matching: list[tuple[str, Path, dict[str, Any]]] = []
        for workspace_dir in self._workspace_dirs():
            for session_file in session_files(workspace_dir):
                data = read_json_object(session_file)
                for message in messages_from_data(data):
                    if (
                        message.get("role") == "tool"
                        and message.get("tool_call_id") == tool_call_id
                    ):
                        matching.append((workspace_dir.name, session_file, message))
        if not matching:
            raise ValueError(f"tool result not found for tool_call_id: {tool_call_id}")
        if len(matching) > 1:
            locations = ", ".join(
                f"{workspace_id}/{session_file.stem}"
                for workspace_id, session_file, _ in matching
            )
            raise ValueError(
                f"duplicate tool results for tool_call_id {tool_call_id}: {locations}"
            )
        workspace_id, session_file, message = matching[0]
        content = visible_message_text(message)
        total = len(content)
        if normalized_offset >= total and total:
            return (
                f"No characters found at char_offset {normalized_offset}. "
                f"Total characters: {total}."
            )
        page = content[normalized_offset : normalized_offset + normalized_count]
        end = normalized_offset + len(page)
        lines = [
            f"Session: {workspace_id}/{session_file.stem}",
            f"Tool call id: {tool_call_id}",
            f"Characters: {normalized_offset}-{max(normalized_offset, end - 1)} of {total}",
            page,
        ]
        if end < total:
            lines.append(f"... ({total - end} more characters; next char_offset={end})")
        return "\n".join(lines)

    def search_sessions(
        self,
        *,
        query: str,
        workspace_id: str | None = None,
        session: str | None = None,
        limit: Any = _DEFAULT_LIMIT,
        offset: Any = 0,
    ) -> str:
        normalized_limit = normalize_limit(limit)
        normalized_offset = normalize_offset(offset)
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        matches: list[str] = []
        # Restricting by both workspace and session resolves exactly one file;
        # looser searches walk saved sessions but still stay under data_dir via
        # resolve_workspace_dir/session_files.
        if workspace_id is not None and session is not None:
            searched_files = [(workspace_id, self.resolve_session_file(workspace_id, session))]
        else:
            workspace_dirs = (
                [self.resolve_workspace_dir(workspace_id)]
                if workspace_id
                else self._workspace_dirs()
            )
            searched_files = [
                (workspace_dir.name, session_file)
                for workspace_dir in workspace_dirs
                for session_file in session_files(workspace_dir)
                if session is None or session_file.stem == session
            ]

        for entry_workspace_id, session_file in searched_files:
            data = read_json_object(session_file)
            messages = messages_from_data(data)
            turns = session_turns(messages)
            latest_summaries = latest_tool_batch_summaries(data, messages)
            for turn in turns:
                for message in turn.messages:
                    role = str(message.get("role", "unknown"))
                    if role not in {"user", "assistant"}:
                        continue
                    text = visible_message_text(message)
                    match = pattern.search(text)
                    if match is None:
                        continue
                    matches.append(
                        f"{entry_workspace_id}/{session_file.stem} turn={turn.number} "
                        f"{role}: {search_snippet(text, match.start())}"
                    )
                checkpoint = latest_summaries.get(turn.number)
                if checkpoint is None:
                    continue
                match = pattern.search(checkpoint.summary)
                if match is not None:
                    matches.append(
                        f"{entry_workspace_id}/{session_file.stem} turn={turn.number} "
                        "tool_batch_summary "
                        f"through_batch={checkpoint.batch_index}: "
                        f"{search_snippet(checkpoint.summary, match.start())}"
                    )

        page = matches[normalized_offset : normalized_offset + normalized_limit]
        if not page:
            return (
                f"No matches found at offset {normalized_offset}. "
                f"Total matches: {len(matches)}."
            )
        lines = pagination_prefix(normalized_offset, len(matches))
        lines.extend(page)
        lines.extend(pagination_suffix(normalized_offset, normalized_limit, len(matches)))
        return "\n".join(lines)

    def _workspace_dirs(self) -> list[Path]:
        if not self.workspaces_dir.exists():
            return []
        if not self.workspaces_dir.is_dir():
            raise ValueError(f"workspaces path is not a directory: {self.workspaces_dir}")
        return sorted(path for path in self.workspaces_dir.iterdir() if path.is_dir())

    def resolve_workspace_dir(self, workspace_id: str) -> Path:
        workspace_dir = resolve_data_dir_child(self.workspaces_dir, workspace_id)
        if not workspace_dir.is_dir():
            raise ValueError(f"workspace not found: {workspace_id}")
        return workspace_dir

    def resolve_session_file(self, workspace_id: str, session: str) -> Path:
        workspace_dir = self.resolve_workspace_dir(workspace_id)
        session_name = session if session.endswith(".json") else f"{session}.json"
        session_file = resolve_data_dir_child(workspace_dir / "sessions", session_name)
        if not session_file.is_file():
            raise ValueError(f"session not found: {workspace_id}/{session}")
        return session_file


def run_history_read(operation: str, callback: Any) -> str:
    try:
        return callback()
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON session data: {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error: {exc}"


def normalize_limit(limit: Any) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < 1 or limit > _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    return limit


def normalize_offset(offset: Any, *, name: str = "offset") -> int:
    if not isinstance(offset, int):
        raise ValueError(f"{name} must be an integer")
    if offset < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return offset


def normalize_start_turn(start_turn: Any) -> int:
    if not isinstance(start_turn, int):
        raise ValueError("start_turn must be an integer")
    if start_turn < 1:
        raise ValueError("start_turn must be at least 1")
    return start_turn


def normalize_turn_count(turn_count: Any) -> int:
    if not isinstance(turn_count, int):
        raise ValueError("turn_count must be an integer")
    if turn_count < 1 or turn_count > _MAX_TURN_COUNT:
        raise ValueError(f"turn_count must be between 1 and {_MAX_TURN_COUNT}")
    return turn_count


def normalize_char_count(char_count: Any) -> int:
    if not isinstance(char_count, int):
        raise ValueError("char_count must be an integer")
    if char_count < 1 or char_count > _MAX_TOOL_RESULT_CHAR_COUNT:
        raise ValueError(
            f"char_count must be between 1 and {_MAX_TOOL_RESULT_CHAR_COUNT}"
        )
    return char_count


def resolve_data_dir_child(parent: Path, name: str) -> Path:
    if not name or Path(name).is_absolute():
        raise ValueError("path segment must be a relative name")
    target = (parent / name).resolve()
    parent_resolved = parent.resolve()
    if not target.is_relative_to(parent_resolved):
        raise ValueError("path segment must stay within the data directory")
    return target


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def messages_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("version") != SESSION_VERSION:
        raise ValueError(f"unsupported session version: {data.get('version')}")
    messages = data.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("session messages must be a list of objects")
    return messages


@dataclass(frozen=True)
class _TurnToolBatchSummaries:
    batch_index: int
    summary: str


def latest_tool_batch_summaries(
    data: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[int, _TurnToolBatchSummaries]:
    """Return source-valid independent batch summaries grouped by turn."""
    raw_summaries = data.get("tool_batch_summaries", [])
    if not isinstance(raw_summaries, list):
        raise ValueError("session tool_batch_summaries must be a list of objects")
    records = [
        record
        for value in raw_summaries
        if (record := ToolBatchSummary.from_dict(value)) is not None
    ]
    by_source = {
        (record.source_start, record.source_end, record.source_fingerprint): record
        for record in records
    }
    result: dict[int, _TurnToolBatchSummaries] = {}
    for turn in session_turns(messages):
        summaries: list[tuple[int, str]] = []
        batch_index = 0
        index = turn.start_index
        while index < turn.end_index:
            message = messages[index]
            tool_calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
                index += 1
                continue
            batch_index += 1
            source_end = index + 1 + len(tool_calls)
            if source_end > turn.end_index:
                break
            source = messages[index:source_end]
            fingerprint = tool_batch_source_fingerprint(source)
            record = by_source.get((index, source_end, fingerprint))
            if record is not None:
                summaries.append((batch_index, record.summary))
            index = source_end
        if summaries:
            result[turn.number] = _TurnToolBatchSummaries(
                batch_index=max(batch for batch, _ in summaries),
                summary="\n".join(
                    f"Batch {batch}: {summary}"
                    for batch, summary in summaries
                ),
            )
    return result


def session_files(workspace_dir: Path) -> list[Path]:
    sessions_dir = workspace_dir / "sessions"
    if not sessions_dir.exists():
        return []
    if not sessions_dir.is_dir():
        raise ValueError(f"sessions path is not a directory: {sessions_dir}")
    return sorted(path for path in sessions_dir.glob("*.json") if path.is_file())


def pagination_prefix(offset: int, total: int) -> list[str]:
    if offset <= 0:
        return []
    return [f"... ({min(offset, total)} earlier items)"]


def pagination_suffix(offset: int, limit: int, total: int) -> list[str]:
    remaining = total - offset - limit
    if remaining <= 0:
        return []
    return [f"... ({remaining} more items)"]
