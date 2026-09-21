"""In-memory conversation state with atomic session-file persistence."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..context.model_context import ModelContextCheckpoint


logger = logging.getLogger(__name__)

SESSION_VERSION = 1
MODEL_CONTEXT_CHECKPOINTS_FIELD = "model_context_checkpoints"
TOOL_BATCH_SUMMARIES_FIELD = "tool_batch_summaries"
USED_TOOL_NAMES_FIELD = "used_tool_names"
_REPLACE_RETRY_DELAYS_SECONDS = (0.01, 0.025, 0.05, 0.1, 0.2)


class ConversationSession:
    """Own one conversation's canonical messages and rolling checkpoints."""

    def __init__(
        self,
        messages: Iterable[dict[str, Any]] = (),
        *,
        path: str | Path | None = None,
        checkpoint_data: Iterable[dict[str, Any]] = (),
        tool_batch_summary_data: Iterable[dict[str, Any]] = (),
        usage_data: Iterable[dict[str, Any]] = (),
        used_tool_names: Iterable[str] = (),
    ) -> None:
        self._lock = RLock()
        self._usage = _copy_objects(usage_data, name="usage")
        for row in self._usage:
            if not all(isinstance(row.get(key), str) and row[key] for key in ("model", "purpose")):
                raise ValueError("invalid session usage model/purpose")
            row.setdefault("cached_prompt_tokens", 0)
            for key in (
                "requests",
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "unreported",
                "failed",
            ):
                if type(row.get(key)) is not int or row[key] < 0:
                    raise ValueError(f"invalid session usage {key}")
        self.path = None if path is None else Path(path).resolve()
        self._messages = _copy_objects(messages, name="messages")
        self._checkpoint_data = _copy_objects(
            checkpoint_data,
            name="checkpoints",
        )
        self._tool_batch_summary_data = _copy_objects(
            tool_batch_summary_data,
            name="tool batch summaries",
        )
        self._used_tool_names: list[str] = []
        for name in used_tool_names:
            if not isinstance(name, str) or not name:
                raise ValueError("invalid session used tool name")
            if name not in self._used_tool_names:
                self._used_tool_names.append(name)
        self._clear_generation = 0

    @classmethod
    def open(cls, path: str | Path) -> ConversationSession:
        """Load a file-backed session, or create empty state if it is absent."""
        resolved_path = Path(path).resolve()
        if not resolved_path.exists():
            return cls(path=resolved_path)
        with resolved_path.open("r", encoding="utf-8") as session_file:
            data = json.load(session_file)
        if not isinstance(data, dict):
            raise ValueError("session file must contain a JSON object")
        if data.get("version") != SESSION_VERSION:
            raise ValueError(f"unsupported session version: {data.get('version')}")
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("session messages must be a list of objects")
        checkpoint_data = data.get(MODEL_CONTEXT_CHECKPOINTS_FIELD, [])
        if not isinstance(checkpoint_data, list):
            raise ValueError("model context checkpoints must be a list of objects")
        tool_batch_summary_data = data.get(TOOL_BATCH_SUMMARIES_FIELD, [])
        if not isinstance(tool_batch_summary_data, list):
            raise ValueError("tool batch summaries must be a list of objects")
        used_tool_names = data.get(USED_TOOL_NAMES_FIELD, [])
        if not isinstance(used_tool_names, list):
            raise ValueError("session used_tool_names must be a list of strings")
        return cls(
            messages,
            path=resolved_path,
            checkpoint_data=checkpoint_data,
            tool_batch_summary_data=tool_batch_summary_data,
            usage_data=data.get("usage", []),
            used_tool_names=used_tool_names,
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        messages: Iterable[dict[str, Any]] = (),
    ) -> ConversationSession:
        """Create and persist a new file-backed session."""
        session = cls(messages, path=path)
        session._write_document()
        return session

    def record_usage(
        self,
        *,
        model: str,
        purpose: str,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        unreported: bool,
        failed: bool,
    ) -> None:
        with self._lock:
            row = next((r for r in self._usage if r["model"] == model and r["purpose"] == purpose), None)
            if row is None:
                row = {
                    "model": model,
                    "purpose": purpose,
                    "requests": 0,
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 0,
                    "unreported": 0,
                    "failed": 0,
                }
                self._usage.append(row)
            for key, value in dict(
                requests=1,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                completion_tokens=completion_tokens,
                unreported=int(unreported),
                failed=int(failed),
            ).items():
                row[key] += value
            self._persist_safely()

    def usage_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._usage)

    def record_used_tool(self, name: str) -> None:
        """Persist a deferred tool after its first dispatched call."""
        if not isinstance(name, str) or not name:
            raise ValueError("used tool name must be a non-empty string")
        with self._lock:
            if name in self._used_tool_names:
                return
            self._used_tool_names.append(name)
            self._persist_safely()

    def used_tool_names(self) -> tuple[str, ...]:
        """Return deferred tools used by this session in first-use order."""
        with self._lock:
            return tuple(self._used_tool_names)

    def append_message(self, message: dict[str, Any]) -> None:
        """Append one detached canonical message and persist the session."""
        self.append_messages([message])

    def append_messages(self, messages: Iterable[dict[str, Any]]) -> None:
        """Append detached canonical messages with one persistence write."""
        copied = _copy_objects(messages, name="messages")
        if not copied:
            return
        with self._lock:
            self._messages.extend(copied)
            self._persist_safely()

    def clear(self) -> None:
        """Clear conversation state while retaining incurred API usage."""
        with self._lock:
            self._messages.clear()
            self._checkpoint_data.clear()
            self._tool_batch_summary_data.clear()
            self._clear_generation += 1
            self._persist_safely()

    def move_to(self, path: str | Path) -> None:
        """Move this session's persistence target and keep its in-memory state."""
        with self._lock:
            target = Path(path).resolve()
            if self.path == target:
                return
            if target.exists():
                raise ValueError(f"session file already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.path is not None and self.path.exists():
                self.path.replace(target)
                self.path = target
                return
            self.path = target
            self._persist_safely()

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a detached copy of all canonical messages."""
        with self._lock:
            return deepcopy(self._messages)

    def messages_since(self, index: int) -> list[dict[str, Any]]:
        """Return detached messages appended at or after a zero-based index."""
        with self._lock:
            if index < 0 or index > len(self._messages):
                raise ValueError("message index is outside the current session")
            return deepcopy(self._messages[index:])

    def message_state(self) -> tuple[int, int]:
        """Return clear generation and current message count for incremental readers."""
        with self._lock:
            return self._clear_generation, len(self._messages)

    def clear_generation(self) -> int:
        with self._lock:
            return self._clear_generation

    def projection_snapshot(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Atomically snapshot canonical messages and tool-batch summaries."""
        with self._lock:
            return (
                deepcopy(self._messages),
                deepcopy(self._tool_batch_summary_data),
                self._clear_generation,
            )

    def last_message(self) -> dict[str, Any] | None:
        """Return a detached copy of the latest canonical message."""
        with self._lock:
            if not self._messages:
                return None
            return deepcopy(self._messages[-1])

    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    def checkpoint_count(self) -> int:
        with self._lock:
            return len(self._checkpoint_data)

    def tool_batch_summary_count(self) -> int:
        with self._lock:
            return len(self._tool_batch_summary_data)

    def latest_checkpoint(self) -> ModelContextCheckpoint | None:
        """Return the latest rolling checkpoint, if one exists."""
        with self._lock:
            if not self._checkpoint_data:
                return None
            from ..context.model_context import ModelContextCheckpoint

            return ModelContextCheckpoint.from_dict(deepcopy(self._checkpoint_data[-1]))

    def append_checkpoint(self, checkpoint: ModelContextCheckpoint) -> None:
        """Append a detached rolling checkpoint and persist the session."""
        with self._lock:
            self._checkpoint_data.append(deepcopy(checkpoint.to_dict()))
            self._persist_safely()

    def append_tool_batch_summary(
        self,
        summary: Any,
        *,
        expected_clear_generation: int,
    ) -> bool:
        """Commit a completed tool-batch summary if its source is still current."""
        from ..context.tool_batch_summary import tool_batch_source_fingerprint

        with self._lock:
            if self._clear_generation != expected_clear_generation:
                return False
            if summary.source_end > len(self._messages):
                return False
            source = self._messages[summary.source_start : summary.source_end]
            if tool_batch_source_fingerprint(source) != summary.source_fingerprint:
                return False
            serialized = deepcopy(summary.to_dict())
            if serialized in self._tool_batch_summary_data:
                return True
            self._tool_batch_summary_data.append(serialized)
            try:
                self._write_document()
            except Exception:
                self._tool_batch_summary_data.pop()
                raise
            return True

    def _persist_safely(self) -> None:
        with self._lock:
            if self.path is None:
                return
            try:
                self._write_document()
            except Exception:
                logger.exception("Failed to persist conversation session")

    def _write_document(self) -> None:
        with self._lock:
            if self.path is None:
                return
            data: dict[str, Any] = {
                "version": SESSION_VERSION,
                "messages": self._messages,
            }
            if self._usage:
                data["usage"] = self._usage
            if self._checkpoint_data:
                data[MODEL_CONTEXT_CHECKPOINTS_FIELD] = self._checkpoint_data
            if self._tool_batch_summary_data:
                data[TOOL_BATCH_SUMMARIES_FIELD] = self._tool_batch_summary_data
            if self._used_tool_names:
                data[USED_TOOL_NAMES_FIELD] = self._used_tool_names
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(data, temp_file, ensure_ascii=False, indent=2)
                temp_file.write("\n")
            try:
                _replace_file_with_retry(temp_path, self.path)
            finally:
                temp_path.unlink(missing_ok=True)


def _replace_file_with_retry(source: Path, target: Path) -> None:
    """Atomically replace a file, tolerating brief Windows file-handle races."""
    for delay in _REPLACE_RETRY_DELAYS_SECONDS:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(source, target)


def _copy_objects(
    values: Iterable[dict[str, Any]],
    *,
    name: str,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError(f"session {name} must contain only dictionaries")
        copied.append(deepcopy(value))
    return copied
