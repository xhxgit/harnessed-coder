"""JSON persistence for manually managed long-term memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self
from uuid import uuid4

from ..constants import MEMORY_DIR_NAME, MEMORY_FILE_NAME, USER_MEMORY_FILE_NAME
from .types import MemoryRecord, MemoryScope


MEMORY_VERSION = 2
_MANUAL_MEMORY_RATIONALE = "Added manually by the user."


@dataclass(frozen=True)
class MemoryAddResult:
    """Result of adding a memory, including duplicate detection."""

    record: MemoryRecord
    created: bool


class MemoryStore:
    """Store one scope of long-term memory in a versioned JSON file."""

    def __init__(self, path: str | Path, *, scope: MemoryScope) -> None:
        self.path = Path(path).resolve()
        self.scope: MemoryScope = scope

    @classmethod
    def for_user(cls, data_dir: str | Path) -> Self:
        """Create a store for user-scoped memory in the application data dir."""
        path = (
            Path(data_dir).expanduser().resolve()
            / MEMORY_DIR_NAME
            / USER_MEMORY_FILE_NAME
        )
        return cls(path, scope="user")

    @classmethod
    def for_workspace(cls, workspace_data_dir: str | Path) -> Self:
        """Create a store for memory scoped to one workspace data slot."""
        path = Path(workspace_data_dir).expanduser().resolve() / MEMORY_FILE_NAME
        return cls(path, scope="workspace")

    def list(self) -> list[MemoryRecord]:
        """Return all memories in stable file order."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as memory_file:
            data = json.load(memory_file)
        if not isinstance(data, dict):
            raise ValueError(f"memory file must contain a JSON object: {self.path}")
        if data.get("version") != MEMORY_VERSION:
            raise ValueError(f"unsupported memory version: {data.get('version')}")
        if data.get("scope") != self.scope:
            raise ValueError(
                f"memory file scope must be {self.scope!r}, "
                f"got {data.get('scope')!r}"
            )
        raw_memories = data.get("memories")
        if not isinstance(raw_memories, list):
            raise ValueError("memory records must be a list")
        records: list[MemoryRecord] = []
        for raw_memory in raw_memories:
            if not isinstance(raw_memory, dict):
                raise ValueError("each memory record must be a JSON object")
            records.append(
                MemoryRecord.from_dict(raw_memory, expected_scope=self.scope)
            )
        return records

    def add(
        self,
        content: str,
        *,
        rationale: str | None = None,
    ) -> MemoryAddResult:
        """Add normalized content or return the existing exact duplicate."""
        normalized = _normalize_content(content)
        normalized_rationale = _normalize_content(
            _MANUAL_MEMORY_RATIONALE if rationale is None else rationale
        )
        records = self.list()
        duplicate = _find_exact(records, normalized)
        if duplicate is not None:
            return MemoryAddResult(record=duplicate, created=False)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = MemoryRecord(
            id=f"mem-{uuid4().hex[:12]}",
            scope=self.scope,
            content=normalized,
            rationale=normalized_rationale,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._write([*records, record])
        return MemoryAddResult(record=record, created=True)

    def find_exact(self, content: str) -> MemoryRecord | None:
        """Return an exact normalized, case-insensitive content match."""
        normalized = _normalize_content(content)
        return _find_exact(self.list(), normalized)

    def replace(
        self,
        memory_id: str,
        content: str,
        *,
        rationale: str | None = None,
    ) -> MemoryRecord:
        """Replace one memory's content while preserving its stable identity."""
        normalized_id = memory_id.strip()
        normalized_content = _normalize_content(content)
        normalized_rationale = _normalize_content(
            _MANUAL_MEMORY_RATIONALE if rationale is None else rationale
        )
        records = self.list()
        index = next(
            (
                index
                for index, record in enumerate(records)
                if record.id == normalized_id
            ),
            None,
        )
        if index is None:
            raise ValueError(f"memory not found: {normalized_id}")

        previous = records[index]
        updated = MemoryRecord(
            id=previous.id,
            scope=previous.scope,
            content=normalized_content,
            rationale=normalized_rationale,
            created_at=previous.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        records[index] = updated
        self._write(records)
        return updated

    def search(self, query: str) -> list[MemoryRecord]:
        """Return memories containing the normalized query, newest first."""
        normalized_query = _normalize_content(query).casefold()
        return [
            record
            for record in reversed(self.list())
            if normalized_query in record.content.casefold()
        ]

    def delete(self, memory_id: str) -> MemoryRecord | None:
        """Delete one memory by id and return it when found."""
        normalized_id = memory_id.strip()
        records = self.list()
        deleted = next(
            (record for record in records if record.id == normalized_id),
            None,
        )
        if deleted is None:
            return None
        self._write([record for record in records if record.id != normalized_id])
        return deleted

    def _write(self, records: list[MemoryRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": MEMORY_VERSION,
            "scope": self.scope,
            "memories": [record.to_dict() for record in records],
        }
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
        temp_path.replace(self.path)


def _normalize_content(content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("memory content must be a string")
    normalized = " ".join(content.split())
    if not normalized:
        raise ValueError("memory content must not be empty")
    return normalized


def _find_exact(
    records: list[MemoryRecord],
    normalized_content: str,
) -> MemoryRecord | None:
    duplicate_key = normalized_content.casefold()
    return next(
        (
            record
            for record in records
            if _normalize_content(record.content).casefold() == duplicate_key
        ),
        None,
    )
