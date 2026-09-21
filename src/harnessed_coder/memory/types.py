"""Data types for manually managed long-term memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MemoryScope = Literal["user", "workspace"]


@dataclass(frozen=True)
class MemoryRecord:
    """One durable user- or workspace-scoped memory."""

    id: str
    scope: MemoryScope
    content: str
    rationale: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "scope": self.scope,
            "content": self.content,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        expected_scope: MemoryScope,
    ) -> MemoryRecord:
        memory_id = data.get("id")
        scope = data.get("scope")
        content = data.get("content")
        rationale = data.get("rationale")
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory id must be a non-empty string")
        if scope != expected_scope:
            raise ValueError(
                f"memory scope must be {expected_scope!r}, got {scope!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("memory content must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("memory rationale must be a non-empty string")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("memory created_at must be a non-empty string")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("memory updated_at must be a non-empty string")
        return cls(
            id=memory_id,
            scope=expected_scope,
            content=content,
            rationale=rationale,
            created_at=created_at,
            updated_at=updated_at,
        )
