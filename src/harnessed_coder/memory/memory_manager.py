"""Public application interface for long-term memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..context import ContextConversationView
from .conversation_extraction import (
    LLMMemoryExtractor,
    MemoryCandidateValidator,
    MemoryExtractionResult,
    MemoryExtractor,
    select_recent_memory_extraction_messages,
    select_memory_extraction_messages,
)
from .memory_store import MemoryStore
from .semantic_resolution import LLMMemoryResolver, MemoryResolver
from .types import MemoryRecord


MemoryScope = Literal["user", "workspace"]
MemoryWriteAction = Literal["added", "duplicate", "replaced"]


@dataclass(frozen=True)
class MemoryWriteResult:
    """Outcome of a semantically governed memory addition."""

    action: MemoryWriteAction
    record: MemoryRecord
    previous: MemoryRecord | None = None
    reason: str | None = None


class MemoryManager:
    """Manage user and active-workspace memory for one REPL target."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        workspace_data_dir: str | Path | None,
        model: str,
        resolver: MemoryResolver | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self._user_store = MemoryStore.for_user(data_dir)
        self._workspace_store = (
            MemoryStore.for_workspace(workspace_data_dir)
            if workspace_data_dir is not None
            else None
        )
        self._resolver = resolver or LLMMemoryResolver(model=model)
        self._extractor = extractor or LLMMemoryExtractor(model=model)
        self._candidate_validator = MemoryCandidateValidator()

    def add(
        self,
        scope: MemoryScope,
        content: str,
        *,
        rationale: str | None = None,
    ) -> MemoryWriteResult:
        """Add, reject, or replace memory according to semantic relation."""
        store = self._store(scope)
        exact = store.find_exact(content)
        if exact is not None:
            return MemoryWriteResult(
                action="duplicate",
                record=exact,
                reason="Exact normalized content already exists.",
            )

        existing = store.list()
        if not existing:
            added = store.add(content, rationale=rationale)
            return MemoryWriteResult(action="added", record=added.record)

        resolution = self._resolver.resolve(content, existing)
        if resolution.action == "add":
            added = store.add(content, rationale=rationale)
            return MemoryWriteResult(
                action="added",
                record=added.record,
                reason=resolution.reason,
            )

        target = next(
            record
            for record in existing
            if record.id == resolution.target_id
        )
        if resolution.action == "duplicate":
            return MemoryWriteResult(
                action="duplicate",
                record=target,
                reason=resolution.reason,
            )

        updated = store.replace(target.id, content, rationale=rationale)
        return MemoryWriteResult(
            action="replaced",
            record=updated,
            previous=target,
            reason=resolution.reason,
        )

    def extract_from_conversation(
        self,
        canonical_messages: list[dict[str, Any]],
        *,
        conversation_view: ContextConversationView | None = None,
    ) -> list[MemoryWriteResult]:
        """Extract and govern memories from the latest completed main-agent turn."""
        messages = select_memory_extraction_messages(canonical_messages)
        if not messages:
            return []
        available_scopes: tuple[MemoryScope, ...] = (
            ("user", "workspace")
            if self._workspace_store is not None
            else ("user",)
        )
        extraction_messages = [
            message
            for message in messages
            if message.current_turn and message.role == "user"
        ]
        extraction = self._extractor.extract(
            extraction_messages,
            available_scopes=available_scopes,
            allow_context_request=True,
        )
        if extraction.action == "needs_context":
            extraction_messages = select_recent_memory_extraction_messages(
                canonical_messages
            )
            extraction = self._extractor.extract(
                extraction_messages,
                available_scopes=available_scopes,
                allow_context_request=True,
            )
        if extraction.action == "needs_context":
            fallback_view = conversation_view or ContextConversationView.from_messages(
                canonical_messages
            )
            extraction = self._extractor.extract(
                extraction_messages,
                available_scopes=available_scopes,
                allow_context_request=False,
                conversation_view=fallback_view,
            )
        return self._write_extraction(
            extraction,
            available_scopes=available_scopes,
        )

    def _write_extraction(
        self,
        extraction: MemoryExtractionResult,
        *,
        available_scopes: tuple[MemoryScope, ...],
    ) -> list[MemoryWriteResult]:
        if extraction.action == "none":
            return []
        if extraction.action != "extracted":
            raise ValueError("memory extraction did not resolve")
        candidates = self._candidate_validator.validate(
            list(extraction.candidates),
            available_scopes=available_scopes,
        )
        return [
            self.add(
                candidate.scope,
                candidate.content,
                rationale=candidate.rationale,
            )
            for candidate in candidates
        ]

    def list(self, scope: MemoryScope) -> list[MemoryRecord]:
        """Return memories from the selected scope."""
        return self._store(scope).list()

    def list_all(self) -> list[MemoryRecord]:
        """Return user and active-workspace memories in scope order."""
        return [
            *self._user_store.list(),
            *(
                self._workspace_store.list()
                if self._workspace_store is not None
                else []
            ),
        ]

    def search(
        self,
        scope: MemoryScope,
        query: str,
    ) -> list[MemoryRecord]:
        """Search memories in the selected scope."""
        return self._store(scope).search(query)

    def delete(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> MemoryRecord | None:
        """Delete one memory from the selected scope."""
        return self._store(scope).delete(memory_id)

    def _store(self, scope: MemoryScope) -> MemoryStore:
        if scope == "user":
            return self._user_store
        if scope == "workspace":
            if self._workspace_store is None:
                raise ValueError(
                    "workspace memory requires a configured workspace"
                )
            return self._workspace_store
        raise ValueError(f"unsupported memory scope: {scope}")
