"""Shared public value types for conversation context."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..tokenization import messages_tokens
from .summary_generation import SummaryDiagnostic


@dataclass(frozen=True)
class ContextCompressionBreakdown:
    """Explain how one compressed provider context is composed."""

    summary_source_count: int
    summary_source_tokens: int
    summary_tokens: int
    recent_turn_count: int
    recent_message_count: int
    recent_tokens: int
    current_turn_present: bool
    current_turn_message_count: int
    current_turn_tokens: int


@dataclass(frozen=True)
class ContextConversationView:
    """Filtered conversation projection derived from model-facing context."""

    summary: str | None
    messages: list[dict[str, Any]]

    @classmethod
    def from_messages(
        cls,
        messages: list[dict[str, Any]],
        *,
        summary: str | None = None,
    ) -> ContextConversationView:
        visible: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant" and message.get("tool_calls"):
                continue
            content = message.get("content")
            if isinstance(content, str):
                visible.append({"role": role, "content": content})
        return cls(summary=summary, messages=visible)

    def to_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "messages": deepcopy(self.messages),
        }


@dataclass(frozen=True)
class ModelInput:
    """Final model messages plus detached context artifacts for the Agent boundary."""

    messages: list[dict[str, Any]]
    conversation_view: ContextConversationView
    compressed: bool
    original_count: int
    omitted_count: int
    original_tokens: int
    context_sent_count: int
    context_sent_tokens: int
    tool_results_snipped: int
    summary_diagnostics: tuple[SummaryDiagnostic, ...]
    canonical_message_count: int = 0
    compression_breakdown: ContextCompressionBreakdown | None = None

    @property
    def sent_count(self) -> int:
        return len(self.messages)

    @property
    def sent_tokens(self) -> int:
        return messages_tokens(self.messages)
