"""Public value types returned by the Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..context import ContextConversationView
from ..context.types import ContextCompressionBreakdown


@dataclass(frozen=True)
class AgentTurnResult:
    """Completed main or subagent turn and its explicit context artifact."""

    content: str
    conversation_view: ContextConversationView


@dataclass(frozen=True)
class ContextCompressionNotice:
    """Host-facing progress for one context compression attempt."""

    stage: Literal["started", "completed", "failed"]
    trigger: Literal["automatic", "manual"]
    before_count: int
    before_tokens: int
    sent_count: int | None = None
    sent_tokens: int | None = None
    omitted_count: int | None = None
    generation_requests: int | None = None
    review_performed: bool | None = None
    error: str | None = None
    canonical_count: int | None = None
    canonical_tokens: int | None = None
    breakdown: ContextCompressionBreakdown | None = None
    model_input_count: int | None = None
    model_input_tokens: int | None = None
    system_prompt_tokens: int | None = None
