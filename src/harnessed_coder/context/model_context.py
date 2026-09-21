"""Model-facing context values and rolling-checkpoint serialization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .summary_generation import SummaryDiagnostic
from .types import ContextCompressionBreakdown, ContextConversationView


def _compact_context_content(
    summary: str,
    transcript_messages: list[dict[str, Any]],
) -> str:
    transcript = json.dumps(
        {"version": 1, "messages": transcript_messages},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        [
            '<compact_context type="historical_data">',
            "  <summary>",
            summary,
            "  </summary>",
            "",
            '  <transcript format="openai_messages_json">',
            transcript,
            "  </transcript>",
            "</compact_context>",
        ]
    )


@dataclass(frozen=True)
class ModelContext:
    """Governed model context, independent of any request-level system prompt."""

    summary: str | None
    transcript_messages: list[dict[str, Any]]
    retained_messages: list[dict[str, Any]]
    original_count: int
    omitted_count: int
    original_tokens: int
    sent_tokens: int
    tool_results_snipped: int = 0
    summary_diagnostics: tuple[SummaryDiagnostic, ...] = ()
    compression_breakdown: ContextCompressionBreakdown | None = None

    @property
    def compressed(self) -> bool:
        return self.summary is not None

    @property
    def sent_count(self) -> int:
        return len(self.retained_messages) + int(self.summary is not None)

    def model_messages(self) -> list[dict[str, Any]]:
        messages = deepcopy(self.retained_messages)
        if self.summary is not None:
            messages.insert(0, {"role": "user", "content": self._compact_message()})
        return messages

    def conversation_view(self) -> ContextConversationView:
        return ContextConversationView.from_messages(
            [*self.transcript_messages, *self.retained_messages],
            summary=self.summary,
        )

    def _compact_message(self) -> str:
        if self.summary is None:
            raise ValueError("uncompressed model context has no compact message")
        return _compact_context_content(
            self.summary,
            self.transcript_messages,
        )

    def without_diagnostics(self) -> ModelContext:
        return ModelContext(
            summary=self.summary,
            transcript_messages=deepcopy(self.transcript_messages),
            retained_messages=deepcopy(self.retained_messages),
            original_count=self.original_count,
            omitted_count=self.omitted_count,
            original_tokens=self.original_tokens,
            sent_tokens=self.sent_tokens,
            tool_results_snipped=self.tool_results_snipped,
            compression_breakdown=self.compression_breakdown,
        )


@dataclass(frozen=True)
class ModelContextCheckpoint:
    """Persistent summary checkpoint for a canonical-history prefix."""

    version: int
    source_fingerprint: str
    source_count: int
    summary: str
    transcript_messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_fingerprint": self.source_fingerprint,
            "source_count": self.source_count,
            "summary": self.summary,
            "transcript_messages": deepcopy(self.transcript_messages),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ModelContextCheckpoint | None:
        if not isinstance(value, dict):
            return None
        summary = value.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        transcript_messages = value.get("transcript_messages")
        if not isinstance(transcript_messages, list) or not all(
            isinstance(message, dict) for message in transcript_messages
        ):
            return None
        try:
            return cls(
                version=int(value["version"]),
                source_fingerprint=str(value["source_fingerprint"]),
                source_count=int(value["source_count"]),
                summary=summary,
                transcript_messages=deepcopy(transcript_messages),
            )
        except (KeyError, TypeError, ValueError):
            return None
