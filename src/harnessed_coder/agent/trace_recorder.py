"""Session-scoped business trace recording for the Agent runtime."""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
import hashlib
from itertools import count
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from ..context import ContextSummaryError, ModelContext, ModelInput
from ..diagnostics.trace import write_trace_event
from ..llm import LLMResponse, ToolCall
from ..permissions import PermissionDecision, permission_trace_payload


logger = logging.getLogger(__name__)
_turn_ids = count(1)
_attempt_ids = count(1)
_model_call_ids = count(1)
_error_ids = count(1)
_current_turn_id: ContextVar[str | None] = ContextVar("trace_turn_id", default=None)


class AgentTraceRecorder:
    """Publish one Agent's events into its owning session trace."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        trace_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run_id = run_id
        self._trace_path = None if trace_path is None else Path(trace_path).resolve()
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("scope", "main")
        self._fixed_turn_id = self._metadata.pop("turn_id", None)
        self._turn_id: str | None = self._fixed_turn_id
        self._attempt_id: str | None = None
        self._turn_active = False
        self._system_prompt_recorded = False
        self._last_system_prompts: tuple[str, ...] = ()
        self._last_system_prompt_sha256: str | None = None

    @property
    def trace_path(self) -> Path | None:
        return self._trace_path

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    @property
    def turn_active(self) -> bool:
        return self._turn_active

    def move_to(self, path: str | Path) -> None:
        target = Path(path).resolve()
        if self._trace_path == target:
            return
        if target.exists():
            raise ValueError(f"session trace already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._trace_path is not None and self._trace_path.exists():
            self._trace_path.replace(target)
        self._trace_path = target

    def begin_turn(self, *, turn_number: int | None = None) -> str:
        if self._fixed_turn_id is not None:
            self._turn_id = self._fixed_turn_id
        elif turn_number is not None:
            self._turn_id = f"turn-{turn_number}"
        else:
            self._turn_id = f"turn-{next(_turn_ids)}"
        self._attempt_id = f"attempt-{next(_attempt_ids)}"
        self._turn_active = True
        if self._fixed_turn_id is None:
            _current_turn_id.set(self._turn_id)
        self._write("agent_attempt", {"status": "started"})
        assert self._turn_id is not None
        return self._turn_id

    def complete_turn(
        self,
        *,
        model_calls: int,
        tool_rounds: int,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
    ) -> None:
        self._write(
            "agent_attempt",
            {
                "status": "completed",
                "model_calls": model_calls,
                "tool_rounds": tool_rounds,
                "usage": _usage(prompt_tokens, completion_tokens),
                "duration_ms": duration_ms,
            },
        )
        if self._fixed_turn_id is None:
            _current_turn_id.set(None)
        self._turn_active = False

    def fail_turn(self, exc: BaseException, *, stage: str, duration_ms: int) -> None:
        error_id = _error_id()
        self._write(
            "operation_failed",
            {
                "stage": stage,
                "error_id": error_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        self._write(
            "agent_attempt",
            {"status": "failed", "duration_ms": duration_ms, "error_id": error_id},
        )
        if self._fixed_turn_id is None:
            _current_turn_id.set(None)
        self._turn_active = False

    def session_message(self, message: dict[str, Any], *, source: str | None = None) -> None:
        data: dict[str, Any] = {"message": deepcopy(message)}
        if source is not None:
            data["source"] = source
        self._write("session_message", data)

    def model_call_started(self, purpose: str, *, model: str) -> tuple[str, float]:
        model_call_id = f"model-call-{next(_model_call_ids)}"
        self._write(
            "model_call",
            {
                "status": "started",
                "model_call_id": model_call_id,
                "purpose": purpose,
                "model": model,
            },
        )
        return model_call_id, perf_counter()

    def system_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
    ) -> None:
        """Record the first request system message and each later content change."""
        system_messages = _system_prompt_messages(messages)
        prompt_contents = tuple(
            str(item["message"]["content"]) for item in system_messages
        )
        if not prompt_contents and not self._system_prompt_recorded:
            return
        if (
            self._system_prompt_recorded
            and prompt_contents == self._last_system_prompts
        ):
            return

        previous_sha256 = self._last_system_prompt_sha256
        prompt_sha256 = _system_prompts_sha256(prompt_contents)
        self._write(
            "system_prompt",
            {
                "change": (
                    "initial" if not self._system_prompt_recorded else "changed"
                ),
                "purpose": purpose,
                "messages": system_messages,
                "message_count": len(system_messages),
                "content_chars": sum(len(content) for content in prompt_contents),
                "sha256": prompt_sha256,
                "previous_sha256": previous_sha256,
            },
        )
        self._system_prompt_recorded = True
        self._last_system_prompts = prompt_contents
        self._last_system_prompt_sha256 = prompt_sha256

    def model_call_completed(
        self,
        call: tuple[str, float],
        *,
        purpose: str,
        model: str,
        response: LLMResponse,
    ) -> None:
        model_call_id, started_at = call
        duration_ms = round((perf_counter() - started_at) * 1000)
        self._write(
            "model_call",
            {
                "status": "completed",
                "model_call_id": model_call_id,
                "purpose": purpose,
                "model": model,
                "duration_ms": duration_ms,
                "performance": _model_performance(response),
                "transport": _model_transport(response),
                "usage": _model_usage(response),
                "response": {
                    "reasoning_chars": len(response.reasoning_content),
                    "content_chars": len(response.content),
                    "tool_calls": len(response.tool_calls),
                },
            },
        )

    def model_call_failed(
        self,
        call: tuple[str, float],
        *,
        purpose: str,
        model: str,
    ) -> None:
        model_call_id, started_at = call
        error_id = _error_id()
        self._write(
            "model_call",
            {
                "status": "failed",
                "model_call_id": model_call_id,
                "purpose": purpose,
                "model": model,
                "duration_ms": round((perf_counter() - started_at) * 1000),
                "error_id": error_id,
            },
        )

    def context_compression(
        self,
        *,
        trigger: str,
        result: ModelContext | ModelInput,
        before_count: int | None = None,
        before_tokens: int | None = None,
        canonical_count: int | None = None,
    ) -> None:
        if not result.summary_diagnostics:
            return
        logger.info(
            "Context compression completed: trigger=%s omitted_messages=%s diagnostics=%s",
            trigger,
            result.omitted_count,
            len(result.summary_diagnostics),
        )
        include_turn = trigger != "manual"
        if isinstance(result, ModelInput):
            history_after = {
                "message_count": result.context_sent_count,
                "estimated_tokens": result.context_sent_tokens,
            }
            model_input = {
                "message_count": result.sent_count,
                "estimated_tokens": result.sent_tokens,
                "system_prompt_tokens": max(
                    0,
                    result.sent_tokens - result.context_sent_tokens,
                ),
            }
        else:
            history_after = {
                "message_count": result.sent_count,
                "estimated_tokens": result.sent_tokens,
            }
            model_input = None
        history_before = {
            "message_count": (
                result.original_count if before_count is None else before_count
            ),
            "estimated_tokens": (
                result.original_tokens if before_tokens is None else before_tokens
            ),
        }
        self._write(
            "context_compression",
            {
                "status": "completed",
                "trigger": trigger,
                # Keep these aliases for existing trace consumers. Their fixed
                # meaning is provider history, excluding the dynamic system prompt.
                "before": history_before,
                "after": history_after,
                "provider_history": {
                    "before": history_before,
                    "after": history_after,
                },
                "model_input": model_input,
                "canonical_history": {
                    "message_count": (
                        getattr(result, "canonical_message_count", result.original_count)
                        if canonical_count is None
                        else canonical_count
                    ),
                    "estimated_tokens": result.original_tokens,
                },
                "breakdown": (
                    None
                    if result.compression_breakdown is None
                    else {
                        "summary": {
                            "source_message_count": result.compression_breakdown.summary_source_count,
                            "source_estimated_tokens": result.compression_breakdown.summary_source_tokens,
                            "result_tokens": result.compression_breakdown.summary_tokens,
                        },
                        "recent_completed_turns": {
                            "turn_count": result.compression_breakdown.recent_turn_count,
                            "message_count": result.compression_breakdown.recent_message_count,
                            "estimated_tokens": result.compression_breakdown.recent_tokens,
                        },
                        "current_turn": {
                            "present": result.compression_breakdown.current_turn_present,
                            "message_count": result.compression_breakdown.current_turn_message_count,
                            "estimated_tokens": result.compression_breakdown.current_turn_tokens,
                        },
                    }
                ),
                "omitted_count": result.omitted_count,
                "diagnostics": [item.to_dict() for item in result.summary_diagnostics],
            },
            include_turn=include_turn,
        )
        messages = result.messages if isinstance(result, ModelInput) else result.model_messages()
        self._write(
            "context_snapshot",
            {"reason": "context_compression", "messages": messages},
            include_turn=include_turn,
        )

    def context_summary_error(
        self,
        *,
        trigger: str,
        error: ContextSummaryError,
        before_count: int | None = None,
        before_tokens: int | None = None,
    ) -> None:
        before = None
        if before_count is not None and before_tokens is not None:
            before = {
                "message_count": before_count,
                "estimated_tokens": before_tokens,
            }
        self._write(
            "context_compression",
            {
                "status": "failed",
                "trigger": trigger,
                "error_id": _error_id(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "before": before,
                "diagnostics": [error.diagnostic.to_dict()],
            },
            include_turn=trigger != "manual",
        )

    def permission_requested(self, tool_call: ToolCall) -> None:
        self._write(
            "permission_requested",
            {"tool_call_id": tool_call.id, "tool_name": tool_call.name},
        )

    def permission_decision(self, tool_call: ToolCall, permission: PermissionDecision) -> None:
        self._write("permission_decision", permission_trace_payload(tool_call, permission))

    def tool_started(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
    ) -> None:
        self._write(
            "tool_started",
            {
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "tool_round": tool_round,
                "call_index": call_index,
                "total_calls": total_calls,
            },
        )

    def tool_completed(
        self,
        tool_call: ToolCall,
        *,
        status: str,
        duration_ms: int,
        result_chars: int,
    ) -> None:
        self._write(
            "tool_completed",
            {
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "status": status,
                "duration_ms": duration_ms,
                "result_chars": result_chars,
            },
        )

    def tool_round_limit(self, *, limit: int, tool_round: int, fallback: bool) -> None:
        self._write(
            "tool_round_limit",
            {"limit": limit, "tool_round": tool_round, "fallback": fallback},
        )

    def turn_token_limit(
        self,
        *,
        limit: int,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
        unreported_requests: int,
        fallback: bool,
    ) -> None:
        self._write(
            "turn_token_limit",
            {
                "limit": limit,
                "usage": {
                    **_usage(prompt_tokens, completion_tokens),
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "noncached_prompt_tokens": max(
                        0, prompt_tokens - cached_prompt_tokens
                    ),
                    "weighted_tokens": (
                        max(0, prompt_tokens - cached_prompt_tokens)
                        + completion_tokens
                        + min(prompt_tokens, cached_prompt_tokens) * 0.1
                    ),
                },
                "unreported_requests": unreported_requests,
                "fallback": fallback,
            },
        )

    def tool_batch_summary(self, *, status: str, **data: Any) -> None:
        """Record background tool-batch summary lifecycle without mutable turn state."""
        self._write(
            "tool_batch_summary",
            {"status": status, **data},
            include_turn=False,
        )

    def session_recovery(self, *, appended_messages: int, message_count: int) -> None:
        self._write(
            "session_recovery",
            {
                "reason": "closed_interrupted_tool_turn",
                "appended_messages": appended_messages,
                "message_count": message_count,
            },
            include_turn=False,
        )

    def session_cleared(self, *, message_count: int, checkpoint_count: int) -> None:
        self._write(
            "session_cleared",
            {
                "previous_message_count": message_count,
                "previous_checkpoint_count": checkpoint_count,
            },
            include_turn=False,
        )

    def session_renamed(self, *, previous: str, current: str) -> None:
        self._write(
            "session_renamed",
            {"previous": previous, "current": current},
            include_turn=False,
        )

    def memory_extraction(self, *, status: str, **data: Any) -> None:
        self._write("memory_extraction", {"status": status, **data})

    def memory_updated(self, *, action: str, memory_id: str, memory_scope: str) -> None:
        self._write(
            "memory_updated",
            {"action": action, "memory_id": memory_id, "memory_scope": memory_scope},
        )

    def subagent(self, *, status: str, subagent_id: str, **data: Any) -> None:
        self._write("subagent", {"status": status, "subagent_id": subagent_id, **data})

    def _write(
        self,
        event: str,
        body: dict[str, Any],
        *,
        include_turn: bool = True,
    ) -> None:
        metadata = dict(self._metadata)
        if self._run_id is not None:
            metadata["run_id"] = self._run_id
        if include_turn and self._turn_id is not None:
            metadata["turn_id"] = self._turn_id
        if include_turn and self._attempt_id is not None:
            metadata["attempt_id"] = self._attempt_id
        try:
            write_trace_event(
                event,
                body,
                trace_path=self._trace_path,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to write trace event: %s", event)


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _model_usage(response: LLMResponse) -> dict[str, Any]:
    usage: dict[str, Any] = _usage(
        response.prompt_tokens,
        response.completion_tokens,
    )
    usage["prompt_tokens_details"] = {
        "cached_tokens": response.cached_prompt_tokens,
        "cache_creation_input_tokens": response.cache_creation_prompt_tokens,
    }
    return usage


def _model_performance(response: LLMResponse) -> dict[str, Any]:
    generation_duration_ms = None
    completion_tokens_per_second = None
    if (
        response.request_duration_ms is not None
        and response.time_to_first_event_ms is not None
    ):
        generation_duration_ms = max(
            0,
            response.request_duration_ms - response.time_to_first_event_ms,
        )
        if generation_duration_ms > 0 and response.completion_tokens > 0:
            completion_tokens_per_second = round(
                response.completion_tokens / (generation_duration_ms / 1000),
                2,
            )
    return {
        "first_event": response.first_event_kind,
        "ttft_ms": response.time_to_first_event_ms,
        "request_duration_ms": response.request_duration_ms,
        "generation_duration_ms": generation_duration_ms,
        "completion_tokens_per_second": completion_tokens_per_second,
    }


def _model_transport(response: LLMResponse) -> dict[str, Any]:
    return {
        "protocol": response.protocol,
        "response_id": response.provider_response_id,
        "previous_response_id": response.previous_response_id,
        "state_reuse": response.response_state_reuse,
        "matched_messages": response.response_matched_messages,
        "input_items": response.response_input_items,
    }


def _system_prompt_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        result.append(
            {
                "index": index,
                "message": {"role": "system", "content": content},
            }
        )
    return result


def _system_prompts_sha256(contents: tuple[str, ...]) -> str:
    serialized = json.dumps(contents, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _error_id() -> str:
    return f"error-{next(_error_ids)}"


def current_trace_turn_id() -> str | None:
    """Return the parent Agent turn for synchronous delegated work."""
    return _current_turn_id.get()
