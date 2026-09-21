"""Agent orchestration loop."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, Protocol

from ..context import (
    ContextCompressionAnalysis,
    ContextCompressor,
    ContextSummaryError,
    ContextConversationView,
    ModelContext,
    RequestContextAnalysis,
    RequestContextManager,
    SystemPromptProvider,
)
from ..context.tool_batch_summary import ToolBatchSummaryFunction, ToolBatchSummaryNotice
from harnessed_coder.session.usage import tracked_call, agent_usage
from ..llm import (
    LLMResponse,
    ResponseContextHandle,
    ToolCall,
    open_response_context,
    reset_response_context,
    responses as llm_responses,
)
from ..user_config import get_responses_full_history
from ..permissions import (
    PermissionApprover,
    PermissionReviewer,
)
from ..context.message_sequence import validate_message_sequence
from .recovery import close_interrupted_tool_turn
from .tool_execution import ToolCallExecutor
from .tool_call_policy import (
    MISSING_FIRST_TOOL_CALL_CONTENT_MESSAGE,
    MISSING_FIRST_TOOL_CALL_CONTENT_STATUS,
)
from .turn_limit import TOOL_ROUND_LIMIT, TURN_TOKEN_LIMIT, TurnLimit
from ..tools import ToolExecutionResult, ToolRegistry
from ..session import ConversationSession, conversation_turns
from ..session.message_metadata import SESSION_RECOVERY_CONTENT_SOURCE
from .trace_recorder import AgentTraceRecorder
from .types import AgentTurnResult, ContextCompressionNotice

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ROUNDS = 100
DEFAULT_MAX_TURN_TOKENS = 1_000_000


class _UsageTotals(NamedTuple):
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    unreported_requests: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def noncached_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_prompt_tokens)

    @property
    def weighted_token_tenths(self) -> int:
        return (
            (self.noncached_prompt_tokens + self.completion_tokens) * 10
            + min(self.prompt_tokens, self.cached_prompt_tokens)
        )

    @property
    def weighted_tokens(self) -> float:
        return self.weighted_token_tenths / 10


class ChatFunction(Protocol):
    """LLM function contract required by the agent loop."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        base_url: str | None,
        tools: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None,
        on_activity_delta: Callable[[str, int], None] | None,
    ) -> LLMResponse: ...


class Agent:
    """Manage conversation history and execute model-requested tools."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        model: str,
        base_url: str | None = None,
        chat_function: ChatFunction | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_turn_tokens: int = DEFAULT_MAX_TURN_TOKENS,
        run_id: str | None = None,
        trace_path: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        session: ConversationSession | None = None,
        context_compressor: ContextCompressor | None = None,
        system_prompt_provider: SystemPromptProvider | None = None,
        permission_approver: PermissionApprover | None = None,
        permission_reviewer: PermissionReviewer | None = None,
        tool_batch_summary_function: ToolBatchSummaryFunction | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if max_turn_tokens < 1:
            raise ValueError("max_turn_tokens must be at least 1")

        self._usage_purpose = "subagent" if (trace_metadata or {}).get("scope") == "subagent" else "agent"
        self.tools = tools
        self.model = model
        self.base_url = base_url
        self.session = session or ConversationSession()
        self._chat_function = chat_function or llm_responses
        self._response_context: ResponseContextHandle | None = (
            open_response_context() if chat_function is None else None
        )
        self._max_tool_rounds = max_tool_rounds
        self._max_turn_tokens = max_turn_tokens
        self._trace = AgentTraceRecorder(
            run_id=run_id,
            trace_path=trace_path,
            metadata=trace_metadata,
        )
        self._context_manager = RequestContextManager(
            session=self.session,
            compressor=context_compressor,
            system_prompt_provider=system_prompt_provider,
            tool_batch_summary_function=tool_batch_summary_function,
            tool_batch_summary_event_recorder=self._trace.tool_batch_summary,
            preserve_reasoning_content=(
                chat_function is None and get_responses_full_history()
            ),
        )
        self._context_compression_reporter: (
            Callable[[ContextCompressionNotice], None] | None
        ) = None
        self._automatic_noop_compression_reported_in_turn = False
        self._model_request_reporter: Callable[[str], None] | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.llm_request_count = 0
        self._tool_executor = ToolCallExecutor(
            self.tools,
            on_tool_started=self._trace.tool_started,
            on_tool_response=self._append_tool_response,
            on_permission_decision=self._trace.permission_decision,
            on_permission_requested=self._trace.permission_requested,
            permission_approver=permission_approver,
            permission_reviewer=permission_reviewer,
            permission_context_provider=self._permission_task_context,
        )
        self.tools.set_permission_approver(permission_approver)

    def set_permission_approver(
        self,
        permission_approver: PermissionApprover | None,
    ) -> None:
        """Set the host callback used for future interactive permission checks."""
        self._tool_executor.set_permission_approver(permission_approver)
        self.tools.set_permission_approver(permission_approver)

    def set_permission_review_reporter(self, reporter: Callable[..., None] | None) -> None:
        """Set the host callback for automatic permission-review progress."""
        self._tool_executor.set_permission_review_reporter(reporter)

    def set_context_compression_reporter(
        self,
        reporter: Callable[[ContextCompressionNotice], None] | None,
    ) -> None:
        """Set the host callback for automatic context-compression progress."""
        self._context_compression_reporter = reporter

    def set_tool_batch_summary_reporter(
        self,
        reporter: Callable[[ToolBatchSummaryNotice], None] | None,
    ) -> None:
        """Set the host callback for waits and asynchronous summary failures."""
        self._context_manager.set_tool_batch_summary_reporter(reporter)

    def set_model_request_reporter(
        self,
        reporter: Callable[[str], None] | None,
    ) -> None:
        """Set the host callback for model-request start and finish events."""
        self._model_request_reporter = reporter

    @agent_usage
    def compact_context(
        self,
        *,
        on_compression_started: Callable[[ContextCompressionAnalysis], None] | None = None,
    ) -> ModelContext:
        """Explicitly compact this session's current model context."""
        compression_start: ContextCompressionAnalysis | None = None

        def capture_start(analysis: ContextCompressionAnalysis) -> None:
            nonlocal compression_start
            compression_start = analysis
            if on_compression_started is not None:
                on_compression_started(analysis)

        try:
            result = self._context_manager.compact_history(
                on_compression_started=capture_start,
            )
        except ContextSummaryError as exc:
            self._trace.context_summary_error(
                trigger="manual",
                error=exc,
                before_count=(
                    compression_start.request_count
                    if compression_start is not None
                    else None
                ),
                before_tokens=(
                    compression_start.request_tokens
                    if compression_start is not None
                    else None
                ),
            )
            raise
        self._trace.context_compression(
            trigger="manual",
            result=result,
            canonical_count=self.session.message_count(),
            before_count=(
                compression_start.request_count
                if compression_start is not None
                else None
            ),
            before_tokens=(
                compression_start.request_tokens
                if compression_start is not None
                else None
            ),
        )
        return result

    @property
    def trace_path(self) -> Path | None:
        return self._trace.trace_path

    @property
    def current_turn_id(self) -> str | None:
        return self._trace.turn_id

    def clear_session(self) -> None:
        message_count = self.session.message_count()
        checkpoint_count = self.session.checkpoint_count()
        self.session.clear()
        if self._response_context is not None:
            reset_response_context(self._response_context)
        self._trace.session_cleared(
            message_count=message_count,
            checkpoint_count=checkpoint_count,
        )

    def move_trace_to(self, path: str, *, previous: str, current: str) -> None:
        self._trace.move_to(path)
        self._trace.session_renamed(previous=previous, current=current)

    def trace_memory_extraction(self, *, status: str, **data: Any) -> None:
        self._trace.memory_extraction(status=status, **data)

    def trace_memory_updated(self, *, action: str, memory_id: str, scope: str) -> None:
        self._trace.memory_updated(
            action=action,
            memory_id=memory_id,
            memory_scope=scope,
        )

    def analyze_context(self) -> RequestContextAnalysis:
        """Return read-only context usage for this session."""
        return self._context_manager.analyze()

    def _permission_task_context(self) -> str:
        turns = conversation_turns(self.session.snapshot())
        turn_messages = turns[-1].messages if turns else []
        user_contents = [
            content
            for message in turn_messages
            if message.get("role") == "user"
            if isinstance((content := message.get("content")), str)
        ]
        return "\n\n".join(user_contents)[-4_000:]

    @agent_usage
    def chat(
        self,
        user_input: str,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_activity_delta: Callable[[str, int], None] | None = None,
        on_tool_call_start: Callable[..., None] | None = None,
        on_tool_call_end: Callable[..., None] | None = None,
    ) -> AgentTurnResult:
        """Process one user turn until the model returns a final text response."""
        self._automatic_noop_compression_reported_in_turn = False
        started_at = perf_counter()
        try:
            return self._chat(
                user_input,
                on_text_delta=on_text_delta,
                on_activity_delta=on_activity_delta,
                on_tool_call_start=on_tool_call_start,
                on_tool_call_end=on_tool_call_end,
            )
        except Exception as exc:
            if self._trace.turn_active:
                self._trace.fail_turn(
                    exc,
                    stage="agent_attempt",
                    duration_ms=round((perf_counter() - started_at) * 1000),
                )
            raise

    def _chat(
        self,
        user_input: str,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_activity_delta: Callable[[str, int], None] | None = None,
        on_tool_call_start: Callable[..., None] | None = None,
        on_tool_call_end: Callable[..., None] | None = None,
    ) -> AgentTurnResult:
        started_at = perf_counter()
        request_count_before = self.llm_request_count
        prompt_tokens_before = self.total_prompt_tokens
        completion_tokens_before = self.total_completion_tokens
        if self._close_interrupted_turn_if_needed():
            logger.warning("Closed interrupted saved turn before appending new user input")
        # Reject corrupt restored history before adding or persisting a new turn.
        validate_message_sequence(self.session.snapshot())
        turns = conversation_turns(self.session.snapshot())
        continuing_turn = bool(turns and turns[-1].status == "in_progress")
        if not continuing_turn:
            self.tools.clear_turn_exposures()
        turn_number = turns[-1].number if continuing_turn else len(turns) + 1
        # The REPL has accepted the input, but keep it outside canonical history
        # until earlier-turn summary attempts settle. An interruption during
        # this wait therefore cannot leave a user-only partial turn behind.
        self._context_manager.ensure_ready_for_user_input()
        session_usage_before = _session_usage_totals(self.session.usage_snapshot())
        self._trace.begin_turn(turn_number=turn_number)
        self._append_message({"role": "user", "content": user_input})
        # Counts model responses that request tool calls, not rounds where tools
        # were actually executed. The first over-limit request receives tool
        # error responses instead of real execution.
        tool_rounds = 0
        first_tool_call_content_satisfied = False
        # Guards against a model that keeps requesting tools after receiving the
        # over-limit tool errors and final-response system guidance.
        execution_limit_reported: TurnLimit | None = None

        while True:
            tool_definitions = self.tools.definitions()
            response, conversation_view = self._request_llm_response(
                tool_definitions=tool_definitions,
                on_text_delta=on_text_delta,
                on_activity_delta=on_activity_delta,
            )
            self._record_usage(response)
            turn_usage = _usage_since(
                session_usage_before,
                _session_usage_totals(self.session.usage_snapshot()),
            )
            if not response.tool_calls and not response.content.strip():
                raise RuntimeError("model returned no final answer")
            self._append_message(response.to_history_message())

            if not response.tool_calls:
                logger.info(
                    "Agent turn completed: model_calls=%s tool_rounds=%s "
                    "prompt_tokens=%s completion_tokens=%s response_chars=%s "
                    "duration_ms=%s",
                    self.llm_request_count - request_count_before,
                    tool_rounds,
                    self.total_prompt_tokens - prompt_tokens_before,
                    self.total_completion_tokens - completion_tokens_before,
                    len(response.content),
                    round((perf_counter() - started_at) * 1000),
                )
                self._trace.complete_turn(
                    model_calls=self.llm_request_count - request_count_before,
                    tool_rounds=tool_rounds,
                    prompt_tokens=self.total_prompt_tokens - prompt_tokens_before,
                    completion_tokens=self.total_completion_tokens - completion_tokens_before,
                    duration_ms=round((perf_counter() - started_at) * 1000),
                )
                return AgentTurnResult(
                    content=response.content,
                    conversation_view=conversation_view,
                )

            missing_first_tool_call_content = (
                not first_tool_call_content_satisfied
                and not response.content.strip()
            )
            if not missing_first_tool_call_content:
                first_tool_call_content_satisfied = True
            tool_rounds += 1
            tool_round_limit_reached = tool_rounds > self._max_tool_rounds
            turn_token_limit_reached = (
                turn_usage.weighted_token_tenths > self._max_turn_tokens * 10
            )
            execution_limit = execution_limit_reported
            if execution_limit is None and turn_token_limit_reached:
                execution_limit = TURN_TOKEN_LIMIT
            if execution_limit is None and tool_round_limit_reached:
                execution_limit = TOOL_ROUND_LIMIT
            if execution_limit is TOOL_ROUND_LIMIT:
                logger.error(
                    "Maximum tool-call rounds exceeded: limit=%s",
                    self._max_tool_rounds,
                )
                self._trace.tool_round_limit(
                    limit=self._max_tool_rounds,
                    tool_round=tool_rounds,
                    fallback=execution_limit_reported is not None,
                )
            elif execution_limit is TURN_TOKEN_LIMIT:
                logger.error(
                    "Maximum weighted cumulative turn tokens exceeded: limit=%s used=%s",
                    self._max_turn_tokens,
                    turn_usage.weighted_tokens,
                )
                self._trace.turn_token_limit(
                    limit=self._max_turn_tokens,
                    prompt_tokens=turn_usage.prompt_tokens,
                    cached_prompt_tokens=turn_usage.cached_prompt_tokens,
                    completion_tokens=turn_usage.completion_tokens,
                    unreported_requests=turn_usage.unreported_requests,
                    fallback=execution_limit_reported is not None,
                )

            self._execute_tool_calls(
                response.tool_calls,
                available_tool_names={
                    definition["function"]["name"]
                    for definition in tool_definitions
                },
                tool_round=tool_rounds,
                blocked_status=(
                    execution_limit.status
                    if execution_limit is not None
                    else (
                        MISSING_FIRST_TOOL_CALL_CONTENT_STATUS
                        if missing_first_tool_call_content
                        else None
                    )
                ),
                blocked_message=(
                    execution_limit.tool_result
                    if execution_limit is not None
                    else (
                        MISSING_FIRST_TOOL_CALL_CONTENT_MESSAGE
                        if missing_first_tool_call_content
                        else ""
                    )
                ),
                on_tool_call_start=on_tool_call_start,
                on_tool_call_end=on_tool_call_end,
            )
            if execution_limit is not None:
                if execution_limit_reported is not None:
                    fallback_response = self._request_execution_limit_fallback(
                        execution_limit,
                        on_text_delta=on_text_delta,
                        on_activity_delta=on_activity_delta,
                    )
                    self._record_usage(fallback_response)
                    self._append_message(fallback_response.to_history_message())
                    logger.info(
                        "Agent turn completed with %s fallback: "
                        "model_calls=%s tool_rounds=%s prompt_tokens=%s "
                        "completion_tokens=%s response_chars=%s duration_ms=%s",
                        execution_limit.status,
                        self.llm_request_count - request_count_before,
                        tool_rounds,
                        self.total_prompt_tokens - prompt_tokens_before,
                        self.total_completion_tokens - completion_tokens_before,
                        len(fallback_response.content),
                        round((perf_counter() - started_at) * 1000),
                    )
                    self._trace.complete_turn(
                        model_calls=self.llm_request_count - request_count_before,
                        tool_rounds=tool_rounds,
                        prompt_tokens=self.total_prompt_tokens - prompt_tokens_before,
                        completion_tokens=self.total_completion_tokens - completion_tokens_before,
                        duration_ms=round((perf_counter() - started_at) * 1000),
                    )
                    return AgentTurnResult(
                        content=fallback_response.content,
                        conversation_view=conversation_view,
                    )
                execution_limit_reported = execution_limit
                self._append_message(
                    {
                        "role": "system",
                        "content": execution_limit.system_message,
                    }
                )
                continue

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        *,
        available_tool_names: set[str],
        tool_round: int,
        blocked_status: str | None,
        blocked_message: str,
        on_tool_call_start: Callable[..., None] | None,
        on_tool_call_end: Callable[..., None] | None,
    ) -> None:
        """Execute the model's tool-call batch."""
        self._tool_executor.execute(
            tool_calls,
            available_tool_names=available_tool_names,
            tool_round=tool_round,
            blocked_status=blocked_status,
            blocked_message=blocked_message,
            on_tool_call_start=on_tool_call_start,
            on_tool_call_end=on_tool_call_end,
        )

    def _request_llm_response(
        self,
        *,
        tool_definitions: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None,
        on_activity_delta: Callable[[str, int], None] | None,
    ) -> tuple[LLMResponse, ContextConversationView]:
        compression_to_report: ContextCompressionAnalysis | None = None

        def report_started(analysis: ContextCompressionAnalysis) -> None:
            nonlocal compression_to_report
            if self._automatic_noop_compression_reported_in_turn:
                return
            compression_to_report = analysis
            self._report_context_compression(
                ContextCompressionNotice(
                    stage="started",
                    trigger="automatic",
                    before_count=analysis.request_count,
                    before_tokens=analysis.request_tokens,
                )
            )
        try:
            model_input = self._context_manager.prepare(
                on_compression_started=report_started,
            )
        except ContextSummaryError as exc:
            self._trace.context_summary_error(
                trigger="automatic",
                error=exc,
                before_count=(
                    compression_to_report.request_count
                    if compression_to_report is not None
                    else None
                ),
                before_tokens=(
                    compression_to_report.request_tokens
                    if compression_to_report is not None
                    else None
                ),
            )
            if compression_to_report is not None:
                self._report_context_compression(
                    ContextCompressionNotice(
                        stage="failed",
                        trigger="automatic",
                        before_count=compression_to_report.request_count,
                        before_tokens=compression_to_report.request_tokens,
                        generation_requests=exc.diagnostic.generation_requests,
                        review_performed=exc.diagnostic.review_performed,
                        error=str(exc),
                    )
                )
            raise
        if compression_to_report is not None:
            generation_requests = sum(
                diagnostic.generation_requests or 0
                for diagnostic in model_input.summary_diagnostics
            )
            review_performed = any(
                diagnostic.review_performed is True
                for diagnostic in model_input.summary_diagnostics
            )
            compression_was_noop = (
                generation_requests == 0
                and model_input.context_sent_count
                == compression_to_report.request_count
                and model_input.context_sent_tokens
                == compression_to_report.request_tokens
            )
            self._report_context_compression(
                ContextCompressionNotice(
                    stage="completed",
                    trigger="automatic",
                    before_count=compression_to_report.request_count,
                    before_tokens=compression_to_report.request_tokens,
                    sent_count=model_input.context_sent_count,
                    sent_tokens=model_input.context_sent_tokens,
                    omitted_count=(
                        0 if compression_was_noop else model_input.omitted_count
                    ),
                    generation_requests=generation_requests,
                    review_performed=review_performed,
                    canonical_count=model_input.canonical_message_count,
                    canonical_tokens=model_input.original_tokens,
                    breakdown=model_input.compression_breakdown,
                    model_input_count=model_input.sent_count,
                    model_input_tokens=model_input.sent_tokens,
                    system_prompt_tokens=max(
                        0,
                        model_input.sent_tokens - model_input.context_sent_tokens,
                    ),
                )
            )
            if compression_was_noop:
                self._automatic_noop_compression_reported_in_turn = True
        self._trace.context_compression(
            trigger="automatic",
            result=model_input,
            canonical_count=model_input.canonical_message_count,
            before_count=(
                compression_to_report.request_count
                if compression_to_report is not None
                else None
            ),
            before_tokens=(
                compression_to_report.request_tokens
                if compression_to_report is not None
                else None
            ),
        )
        self._trace.system_prompt(model_input.messages, purpose="agent")
        model_call = self._trace.model_call_started("agent", model=self.model)
        self._report_model_request("started")
        try:
            response = self._call_model(
                model_input.messages,
                tools=tool_definitions,
                on_text_delta=on_text_delta,
                on_activity_delta=on_activity_delta,
            )
        except Exception:
            self._trace.model_call_failed(
                model_call,
                purpose="agent",
                model=self.model,
            )
            raise
        finally:
            self._report_model_request("finished")
        self._trace.model_call_completed(
            model_call,
            purpose="agent",
            model=self.model,
            response=response,
        )
        return response, model_input.conversation_view

    def _report_context_compression(self, notice: ContextCompressionNotice) -> None:
        reporter = self._context_compression_reporter
        if reporter is not None:
            reporter(notice)

    def _record_usage(self, response: LLMResponse) -> None:
        self.last_prompt_tokens = response.prompt_tokens
        self.last_completion_tokens = response.completion_tokens
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.llm_request_count += 1

    def _request_execution_limit_fallback(
        self,
        execution_limit: TurnLimit,
        *,
        on_text_delta: Callable[[str], None] | None,
        on_activity_delta: Callable[[str, int], None] | None,
    ) -> LLMResponse:
        purpose = execution_limit.fallback_purpose
        messages = execution_limit.fallback_messages(self.session.snapshot())
        self._trace.system_prompt(messages, purpose=purpose)
        model_call = self._trace.model_call_started(purpose, model=self.model)
        self._report_model_request("started")
        try:
            response = self._call_model(
                messages,
                tools=[],
                on_text_delta=on_text_delta,
                on_activity_delta=on_activity_delta,
            )
        except Exception:
            self._trace.model_call_failed(
                model_call,
                purpose=purpose,
                model=self.model,
            )
            raise
        finally:
            self._report_model_request("finished")
        self._trace.model_call_completed(
            model_call,
            purpose=purpose,
            model=self.model,
            response=response,
        )
        if not response.content.strip():
            raise RuntimeError("execution-limit fallback response was empty")
        # This request is final-only even if a non-conforming provider returns
        # tool calls despite receiving no tool definitions.
        return LLMResponse(
            content=response.content,
            reasoning_content=response.reasoning_content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            usage_available=response.usage_available,
            first_event_kind=response.first_event_kind,
            time_to_first_event_ms=response.time_to_first_event_ms,
            request_duration_ms=response.request_duration_ms,
            cached_prompt_tokens=response.cached_prompt_tokens,
            cache_creation_prompt_tokens=response.cache_creation_prompt_tokens,
            protocol=response.protocol,
            provider_response_id=response.provider_response_id,
            previous_response_id=response.previous_response_id,
            response_state_reuse=response.response_state_reuse,
            response_matched_messages=response.response_matched_messages,
            response_input_items=response.response_input_items,
        )

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        on_text_delta: Callable[[str], None] | None,
        on_activity_delta: Callable[[str, int], None] | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": self.base_url,
            "tools": tools,
            "on_text_delta": on_text_delta,
            "on_activity_delta": on_activity_delta,
        }
        if self._response_context is not None:
            kwargs["response_context"] = self._response_context
        return tracked_call(
            self._chat_function,
            self._usage_purpose,
            messages,
            **kwargs,
        )

    def _report_model_request(self, stage: str) -> None:
        reporter = self._model_request_reporter
        if reporter is not None:
            reporter(stage)

    def _append_tool_response(
        self,
        tool_call: ToolCall,
        result: ToolExecutionResult,
        *,
        status: str,
        duration_seconds: float,
    ) -> None:
        message: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result.model_content,
        }
        if result.internal_data is not None:
            message["internal_result"] = result.internal_data
        self._append_message(message)
        self._trace.tool_completed(
            tool_call,
            status=status,
            duration_ms=round(duration_seconds * 1000),
            result_chars=len(result.model_content),
        )

    def _close_interrupted_turn_if_needed(self) -> bool:
        original_count = self.session.message_count()
        recovery = close_interrupted_tool_turn(self.session.snapshot())
        if not recovery.changed:
            return False
        validate_message_sequence(recovery.messages)
        appended = recovery.messages[original_count:]
        self.session.append_messages(appended)
        for message in appended:
            self._trace.session_message(
                message,
                source=SESSION_RECOVERY_CONTENT_SOURCE,
            )
        self._trace.session_recovery(
            appended_messages=len(appended),
            message_count=self.session.message_count(),
        )
        return True

    def _append_message(self, message: dict[str, Any]) -> None:
        self.session.append_message(message)
        self._trace.session_message(message)

    def _last_message(self) -> dict[str, Any] | None:
        return self.session.last_message()


def _session_usage_totals(rows: list[dict[str, Any]]) -> _UsageTotals:
    return _UsageTotals(
        prompt_tokens=sum(int(row.get("prompt_tokens", 0)) for row in rows),
        cached_prompt_tokens=sum(
            int(row.get("cached_prompt_tokens", 0)) for row in rows
        ),
        completion_tokens=sum(int(row.get("completion_tokens", 0)) for row in rows),
        unreported_requests=sum(int(row.get("unreported", 0)) for row in rows),
    )


def _usage_since(before: _UsageTotals, current: _UsageTotals) -> _UsageTotals:
    return _UsageTotals(
        prompt_tokens=max(0, current.prompt_tokens - before.prompt_tokens),
        cached_prompt_tokens=max(
            0, current.cached_prompt_tokens - before.cached_prompt_tokens
        ),
        completion_tokens=max(0, current.completion_tokens - before.completion_tokens),
        unreported_requests=max(0, current.unreported_requests - before.unreported_requests),
    )
