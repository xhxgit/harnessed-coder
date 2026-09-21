"""Permission-aware tool-call scheduling and execution."""

from __future__ import annotations

from contextvars import copy_context

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from time import perf_counter

from ..llm import ToolCall
from ..constants.tool_protocol import HISTORICAL_COMPRESSION_ARGUMENT
from ..permissions import (
    PermissionAction,
    PermissionApprovalRequest,
    PermissionApprover,
    PermissionDecision,
    PermissionReview,
    PermissionReviewAction,
    PermissionReviewNotice,
    PermissionReviewer,
)
from ..tools import ToolExecutionResult, ToolRegistry

logger = logging.getLogger(__name__)

ToolCallStartCallback = Callable[..., None]
ToolCallEndCallback = Callable[..., None]
ToolStartedRecorder = Callable[..., None]
ToolResponseRecorder = Callable[..., None]
PermissionDecisionRecorder = Callable[[ToolCall, PermissionDecision], None]
PermissionRequestedRecorder = Callable[[ToolCall], None]
PermissionReviewReporter = Callable[[PermissionReviewNotice], None]


def _as_execution_result(value: str | ToolExecutionResult) -> ToolExecutionResult:
    if isinstance(value, ToolExecutionResult):
        return value
    return ToolExecutionResult(model_content=value)


def _tool_not_available_message(tool_name: str) -> str:
    return (
        f"Error: Tool '{tool_name}' was not available in the tool definitions for "
        "the model response that requested it. Only call tools included in the "
        "current request. Use tool_search first when another capability is needed, "
        "then call the exposed tool in a later model response."
    )


def _reserved_historical_argument_message() -> str:
    return (
        f"Error: '{HISTORICAL_COMPRESSION_ARGUMENT}' is reserved for host-generated "
        "historical projections and cannot be used in a new tool call. Retry the "
        "call without this argument."
    )


class ToolCallExecutor:
    """Execute model-requested tool calls while preserving response order."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        on_tool_started: ToolStartedRecorder,
        on_tool_response: ToolResponseRecorder,
        on_permission_decision: PermissionDecisionRecorder | None = None,
        on_permission_requested: PermissionRequestedRecorder | None = None,
        permission_approver: PermissionApprover | None = None,
        permission_reviewer: PermissionReviewer | None = None,
        permission_context_provider: Callable[[], str] | None = None,
    ) -> None:
        self._tools = tools
        self._on_tool_started = on_tool_started
        self._on_tool_response = on_tool_response
        self._on_permission_decision = on_permission_decision
        self._on_permission_requested = on_permission_requested
        self._permission_approver = permission_approver
        self._permission_reviewer = permission_reviewer
        self._permission_context_provider = permission_context_provider
        self._permission_review_reporter: PermissionReviewReporter | None = None

    def set_permission_approver(
        self,
        permission_approver: PermissionApprover | None,
    ) -> None:
        """Replace the host callback used to resolve future ask decisions."""
        self._permission_approver = permission_approver

    def set_permission_review_reporter(
        self,
        reporter: PermissionReviewReporter | None,
    ) -> None:
        """Set the host callback for automatic permission-review progress."""
        self._permission_review_reporter = reporter

    def execute(
        self,
        tool_calls: list[ToolCall],
        *,
        available_tool_names: set[str],
        tool_round: int,
        blocked_status: str | None,
        blocked_message: str,
        on_tool_call_start: ToolCallStartCallback | None = None,
        on_tool_call_end: ToolCallEndCallback | None = None,
    ) -> None:
        """Execute a model tool-call batch and emit tool responses."""
        available_tool_names = set(available_tool_names)
        total_tool_calls = len(tool_calls)
        if blocked_status is not None:
            for tool_call in tool_calls:
                self._record_tool_response(
                    tool_call,
                    ToolExecutionResult(model_content=blocked_message),
                    status=blocked_status,
                    duration_seconds=0,
                )
            return

        index = 0
        while index < total_tool_calls:
            tool_call = tool_calls[index]
            if tool_call.name not in available_tool_names:
                self._reject_unavailable_tool_call(
                    tool_call,
                    tool_round=tool_round,
                    call_index=index + 1,
                    total_calls=total_tool_calls,
                    on_tool_call_start=on_tool_call_start,
                    on_tool_call_end=on_tool_call_end,
                )
                index += 1
                continue
            if HISTORICAL_COMPRESSION_ARGUMENT in tool_call.arguments:
                self._reject_reserved_historical_argument(
                    tool_call,
                    tool_round=tool_round,
                    call_index=index + 1,
                    total_calls=total_tool_calls,
                    on_tool_call_start=on_tool_call_start,
                    on_tool_call_end=on_tool_call_end,
                )
                index += 1
                continue

            # Only adjacent read-only/parallel-safe calls are batched. A serial
            # tool acts as a barrier because it may depend on previous reads or
            # mutate state that later calls observe.
            parallel_batch: list[tuple[int, ToolCall]] = []
            while index < total_tool_calls:
                tool_call = tool_calls[index]
                if tool_call.name not in available_tool_names:
                    break
                if HISTORICAL_COMPRESSION_ARGUMENT in tool_call.arguments:
                    break
                if not self._tools.can_execute_in_parallel(tool_call.name):
                    break
                parallel_batch.append((index + 1, tool_call))
                index += 1

            if len(parallel_batch) > 1:
                self._execute_parallel_batch(
                    parallel_batch,
                    tool_round=tool_round,
                    total_calls=total_tool_calls,
                    on_tool_call_start=on_tool_call_start,
                    on_tool_call_end=on_tool_call_end,
                )
                continue
            if len(parallel_batch) == 1:
                call_index, tool_call = parallel_batch[0]
                self._execute_single(
                    tool_call,
                    tool_round=tool_round,
                    call_index=call_index,
                    total_calls=total_tool_calls,
                    on_tool_call_start=on_tool_call_start,
                    on_tool_call_end=on_tool_call_end,
                )
                continue

            tool_call = tool_calls[index]
            self._execute_single(
                tool_call,
                tool_round=tool_round,
                call_index=index + 1,
                total_calls=total_tool_calls,
                on_tool_call_start=on_tool_call_start,
                on_tool_call_end=on_tool_call_end,
            )
            index += 1

    def _reject_unavailable_tool_call(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
        on_tool_call_start: ToolCallStartCallback | None,
        on_tool_call_end: ToolCallEndCallback | None,
    ) -> None:
        """Reject a call absent from the exact tool-schema snapshot sent to the model."""
        started_at = perf_counter()
        result = ToolExecutionResult(
            model_content=_tool_not_available_message(tool_call.name)
        )
        self._notify_tool_call_start(
            tool_call,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            on_tool_call_start=on_tool_call_start,
        )
        self._notify_tool_call_end(
            tool_call,
            result,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            duration_seconds=perf_counter() - started_at,
            on_tool_call_end=on_tool_call_end,
        )
        self._record_tool_response(
            tool_call,
            result,
            status="failed",
            duration_seconds=0,
        )

    def _reject_reserved_historical_argument(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
        on_tool_call_start: ToolCallStartCallback | None,
        on_tool_call_end: ToolCallEndCallback | None,
    ) -> None:
        """Reject model attempts to forge a host-only history marker."""
        started_at = perf_counter()
        result = ToolExecutionResult(model_content=_reserved_historical_argument_message())
        self._notify_tool_call_start(
            tool_call,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            on_tool_call_start=on_tool_call_start,
        )
        self._notify_tool_call_end(
            tool_call,
            result,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            duration_seconds=perf_counter() - started_at,
            status="failed",
            on_tool_call_end=on_tool_call_end,
        )
        self._record_tool_response(
            tool_call,
            result,
            status="failed",
            duration_seconds=0,
        )

    def _execute_single(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
        on_tool_call_start: ToolCallStartCallback | None,
        on_tool_call_end: ToolCallEndCallback | None,
    ) -> None:
        decision = self._evaluate_permission(
            tool_call,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
        )
        self._record_permission_decision(tool_call, decision)
        execution_started_at: float | None = None
        if decision.action == PermissionAction.ALLOW:
            execution_started_at = perf_counter()
            self._record_tool_started(
                tool_call,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
            )
            self._notify_tool_call_start(
                tool_call,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
                on_tool_call_start=on_tool_call_start,
            )
        result = self._resolve_tool_call(tool_call, decision)
        duration_seconds = (
            perf_counter() - execution_started_at
            if execution_started_at is not None
            else 0
        )
        status = (
            "denied"
            if decision.action == PermissionAction.DENY
            else _tool_result_status(result.model_content)
        )
        self._notify_tool_call_end(
            tool_call,
            result,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            duration_seconds=duration_seconds,
            status=status,
            on_tool_call_end=on_tool_call_end,
        )
        self._record_tool_response(
            tool_call,
            result,
            status=status,
            duration_seconds=duration_seconds,
        )

    def _execute_parallel_batch(
        self,
        batch: list[tuple[int, ToolCall]],
        *,
        tool_round: int,
        total_calls: int,
        on_tool_call_start: ToolCallStartCallback | None,
        on_tool_call_end: ToolCallEndCallback | None,
    ) -> None:
        execution_started_at_by_index: dict[int, float] = {}
        decision_by_index: dict[int, PermissionDecision] = {}
        for call_index, tool_call in batch:
            decision = self._evaluate_permission(
                tool_call,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
            )
            decision_by_index[call_index] = decision
            self._record_permission_decision(tool_call, decision)

        results_by_index: dict[int, ToolExecutionResult] = {}
        with ThreadPoolExecutor(
            max_workers=len(batch),
            thread_name_prefix="harnessed-coder-tool",
        ) as executor:
            futures: dict[Future[ToolExecutionResult], tuple[int, ToolCall]] = {}
            for call_index, tool_call in batch:
                decision = decision_by_index[call_index]
                if decision.action == PermissionAction.ALLOW:
                    execution_started_at_by_index[call_index] = perf_counter()
                    self._record_tool_started(
                        tool_call,
                        tool_round=tool_round,
                        call_index=call_index,
                        total_calls=total_calls,
                    )
                    self._notify_tool_call_start(
                        tool_call,
                        tool_round=tool_round,
                        call_index=call_index,
                        total_calls=total_calls,
                        on_tool_call_start=on_tool_call_start,
                    )
                future = executor.submit(
                    copy_context().run,
                    self._resolve_tool_call,
                    tool_call,
                    decision,
                )
                futures[future] = (call_index, tool_call)
            for future in as_completed(futures):
                call_index, tool_call = futures[future]
                result = future.result()
                results_by_index[call_index] = result
                decision = decision_by_index[call_index]
                status = (
                    "denied"
                    if decision.action == PermissionAction.DENY
                    else _tool_result_status(result.model_content)
                )
                self._notify_tool_call_end(
                    tool_call,
                    result,
                    tool_round=tool_round,
                    call_index=call_index,
                    total_calls=total_calls,
                    duration_seconds=(
                        perf_counter() - execution_started_at_by_index[call_index]
                        if call_index in execution_started_at_by_index
                        else 0
                    ),
                    status=status,
                    on_tool_call_end=on_tool_call_end,
                )

        # Tool-end callbacks fire as each worker completes, but message history
        # must follow the model's original tool_call order for Chat Completions
        # tool_call/tool_result pairing.
        for call_index, tool_call in batch:
            result = results_by_index[call_index]
            decision = decision_by_index[call_index]
            status = (
                "denied"
                if decision.action == PermissionAction.DENY
                else _tool_result_status(result.model_content)
            )
            self._record_tool_response(
                tool_call,
                result,
                status=status,
                duration_seconds=(
                    perf_counter() - execution_started_at_by_index[call_index]
                    if call_index in execution_started_at_by_index
                    else 0
                ),
            )

    def _evaluate_permission(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
    ) -> PermissionDecision:
        decision = self._tools.evaluate_permission(tool_call.name, tool_call.arguments)
        if decision.action != PermissionAction.ASK:
            return decision

        request = PermissionApprovalRequest(
            tool_name=tool_call.name,
            arguments=tool_call.arguments.copy(),
            decision=decision,
            task_context=(
                self._permission_context_provider()
                if self._permission_context_provider is not None
                else ""
            ),
        )
        review = self._review_permission(
            request,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
        )
        if review is not None and review.action == PermissionReviewAction.ALLOW:
            return PermissionDecision(
                PermissionAction.ALLOW,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution="auto_approved",
                resolution_reason=review.reason,
            )
        if review is not None and review.action == PermissionReviewAction.DENY:
            return PermissionDecision(
                PermissionAction.DENY,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution="auto_denied",
                resolution_reason=review.reason,
            )

        if review is not None:
            request = PermissionApprovalRequest(
                tool_name=request.tool_name,
                arguments=request.arguments,
                decision=request.decision,
                task_context=request.task_context,
                review=review,
            )
        if self._permission_approver is None:
            return PermissionDecision(
                PermissionAction.DENY,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution=(
                    "auto_abstained" if review is not None else "approval_unavailable"
                ),
                resolution_reason=review.reason if review is not None else None,
            )

        if self._on_permission_requested is not None:
            self._on_permission_requested(tool_call)
        try:
            approved = self._permission_approver(request)
        except (EOFError, KeyboardInterrupt):
            return PermissionDecision(
                PermissionAction.DENY,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution="approval_interrupted",
                resolution_reason=review.reason if review is not None else None,
            )
        except Exception:
            logger.exception("Permission approval failed: tool=%s", tool_call.name)
            return PermissionDecision(
                PermissionAction.DENY,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution="approval_error",
                resolution_reason=review.reason if review is not None else None,
            )

        if not isinstance(approved, bool):
            logger.error(
                "Permission approver returned a non-boolean result: tool=%s type=%s",
                tool_call.name,
                type(approved).__name__,
            )
            return PermissionDecision(
                PermissionAction.DENY,
                decision.reason,
                initial_action=PermissionAction.ASK,
                resolution="approval_error",
                resolution_reason=review.reason if review is not None else None,
            )

        suffix = "_after_abstain" if review is not None else ""
        return PermissionDecision(
            PermissionAction.ALLOW if approved else PermissionAction.DENY,
            decision.reason,
            initial_action=PermissionAction.ASK,
            resolution=("user_approved" if approved else "user_denied") + suffix,
            resolution_reason=review.reason if review is not None else None,
        )

    def _review_permission(
        self,
        request: PermissionApprovalRequest,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
    ) -> PermissionReview | None:
        if self._permission_reviewer is None:
            return None
        started_at = perf_counter()
        self._report_permission_review(
            PermissionReviewNotice(
                stage="started",
                tool_name=request.tool_name,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
            )
        )
        try:
            review = self._permission_reviewer(request)
        except Exception as exc:
            logger.exception("Automatic permission reviewer failed: tool=%s", request.tool_name)
            review = PermissionReview(
                PermissionReviewAction.ABSTAIN,
                f"automatic reviewer failed: {type(exc).__name__}",
            )
        if not isinstance(review, PermissionReview):
            logger.error(
                "Permission reviewer returned an invalid result: tool=%s type=%s",
                request.tool_name,
                type(review).__name__,
            )
            review = PermissionReview(
                PermissionReviewAction.ABSTAIN,
                "automatic reviewer returned an invalid result",
            )
        self._report_permission_review(
            PermissionReviewNotice(
                stage="completed",
                tool_name=request.tool_name,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
                action=review.action,
                duration_ms=round((perf_counter() - started_at) * 1000),
            )
        )
        return review

    def _report_permission_review(self, notice: PermissionReviewNotice) -> None:
        reporter = self._permission_review_reporter
        if reporter is not None:
            reporter(notice)

    def _resolve_tool_call(
        self,
        tool_call: ToolCall,
        decision: PermissionDecision,
    ) -> ToolExecutionResult:
        logger.debug(
            "Executing tool call: name=%s argument_keys=%s permission=%s",
            tool_call.name,
            sorted(tool_call.arguments.keys()),
            decision.action.value,
        )
        result = _as_execution_result(
            self._tools._apply_permission_decision(
                tool_call.name,
                tool_call.arguments,
                decision,
            )
        )
        logger.debug(
            "Tool call completed: name=%s result_chars=%s",
            tool_call.name,
            len(result.model_content),
        )
        return result

    def _notify_tool_call_start(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
        on_tool_call_start: ToolCallStartCallback | None,
    ) -> None:
        if on_tool_call_start:
            on_tool_call_start(
                tool_call.name,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
            )

    def _notify_tool_call_end(
        self,
        tool_call: ToolCall,
        result: ToolExecutionResult,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
        duration_seconds: float,
        status: str | None = None,
        on_tool_call_end: ToolCallEndCallback | None,
    ) -> None:
        if on_tool_call_end:
            on_tool_call_end(
                tool_call.name,
                tool_round=tool_round,
                call_index=call_index,
                total_calls=total_calls,
                result_chars=len(result.model_content),
                duration_seconds=duration_seconds,
                status=status or _tool_result_status(result.model_content),
            )

    def _record_tool_started(
        self,
        tool_call: ToolCall,
        *,
        tool_round: int,
        call_index: int,
        total_calls: int,
    ) -> None:
        self._on_tool_started(
            tool_call,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
        )

    def _record_permission_decision(
        self,
        tool_call: ToolCall,
        decision: PermissionDecision,
    ) -> None:
        if self._on_permission_decision is not None:
            self._on_permission_decision(tool_call, decision)

    def _record_tool_response(
        self,
        tool_call: ToolCall,
        result: ToolExecutionResult,
        *,
        status: str,
        duration_seconds: float,
    ) -> None:
        self._on_tool_response(
            tool_call,
            result,
            status=status,
            duration_seconds=duration_seconds,
        )


def _tool_result_status(result: str) -> str:
    normalized = result.lstrip().lower()
    if normalized.startswith(("error:", "failed:")):
        return "failed"
    if normalized.startswith(("cancelled:", "canceled:")):
        return "cancelled"
    return "done"
