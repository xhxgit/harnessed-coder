"""Interactive REPL runtime and Agent target switching."""

from __future__ import annotations

import logging
import sys
from threading import RLock
from time import perf_counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TextIO

from ..agent import Agent
from ..constants import DEFAULT_SESSION_NAME
from ..context import ContextCompressor, ToolBatchSummaryFunction, ToolBatchSummaryNotice
from ..memory import MemoryManager
from ..mcp_client import McpArtifactStore, McpConnections
from ..session.usage import session_usage
from ..session import ConversationSession
from ..session import resolve_session_trace_path
from ..shell import select_shell
from .approval import bind_permission_approver
from .commands import ReplState, handle_repl_command, resolve_workspace_display_name
from .input import CallableReplInput, PromptToolkitReplInput
from .rendering import (
    ContextCompressionRenderer,
    ReplTurnRenderer,
    clear_terminal_output,
    write_command_error,
    write_repl_error,
    write_session_history,
    write_startup_detail,
    write_startup_hint,
    write_system_notice,
    write_system_notice_start,
    write_user_input_header,
)

logger = logging.getLogger(__name__)

AgentFactory = Callable[..., Agent]


def run_repl(
    *,
    agent_factory: AgentFactory,
    workspace_root: Path | None = None,
    workspace_name: str | None = None,
    data_dir: Path | None = None,
    workspace_data_dir: Path | None = None,
    log_file: Path | None = None,
    trace_file: Path | None = None,
    session_file: Path | None = None,
    session_name: str = DEFAULT_SESSION_NAME,
    run_id: str | None = None,
    context_compressor: ContextCompressor | None = None,
    mcp_connections: McpConnections | None = None,
    mcp_artifact_store: McpArtifactStore | None = None,
    automatic_memory_extraction: bool = False,
    tool_batch_summary_factory: Callable[[str], ToolBatchSummaryFunction] | None = None,
    model: str,
    input_func: Callable[[str], str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> None:
    """Run an interactive read-eval-print loop for one agent session."""
    tool_batch_notice_writer = _ToolBatchSummaryNoticeWriter(output)
    compression_renderer = ContextCompressionRenderer(output)

    def create_repl_agent(
        target_workspace_root: Path | None,
        *,
        target_workspace_name: str | None,
        target_session_name: str,
        target_session_file: Path | None,
        target_model: str,
        target_memory_manager: MemoryManager | None,
    ) -> Agent:
        session = (
            ConversationSession.open(target_session_file)
            if target_session_file is not None
            else ConversationSession()
        )
        target_trace_path = (
            resolve_session_trace_path(
                target_workspace_root,
                workspace_name=target_workspace_name,
                session_name=target_session_name,
                data_dir=data_dir,
            )
            if target_workspace_root is not None and data_dir is not None
            else None
        )
        logger.info(
            "Session opened: workspace=%s session=%s path=%s messages=%s",
            resolve_workspace_display_name(
                target_workspace_root,
                target_workspace_name,
            ),
            target_session_name,
            session.path,
            session.message_count(),
        )
        agent = agent_factory(
            target_workspace_root,
            session=session,
            context_compressor=context_compressor,
            memory_manager=target_memory_manager,
            data_dir=data_dir,
            workspace_name=target_workspace_name,
            session_name=target_session_name,
            run_id=run_id,
            trace_path=target_trace_path,
            mcp_connections=mcp_connections,
            mcp_artifact_store=mcp_artifact_store,
            tool_batch_summary_function=(
                tool_batch_summary_factory(target_model)
                if tool_batch_summary_factory is not None
                else None
            ),
            model=target_model,
        )
        set_reporter = getattr(agent, "set_context_compression_reporter", None)
        if callable(set_reporter):
            set_reporter(compression_renderer.write)
        set_tool_batch_reporter = getattr(agent, "set_tool_batch_summary_reporter", None)
        if callable(set_tool_batch_reporter):
            set_tool_batch_reporter(tool_batch_notice_writer.write)
        return agent

    repl_input = (
        CallableReplInput(input_func)
        if input_func is not None
        else PromptToolkitReplInput(data_dir=data_dir)
    )
    state = ReplState(
        workspace_root=workspace_root,
        workspace_name=workspace_name,
        data_dir=data_dir,
        workspace_data_dir=workspace_data_dir,
        session_name=session_name,
        session_file=session_file,
        context_compressor=context_compressor,
        model=model,
    )
    active_memory_manager = _create_memory_manager(state)
    active_memory_target = _memory_target(state)
    active_agent = create_repl_agent(
        workspace_root,
        target_workspace_name=state.workspace_name,
        target_session_name=state.session_name,
        target_session_file=state.session_file,
        target_model=state.model,
        target_memory_manager=active_memory_manager,
    )
    clear_terminal_output(output, interactive_only=True)
    _write_startup_banner(
        state,
        repl_input=repl_input,
        output=output,
        log_file=log_file,
        trace_file=trace_file,
        mcp_connections=mcp_connections,
        automatic_memory_extraction=automatic_memory_extraction,
    )
    write_session_history(output, active_agent.session.snapshot())
    bind_permission_approver(active_agent, repl_input=repl_input, output=output)

    while True:
        write_user_input_header(output)
        try:
            user_input = repl_input.read("  ").strip()
        except EOFError:
            output.write("\n")
            output.flush()
            return
        except KeyboardInterrupt:
            output.write("\n")
            output.flush()
            continue

        if not user_input:
            continue
        if user_input == "/exit":
            return
        if user_input.startswith("/"):
            # REPL commands are local control-plane actions. They can replace
            # the active Agent after workspace/session/model switches, but they
            # are not appended to conversation history.
            try:
                with session_usage(active_agent.session):
                    command_result = handle_repl_command(
                        user_input,
                        state=state,
                        current_agent=active_agent,
                        memory_manager=active_memory_manager,
                        compression_reporter=compression_renderer.write,
                        output=output,
                    )
                current_memory_target = _memory_target(state)
                if current_memory_target != active_memory_target:
                    active_memory_manager = _create_memory_manager(state)
                    active_memory_target = current_memory_target
                if command_result.refresh_screen:
                    clear_terminal_output(output)
                    _write_startup_banner(
                        state,
                        repl_input=repl_input,
                        output=output,
                        log_file=log_file,
                        trace_file=trace_file,
                        mcp_connections=mcp_connections,
                        automatic_memory_extraction=automatic_memory_extraction,
                    )
                    write_session_history(output, active_agent.session.snapshot())
                if command_result.rebuild_agent:
                    active_agent = create_repl_agent(
                        state.workspace_root,
                        target_workspace_name=state.workspace_name,
                        target_session_name=state.session_name,
                        target_session_file=state.session_file,
                        target_model=state.model,
                        target_memory_manager=active_memory_manager,
                    )
                    bind_permission_approver(
                        active_agent,
                        repl_input=repl_input,
                        output=output,
                    )
                    if command_result.replay_history:
                        write_session_history(output, active_agent.session.snapshot())
            except Exception as exc:
                logger.exception("REPL command failed: command=%s", user_input.split()[0])
                write_command_error(output, f"{type(exc).__name__}: {exc}")
                continue
            continue

        renderer = ReplTurnRenderer(output)
        set_model_request_reporter = getattr(
            active_agent,
            "set_model_request_reporter",
            None,
        )
        if callable(set_model_request_reporter):
            set_model_request_reporter(renderer.on_model_request)
        set_permission_review_reporter = getattr(
            active_agent,
            "set_permission_review_reporter",
            None,
        )
        if callable(set_permission_review_reporter):
            set_permission_review_reporter(renderer.on_permission_review)
        turn_started_at = perf_counter()
        try:
            turn_result = active_agent.chat(
                user_input,
                on_text_delta=renderer.on_text_delta,
                on_activity_delta=renderer.on_activity_delta,
                on_tool_call_start=renderer.on_tool_call_start,
                on_tool_call_end=renderer.on_tool_call_end,
            )
        except Exception as exc:
            logger.exception("Agent chat failed")
            write_repl_error(error_output, exc)
            continue
        finally:
            renderer.finish_waiting_status()
            if callable(set_model_request_reporter):
                set_model_request_reporter(None)
            if callable(set_permission_review_reporter):
                set_permission_review_reporter(None)

        renderer.write_final_response(turn_result.content)
        analysis = active_agent.analyze_context()
        renderer.write_turn_summary(
            duration_seconds=perf_counter() - turn_started_at,
            current_context_tokens=analysis.current_tokens.total,
            context_max_tokens=(
                state.context_compressor.max_tokens
                if state.context_compressor is not None
                else None
            ),
            history_tokens=analysis.history_tokens,
        )
        if automatic_memory_extraction and active_memory_manager is not None:
            memory_started_at = perf_counter()
            _trace_memory_extraction(active_agent, status="started")
            try:
                with session_usage(active_agent.session):
                    memory_results = active_memory_manager.extract_from_conversation(
                        active_agent.session.snapshot(),
                        conversation_view=turn_result.conversation_view,
                    )
            except Exception as exc:
                logger.exception("Automatic memory extraction failed")
                _trace_memory_extraction(
                    active_agent,
                    status="failed",
                    duration_ms=round((perf_counter() - memory_started_at) * 1000),
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                write_system_notice(
                    output,
                    f"Automatic memory extraction failed: "
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                changed_results = [
                    result
                    for result in memory_results
                    if getattr(result, "action", None) in {"added", "replaced"}
                ]
                _trace_memory_extraction(
                    active_agent,
                    status="completed",
                    result="updated" if changed_results else "no_change",
                    updates=len(changed_results),
                    duration_ms=round((perf_counter() - memory_started_at) * 1000),
                )
                for result in changed_results:
                    record = getattr(result, "record", None)
                    if record is not None:
                        _trace_memory_updated(
                            active_agent,
                            action=result.action,
                            memory_id=record.id,
                            scope=record.scope,
                        )
                _write_automatic_memory_results(output, memory_results)


def _trace_memory_extraction(agent: Agent, *, status: str, **data: object) -> None:
    recorder = getattr(agent, "trace_memory_extraction", None)
    if callable(recorder):
        recorder(status=status, **data)


class _ToolBatchSummaryNoticeWriter:
    """Report cross-turn waits and background summary failures."""

    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._lock = RLock()
        self._pending: tuple[int, int] | None = None

    def write(self, notice: ToolBatchSummaryNotice) -> None:
        key = (notice.turn_number, notice.batch_index)
        with self._lock:
            if notice.stage == "waiting":
                write_system_notice_start(
                    self._output,
                    "Waiting for prior tool-batch summaries through "
                    f"turn {notice.turn_number}, batch #{notice.batch_index}...",
                )
                self._pending = key
                return
            if notice.stage == "completed":
                message = (
                    " done; waited "
                    f"{_format_elapsed_ms(notice.wait_duration_ms)}."
                )
                if self._pending == key:
                    self._output.write(f"{message}\n")
                    self._output.flush()
                else:
                    write_system_notice(self._output, message.lstrip())
                self._pending = None
                return
            if notice.stage != "failed":
                return
            if self._pending == key:
                self._output.write("\n")
                self._output.flush()
                self._pending = None
            message = (
                "Tool-batch summary failed for "
                f"turn {notice.turn_number}, batch #{notice.batch_index}."
            )
            if notice.error:
                message += f" Error: {notice.error}"
            write_system_notice(self._output, message)


def _format_elapsed_ms(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "unknown"
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    return f"{duration_ms / 1000:.1f}s"


def _trace_memory_updated(
    agent: Agent,
    *,
    action: str,
    memory_id: str,
    scope: str,
) -> None:
    recorder = getattr(agent, "trace_memory_updated", None)
    if callable(recorder):
        recorder(action=action, memory_id=memory_id, scope=scope)


def _create_memory_manager(
    state: ReplState,
) -> MemoryManager | None:
    if state.data_dir is None:
        return None
    return MemoryManager(
        state.data_dir,
        workspace_data_dir=state.workspace_data_dir,
        model=state.model,
    )


def _memory_target(
    state: ReplState,
) -> tuple[Path | None, Path | None, str]:
    return state.data_dir, state.workspace_data_dir, state.model


def _write_automatic_memory_results(
    output: TextIO,
    results: Iterable[object],
) -> None:
    for result in results:
        action = getattr(result, "action", None)
        record = getattr(result, "record", None)
        if action == "added" and record is not None:
            write_system_notice(
                output,
                f"Remembered {record.scope} memory "
                f"{record.id}: {record.content}",
            )
        elif action == "replaced" and record is not None:
            previous = getattr(result, "previous", None)
            reason = getattr(result, "reason", None)
            lines = [
                f"Updated {record.scope} memory "
                f"{record.id}: {record.content}",
            ]
            if previous is not None:
                lines.append(f"Replaced previous content: {previous.content}")
            if reason:
                lines.append(f"Reason: {reason}")
            write_system_notice(output, "\n".join(lines))


def _write_startup_banner(
    state: ReplState,
    *,
    repl_input: CallableReplInput | PromptToolkitReplInput,
    output: TextIO,
    log_file: Path | None,
    trace_file: Path | None,
    mcp_connections: McpConnections | None,
    automatic_memory_extraction: bool,
) -> None:
    workspace_display = resolve_workspace_display_name(
        state.workspace_root,
        state.workspace_name,
    )
    output.write("● Harnessed Coder\n")
    output.write(
        f"  {state.model} · {workspace_display} · {state.session_name}\n"
    )
    if state.workspace_root is not None:
        write_startup_detail(
            output,
            "Workspace",
            str(state.workspace_root),
            subdued_value=True,
        )
        write_startup_detail(output, "Bash", f"enabled ({select_shell()})")
    if state.data_dir is not None:
        write_startup_detail(
            output,
            "Data dir",
            str(state.data_dir),
            subdued_value=True,
        )
    if state.workspace_data_dir is not None:
        write_startup_detail(
            output,
            "Workspace data",
            str(state.workspace_data_dir),
            subdued_value=True,
        )
    if not automatic_memory_extraction:
        output.write("  Automatic memory extraction: disabled\n")
    if log_file is not None:
        write_startup_detail(
            output,
            "Log file",
            str(log_file.resolve()),
            subdued_value=True,
        )
    if trace_file is not None:
        write_startup_detail(
            output,
            "JSONL trace",
            str(trace_file.resolve()),
            subdued_value=True,
        )
    if state.session_file is not None:
        write_startup_detail(
            output,
            "Session file",
            str(state.session_file),
            subdued_value=True,
        )
    if mcp_connections is not None:
        snapshot = mcp_connections.snapshot()
        output.write(
            f"  MCP: {len(snapshot.tools)} tools from "
            f"{snapshot.connected_servers}/{snapshot.configured_servers} servers\n"
        )
        for diagnostic in snapshot.diagnostics:
            output.write(
                f"  MCP warning [{diagnostic.server_name}]: "
                f"{diagnostic.message}\n"
            )
    editing_hint = (
        "Enter submits. Alt+Enter inserts a newline; pasted multiline text stays together. "
        if not isinstance(repl_input, PromptToolkitReplInput)
        or repl_input.enhanced_editing
        else "Basic input mode: this host has no interactive terminal. "
    )
    write_startup_hint(
        output,
        f"{editing_hint}Type '/exit' to leave. Type '/help' for CLI commands.",
    )
