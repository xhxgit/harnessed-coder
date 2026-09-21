"""Streaming output rendering for the REPL."""

from __future__ import annotations

import json
from collections.abc import Callable
from threading import Event, RLock, Thread, current_thread
from time import perf_counter
from typing import Any, TextIO

from ..agent import ContextCompressionNotice
from ..permissions import PermissionReviewAction, PermissionReviewNotice
from ..session import conversation_turns


_CLEAR_LINE = "\r\x1b[2K"
_COLORS = {
    "thinking": "\x1b[2m",
    "muted": "\x1b[90m",
    "system": "\x1b[36m",
    "running": "\x1b[36m",
    "done": "\x1b[32m",
    "failed": "\x1b[31m",
    "cancelled": "\x1b[33m",
    "denied": "\x1b[31m",
    "warning": "\x1b[33m",
}
_RESET = "\x1b[0m"
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SPINNER_INTERVAL_SECONDS = 0.1


class ReplTurnRenderer:
    """Render streaming model activity for one REPL user turn."""

    def __init__(
        self,
        output: TextIO,
        *,
        clock: Callable[[], float] = perf_counter,
        interactive: bool | None = None,
    ) -> None:
        self.output = output
        self.streamed = False
        self._clock = clock
        self._interactive = _is_interactive(output) if interactive is None else interactive
        self._thinking_started_at: float | None = None
        self._thinking_chars = 0
        self._thinking_stop: Event | None = None
        self._thinking_thread: Thread | None = None
        self._last_chat_char: str | None = None
        self._assistant_block_started = False
        self._assistant_line_start = True
        self._tool_started_at: dict[tuple[str, int | None, int | None], float] = {}
        self._shown_tool_rounds: set[int] = set()
        self._completed_tool_calls = 0
        self._output_lock = RLock()
        self._spinner_lock = RLock()
        self._spinner_stop: Event | None = None
        self._spinner_thread: Thread | None = None

    def on_model_request(self, stage: str) -> None:
        """Show activity until the current model request yields its first event."""
        if stage == "started":
            self._assistant_block_started = False
            self._assistant_line_start = True
            self._start_waiting_status()
        elif stage == "finished":
            self.finish_waiting_status()

    def on_permission_review(self, notice: PermissionReviewNotice) -> None:
        """Render the otherwise-hidden automatic permission-review wait."""
        self.finish_waiting_status()
        self.finish_thinking_status()
        self._ensure_line_boundary()
        self._write_tool_round_header(notice.tool_round)
        if notice.stage == "started":
            self._write_tool_activity(
                "…",
                notice.tool_name,
                tool_round=notice.tool_round,
                call_index=notice.call_index,
                total_calls=notice.total_calls,
                detail="reviewing permission",
                style="warning",
            )
        elif notice.stage == "completed":
            labels = {
                PermissionReviewAction.ALLOW: "permission approved",
                PermissionReviewAction.DENY: "permission denied",
                PermissionReviewAction.ABSTAIN: "permission review abstained",
            }
            label = (
                labels.get(notice.action, "permission review completed")
                if notice.action is not None
                else "permission review completed"
            )
            details = [label]
            if notice.duration_ms is not None:
                details.append(_format_duration(notice.duration_ms / 1000))
            if notice.action is PermissionReviewAction.ALLOW:
                symbol, style = "✓", "done"
            elif notice.action is PermissionReviewAction.DENY:
                symbol, style = "✗", "denied"
            elif notice.action is PermissionReviewAction.ABSTAIN:
                symbol, style = "!", "warning"
            else:
                symbol, style = "·", "system"
            self._write_tool_activity(
                symbol,
                notice.tool_name,
                tool_round=notice.tool_round,
                call_index=notice.call_index,
                total_calls=notice.total_calls,
                detail="  ".join(details),
                style=style,
            )
        self._last_chat_char = "\n"
        self.output.flush()

    def on_text_delta(self, delta: str) -> None:
        self.finish_waiting_status()
        self.finish_thinking_status()
        self.streamed = True
        self._start_assistant_block()
        rendered = self._indent_assistant_delta(delta)
        self.output.write(rendered)
        if rendered:
            self._last_chat_char = rendered[-1]
        self.output.flush()

    def on_activity_delta(self, kind: str, char_count: int) -> None:
        self.finish_waiting_status()
        if kind != "reasoning":
            if kind == "tool_call":
                self.finish_thinking_status()
            return
        if self._thinking_started_at is None:
            self._thinking_started_at = self._clock()
        self._thinking_chars += char_count
        if self._interactive and self._thinking_stop is None:
            self._start_thinking_status()

    def on_tool_call_start(
        self,
        tool_name: str,
        *,
        tool_round: int | None = None,
        call_index: int | None = None,
        total_calls: int | None = None,
    ) -> None:
        self.finish_waiting_status()
        self.finish_thinking_status()
        self._ensure_line_boundary()
        self._write_tool_round_header(tool_round)
        self._write_tool_activity(
            "›",
            tool_name,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            detail="running",
            style="running",
        )
        self._last_chat_char = "\n"
        self._tool_started_at[(tool_name, tool_round, call_index)] = self._clock()
        self.output.flush()

    def on_tool_call_end(
        self,
        tool_name: str,
        *,
        tool_round: int | None = None,
        call_index: int | None = None,
        total_calls: int | None = None,
        result_chars: int | None = None,
        duration_seconds: float | None = None,
        status: str = "done",
    ) -> None:
        self._write_tool_round_header(tool_round)
        started_at = self._tool_started_at.pop((tool_name, tool_round, call_index), None)
        if duration_seconds is None and started_at is not None:
            duration_seconds = self._clock() - started_at
        details = []
        if duration_seconds is not None:
            details.append(_format_duration(duration_seconds))
        if result_chars is not None:
            details.append(f"{result_chars:,} chars")
        status_label = (
            status if status in {"done", "failed", "cancelled", "denied"} else "done"
        )
        symbol = {
            "done": "✓",
            "failed": "✗",
            "cancelled": "!",
            "denied": "✗",
        }[status_label]
        detail = "  ".join([status_label, *details])
        self._write_tool_activity(
            symbol,
            tool_name,
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
            detail=detail,
            style=status_label,
        )
        self._completed_tool_calls += 1
        self._last_chat_char = "\n"
        self.output.flush()

    def finish_thinking_status(self) -> None:
        if self._thinking_started_at is None:
            return
        stop = self._thinking_stop
        thread = self._thinking_thread
        self._thinking_stop = None
        self._thinking_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=_SPINNER_INTERVAL_SECONDS * 2)
        if self._interactive:
            with self._output_lock:
                self.output.write(
                    f"{_CLEAR_LINE}"
                    f"{_styled(self._thinking_status_text(), 'thinking', True)}\n"
                )
                self.output.flush()
        else:
            self._write_thinking_status()
        self._thinking_started_at = None
        self._thinking_chars = 0
        self._last_chat_char = "\n"

    def finish_waiting_status(self) -> None:
        """Stop and erase the interactive pre-response spinner, if active."""
        with self._spinner_lock:
            stop = self._spinner_stop
            thread = self._spinner_thread
            if stop is None:
                return
            self._spinner_stop = None
            self._spinner_thread = None
            stop.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=_SPINNER_INTERVAL_SECONDS * 2)
        with self._output_lock:
            self.output.write(_CLEAR_LINE)
            self.output.flush()

    def write_final_response(self, response: str) -> None:
        self.finish_waiting_status()
        self.finish_thinking_status()
        if response and not self.streamed:
            self._start_assistant_block()
            rendered = self._indent_assistant_delta(response)
            self.output.write(rendered)
            self._last_chat_char = rendered[-1]
        if response or self.streamed:
            self.output.write("\n")
            self.output.flush()

    def write_turn_summary(
        self,
        *,
        duration_seconds: float,
        current_context_tokens: int | None = None,
        context_max_tokens: int | None = None,
        history_tokens: int | None = None,
    ) -> None:
        """Write a compact, user-visible summary after a completed turn."""
        tool_label = "tool" if self._completed_tool_calls == 1 else "tools"
        details = [
            f"{self._completed_tool_calls} {tool_label}",
            _format_duration(duration_seconds),
        ]
        if current_context_tokens is not None:
            context = f"context {current_context_tokens:,}"
            if context_max_tokens is not None:
                context += f"/{context_max_tokens:,}"
            details.append(context)
        if history_tokens is not None:
            details.append(f"history {history_tokens:,}")
        line = f"─ Turn complete · {' · '.join(details)}"
        self.output.write(f"{_styled(line, 'thinking', self._interactive)}\n")
        self.output.flush()

    def _write_thinking_status(self) -> None:
        status = self._thinking_status_text()
        with self._output_lock:
            if self._interactive:
                self.output.write(f"{_CLEAR_LINE}{_styled(status, 'thinking', True)}")
            else:
                self.output.write(f"{status}\n")
            self.output.flush()
        self._last_chat_char = "\n"

    def _thinking_status_text(self, frame: str | None = None) -> str:
        assert self._thinking_started_at is not None
        elapsed = self._clock() - self._thinking_started_at
        symbol = frame if frame is not None else "✓"
        return f"{symbol} Thinking: {elapsed:.1f}s, {self._thinking_chars} chars"

    def _start_thinking_status(self) -> None:
        stop = Event()
        thread = Thread(
            target=self._animate_thinking_status,
            args=(stop,),
            name="model-thinking-spinner",
            daemon=True,
        )
        self._thinking_stop = stop
        self._thinking_thread = thread
        self._write_thinking_frame(_SPINNER_FRAMES[0])
        thread.start()

    def _animate_thinking_status(self, stop: Event) -> None:
        frame_index = 1
        while not stop.wait(_SPINNER_INTERVAL_SECONDS):
            if self._thinking_stop is not stop:
                return
            self._write_thinking_frame(
                _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]
            )
            frame_index += 1

    def _write_thinking_frame(self, frame: str) -> None:
        with self._output_lock:
            self.output.write(
                f"{_CLEAR_LINE}"
                f"{_styled(self._thinking_status_text(frame), 'thinking', True)}"
            )
            self.output.flush()
        self._last_chat_char = "\n"

    def _start_waiting_status(self) -> None:
        if not self._interactive:
            return
        self.finish_waiting_status()
        stop = Event()
        started_at = self._clock()
        thread = Thread(
            target=self._animate_waiting_status,
            args=(stop, started_at),
            name="model-wait-spinner",
            daemon=True,
        )
        with self._spinner_lock:
            self._spinner_stop = stop
            self._spinner_thread = thread
        self._write_waiting_frame(_SPINNER_FRAMES[0], started_at)
        thread.start()

    def _animate_waiting_status(self, stop: Event, started_at: float) -> None:
        frame_index = 1
        while not stop.wait(_SPINNER_INTERVAL_SECONDS):
            with self._spinner_lock:
                if self._spinner_stop is not stop:
                    return
            self._write_waiting_frame(
                _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)],
                started_at,
            )
            frame_index += 1

    def _write_waiting_frame(self, frame: str, started_at: float) -> None:
        elapsed = self._clock() - started_at
        status = f"{frame} Waiting for model: {elapsed:.1f}s"
        with self._output_lock:
            self.output.write(f"{_CLEAR_LINE}{_styled(status, 'thinking', True)}")
            self.output.flush()

    def _ensure_line_boundary(self) -> None:
        if self._last_chat_char is not None and self._last_chat_char != "\n":
            self.output.write("\n")
            self._last_chat_char = "\n"

    def _write_tool_round_header(self, tool_round: int | None) -> None:
        if tool_round is None or tool_round in self._shown_tool_rounds:
            return
        self._shown_tool_rounds.add(tool_round)
        line = f"Tool round {tool_round}"
        self.output.write(f"{_styled(line, 'system', self._interactive)}\n")

    def _start_assistant_block(self) -> None:
        if self._assistant_block_started:
            return
        self._ensure_line_boundary()
        self.output.write("\n")
        self.output.write(
            f"{_styled('◆ Assistant', 'done', self._interactive)}\n"
        )
        self._assistant_block_started = True
        self._assistant_line_start = True
        self._last_chat_char = "\n"

    def _indent_assistant_delta(self, delta: str) -> str:
        rendered: list[str] = []
        for character in delta:
            if self._assistant_line_start and character != "\n":
                rendered.append("  ")
                self._assistant_line_start = False
            rendered.append(character)
            if character == "\n":
                self._assistant_line_start = True
        return "".join(rendered)

    def _write_tool_activity(
        self,
        symbol: str,
        tool_name: str,
        *,
        tool_round: int | None,
        call_index: int | None,
        total_calls: int | None,
        detail: str,
        style: str,
    ) -> None:
        call_label = _tool_call_label(
            tool_round=tool_round,
            call_index=call_index,
            total_calls=total_calls,
        )
        identity = f"{tool_name:<20}"
        if call_label:
            identity += f" {call_label:>7}"
        line = f"  {symbol} {identity}  {detail}"
        self.output.write(f"{_styled(line.rstrip(), style, self._interactive)}\n")


def write_repl_error(output: TextIO, exc: Exception) -> None:
    """Write a concise turn error, using color only on an interactive terminal."""
    error_type = type(exc).__name__
    line = f"Error ({error_type}): {exc}"
    output.write(f"{_styled(line, 'failed', _is_interactive(output))}\n")
    output.flush()


def write_system_notice(output: TextIO, message: str) -> None:
    """Write a host-generated CLI notice that is distinct from model output."""
    notice = message.rstrip("\r\n")
    marker = _styled("[System]", "system", _is_interactive(output))
    output.write(f"{marker} {notice}\n")
    output.flush()


def write_system_notice_start(output: TextIO, message: str) -> None:
    """Start a host notice that a later state change will finish on the same line."""
    notice = message.rstrip("\r\n")
    marker = _styled("[System]", "system", _is_interactive(output))
    output.write(f"{marker} {notice}")
    output.flush()


def write_command_heading(output: TextIO, title: str) -> None:
    """Write the title of a local slash-command result."""
    output.write(f"{_styled(f'◇ {title}', 'system', _is_interactive(output))}\n")


def write_command_field(
    output: TextIO,
    label: str,
    value: object,
    *,
    subdued_value: bool = False,
) -> None:
    """Write one indented key/value field with stable non-TTY text."""
    interactive = _is_interactive(output)
    rendered_label = _styled(label, "system", interactive)
    value_text = (
        f"{value:,}"
        if isinstance(value, int) and not isinstance(value, bool)
        else str(value)
    )
    rendered_value = _styled(value_text, "muted", interactive and subdued_value)
    output.write(f"  {rendered_label}: {rendered_value}\n")


def write_command_item(output: TextIO, text: str, *, subdued: bool = False) -> None:
    """Write one item in a local command result list."""
    interactive = _is_interactive(output)
    style = "muted" if subdued else "system"
    marker = _styled("•", style, interactive)
    rendered = _styled(text, "muted", interactive and subdued)
    output.write(f"  {marker} {rendered}\n")


def write_command_entry(
    output: TextIO,
    name: str,
    description: str,
    *,
    width: int = 28,
) -> None:
    """Write an aligned command/menu entry."""
    interactive = _is_interactive(output)
    spacing = " " * max(2, width - len(name))
    rendered_name = _styled(f"{name}{spacing}", "system", interactive)
    rendered_description = _styled(description, "muted", interactive)
    output.write(f"  {rendered_name}{rendered_description}\n")


def write_command_note(output: TextIO, message: str) -> None:
    """Write secondary explanatory text for a local command."""
    line = f"  {message.rstrip(chr(13) + chr(10))}"
    output.write(f"{_styled(line, 'muted', _is_interactive(output))}\n")


def write_command_success(output: TextIO, message: str) -> None:
    """Write a successful local command outcome."""
    output.write(f"{_styled(f'✓ {message}', 'done', _is_interactive(output))}\n")
    output.flush()


def write_command_warning(output: TextIO, message: str) -> None:
    """Write local command guidance or a non-fatal empty state."""
    output.write(f"{_styled(f'! {message}', 'warning', _is_interactive(output))}\n")
    output.flush()


def write_command_error(output: TextIO, message: str) -> None:
    """Write a local command error without raising it."""
    output.write(f"{_styled(f'✗ {message}', 'failed', _is_interactive(output))}\n")
    output.flush()


def write_startup_detail(
    output: TextIO,
    label: str,
    value: str,
    *,
    subdued_value: bool = False,
) -> None:
    """Write one startup detail with a highlighted label and optional muted value."""
    interactive = _is_interactive(output)
    rendered_label = _styled(label, "system", interactive)
    rendered_value = (
        _styled(value, "muted", interactive) if subdued_value else value
    )
    output.write(f"  {rendered_label}: {rendered_value}\n")
    output.flush()


def write_startup_hint(output: TextIO, message: str) -> None:
    """Write secondary startup guidance without emphasizing it over runtime state."""
    line = f"  {message.rstrip('\r\n')}"
    output.write(f"{_styled(line, 'muted', _is_interactive(output))}\n")
    output.flush()


def write_user_input_header(output: TextIO) -> None:
    """Start one live user-input block using the same role style as history."""
    interactive = _is_interactive(output)
    role = _styled("› User", "system", interactive)
    output.write(f"\n{role}\n")
    output.flush()


class ContextCompressionRenderer:
    """Render one context-compression attempt as a compact live status."""

    def __init__(
        self,
        output: TextIO,
        *,
        clock: Callable[[], float] = perf_counter,
        interactive: bool | None = None,
    ) -> None:
        self.output = output
        self._clock = clock
        self._interactive = _is_interactive(output) if interactive is None else interactive
        self._lock = RLock()
        self._stop: Event | None = None
        self._thread: Thread | None = None
        self._started_at: float | None = None

    def write(self, notice: ContextCompressionNotice) -> None:
        if notice.stage == "started":
            self._start(notice)
            return
        self._finish(notice)

    def _start(self, notice: ContextCompressionNotice) -> None:
        self._stop_active_spinner()
        self._started_at = self._clock()
        if not self._interactive:
            return
        stop = Event()
        thread = Thread(
            target=self._animate,
            args=(stop, notice),
            name="context-compression-spinner",
            daemon=True,
        )
        self._stop = stop
        self._thread = thread
        self._write_frame(_SPINNER_FRAMES[0], notice)
        thread.start()

    def _finish(self, notice: ContextCompressionNotice) -> None:
        started_at = self._started_at
        self._stop_active_spinner()
        duration = None if started_at is None else self._clock() - started_at
        self._started_at = None
        line, style = _context_compression_result(notice, duration=duration)
        with self._lock:
            if self._interactive:
                self.output.write(_CLEAR_LINE)
            self.output.write(f"{_styled(line, style, self._interactive)}\n")
            self.output.flush()

    def _animate(self, stop: Event, notice: ContextCompressionNotice) -> None:
        frame_index = 1
        while not stop.wait(_SPINNER_INTERVAL_SECONDS):
            if self._stop is not stop:
                return
            self._write_frame(
                _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)],
                notice,
            )
            frame_index += 1

    def _write_frame(self, frame: str, notice: ContextCompressionNotice) -> None:
        started_at = self._started_at
        elapsed = 0.0 if started_at is None else self._clock() - started_at
        status = (
            f"{frame} Compressing context ({notice.trigger}): "
            f"{notice.before_tokens:,} tokens · {elapsed:.1f}s"
        )
        with self._lock:
            self.output.write(f"{_CLEAR_LINE}{_styled(status, 'thinking', True)}")
            self.output.flush()

    def _stop_active_spinner(self) -> None:
        stop = self._stop
        thread = self._thread
        self._stop = None
        self._thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=_SPINNER_INTERVAL_SECONDS * 2)


def _context_compression_result(
    notice: ContextCompressionNotice,
    *,
    duration: float | None,
) -> tuple[str, str]:
    second_pass = (
        "unknown"
        if notice.review_performed is None
        else "yes" if notice.review_performed else "no"
    )
    duration_text = "" if duration is None else f" · {_format_duration(duration)}"
    if notice.stage == "failed":
        return (
            f"✗ Context compression failed ({notice.trigger}): "
            f"{notice.before_tokens:,} tokens retained · "
            f"second pass: {second_pass}{duration_text}",
            "failed",
        )
    if not notice.omitted_count:
        tokens = notice.sent_tokens if notice.sent_tokens is not None else notice.before_tokens
        return (
            f"✓ Context already compact ({notice.trigger}): {tokens:,} tokens · "
            f"second pass: {second_pass}{duration_text}",
            "done",
        )
    sent_tokens = notice.sent_tokens if notice.sent_tokens is not None else 0
    sent_count = notice.sent_count if notice.sent_count is not None else 0
    headline = (
        f"✓ Context compressed ({notice.trigger}): "
        f"{notice.before_tokens:,} → {sent_tokens:,} history tokens · "
        f"{notice.before_count:,} → {sent_count:,} messages · "
        f"second pass: {second_pass}{duration_text}"
    )
    details: list[str] = []
    if notice.canonical_tokens is not None:
        canonical_count = notice.canonical_count or 0
        details.append(
            f"  Canonical history: {notice.canonical_tokens:,} tokens · "
            f"{canonical_count:,} messages"
        )
    breakdown = notice.breakdown
    if breakdown is not None:
        details.extend(
            [
                f"  Summary: {breakdown.summary_source_tokens:,} → "
                f"{breakdown.summary_tokens:,} tokens · "
                f"{breakdown.summary_source_count:,} source messages",
                f"  Recent complete turns: {breakdown.recent_turn_count:,} turns · "
                f"{breakdown.recent_message_count:,} messages · "
                f"{breakdown.recent_tokens:,} tokens",
                f"  Current turn preserved: {breakdown.current_turn_message_count:,} messages · "
                f"{breakdown.current_turn_tokens:,} tokens",
            ]
        )
    if notice.model_input_tokens is not None:
        model_input_count = notice.model_input_count or 0
        system_prompt_tokens = notice.system_prompt_tokens or 0
        details.append(
            f"  Final model input: {notice.model_input_tokens:,} tokens · "
            f"{model_input_count:,} messages · "
            f"system prompt {system_prompt_tokens:,} tokens"
        )
    return ("\n".join([headline, *details]), "done")


def clear_terminal_output(
    output: TextIO,
    *,
    interactive_only: bool = False,
) -> None:
    """Clear the viewport and scrollback, optionally only for an interactive TTY."""
    if interactive_only and not _is_interactive(output):
        return
    output.write("\x1b[2J\x1b[3J\x1b[H")
    output.flush()


def write_session_history(
    output: TextIO,
    messages: list[dict[str, Any]],
) -> None:
    """Replay the user-visible text of one canonical session."""
    visible_messages = [
        (role, text)
        for message in messages
        if (role := message.get("role")) in {"user", "assistant"}
        if (text := _visible_message_text(message))
    ]
    if not visible_messages:
        return

    interactive = _is_interactive(output)
    user_turns = len(conversation_turns(messages))
    turn_label = "user turn" if user_turns == 1 else "user turns"
    message_label = "message" if len(visible_messages) == 1 else "messages"
    header = (
        f"─ Restored conversation · {user_turns} {turn_label} · "
        f"{len(visible_messages)} visible {message_label}"
    )
    output.write(f"{_styled(header, 'muted', interactive)}\n")
    for role, text in visible_messages:
        label = "› User" if role == "user" else "◆ Assistant"
        style = "system" if role == "user" else "done"
        output.write(f"\n{_styled(label, style, interactive)}\n")
        output.write(f"{_indent_history_text(text)}\n")
    continuation = "━━ Continue conversation ━━━━━━━━━━━━━━━━━"
    output.write(f"\n{_styled(continuation, 'system', interactive)}\n")
    output.flush()


def _visible_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.rstrip("\r\n")
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str).rstrip("\r\n")


def _indent_history_text(text: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in text.splitlines())


def _tool_call_label(
    *,
    tool_round: int | None,
    call_index: int | None,
    total_calls: int | None,
) -> str:
    if tool_round is not None and call_index is not None and total_calls is not None:
        return f"{call_index}/{total_calls}"
    return ""


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _is_interactive(output: TextIO) -> bool:
    try:
        return output.isatty()
    except (AttributeError, OSError):
        return False


def _styled(text: str, style: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_COLORS[style]}{text}{_RESET}"
