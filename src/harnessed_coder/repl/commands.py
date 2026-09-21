"""REPL command handling for harnessed_coder."""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from ..agent import Agent, ContextCompressionNotice
from ..constants import WORKSPACE_METADATA_FILE_NAME
from ..context import ContextCompressionAnalysis, ContextCompressor
from ..memory import MemoryManager
from ..session import (
    resolve_session_path,
    resolve_session_trace_path,
    resolve_workspace_data_dir,
    write_workspace_metadata,
)
from ..tools import SkillTool
from ..user_config import resolve_user_config_path, set_configured_model
from .rendering import (
    write_command_error,
    write_command_entry,
    write_command_field,
    write_command_heading,
    write_command_item,
    write_command_note,
    write_command_success,
    write_command_warning,
)


logger = logging.getLogger(__name__)

_HELP_ENTRIES = (
    ("/exit", "Exit the CLI."),
    ("/refresh", "Clear terminal output and redraw this session."),
    ("/workspace", "Show current workspace."),
    ("/workspace list", "List saved workspace slots."),
    ("/workspace <slot|path>", "Switch to a saved slot or directory."),
    ("/workspace <path> <name>", "Switch root and use/create a named slot."),
    ("/session", "Show current session."),
    ("/session clear", "Clear current session history."),
    ("/session list", "List sessions in the current workspace."),
    ("/session rename <name>", "Rename the current session."),
    ("/session <name>", "Switch to a session in the current workspace."),
    ("/model", "Show current model."),
    ("/model <name>", "Switch model and save it to user config."),
    ("/status", "Show workspace, session, model, and context status."),
    ("/tokens", "Show context and session API token usage."),
    ("/history search <query>", "Search messages in the current session."),
    ("/memory add --user <text>", "Add a user-scoped long-term memory."),
    ("/memory add --workspace <text>", "Add a workspace-scoped long-term memory."),
    ("/memory list [scope]", "List memories (--user or --workspace)."),
    ("/memory search [scope] <query>", "Search manually saved memories."),
    ("/memory forget <id>", "Delete a memory by id."),
    ("/skills", "List discovered user/workspace Skills."),
    ("/skills reload", "Reload all Skills from disk."),
    ("/compact", "Prepare and cache compact API context now."),
)


MemoryCommandScope = Literal["user", "workspace"]


@dataclass
class ReplState:
    """Mutable REPL target state."""

    workspace_root: Path | None
    workspace_name: str | None
    data_dir: Path | None
    workspace_data_dir: Path | None
    session_name: str
    session_file: Path | None
    context_compressor: ContextCompressor | None
    model: str


@dataclass(frozen=True)
class ReplCommandResult:
    """Runtime actions requested by one completed REPL command."""

    rebuild_agent: bool = False
    replay_history: bool = False
    refresh_screen: bool = False

    def __post_init__(self) -> None:
        if self.replay_history and not self.rebuild_agent:
            raise ValueError("history replay requires an Agent rebuild")


_NO_REBUILD = ReplCommandResult()
_REBUILD_AGENT = ReplCommandResult(rebuild_agent=True)
_REBUILD_AGENT_AND_REPLAY = ReplCommandResult(
    rebuild_agent=True,
    replay_history=True,
)
_REFRESH_SCREEN = ReplCommandResult(refresh_screen=True)


def resolve_workspace_root(root: str | Path | None = None) -> Path:
    """Return the absolute workspace root used by CLI-created tools."""
    return Path(root or Path.cwd()).resolve()


def resolve_workspace_display_name(
    root: str | Path | None = None,
    workspace_name: str | None = None,
) -> str:
    """Return the human-readable workspace name used in the CLI prompt."""
    raw_name = workspace_name if workspace_name is not None else resolve_workspace_root(root).name
    display_name = " ".join(raw_name.strip().split())
    return display_name or "workspace"


def handle_repl_command(
    command_line: str,
    *,
    state: ReplState,
    current_agent: Agent,
    memory_manager: MemoryManager | None,
    compression_reporter: Callable[[ContextCompressionNotice], None],
    output: TextIO,
) -> ReplCommandResult:
    parts = _split_repl_command(command_line)
    command = parts[0].lower()
    args = parts[1:]
    if command in {"/help", "/?"}:
        write_command_heading(output, "Commands")
        entry_width = max(len(name) for name, _ in _HELP_ENTRIES) + 3
        for name, description in _HELP_ENTRIES:
            write_command_entry(output, name, description, width=entry_width)
        output.flush()
        return _NO_REBUILD
    if command == "/refresh":
        if args:
            write_command_warning(output, "Usage: /refresh")
            return _NO_REBUILD
        return _REFRESH_SCREEN
    if command == "/workspace":
        return _handle_workspace_command(
            args,
            state=state,
            output=output,
        )
    if command == "/session":
        return _handle_session_command(
            args,
            state=state,
            current_agent=current_agent,
            output=output,
        )
    if command == "/model":
        return _handle_model_command(
            args,
            state=state,
            output=output,
        )
    if command == "/tokens":
        _write_token_usage(current_agent, state=state, output=output)
        return _NO_REBUILD
    if command == "/status":
        _write_status(current_agent, state=state, output=output)
        return _NO_REBUILD
    if command == "/history":
        _handle_history_command(args, current_agent=current_agent, output=output)
        return _NO_REBUILD
    if command == "/memory":
        if memory_manager is None:
            raise ValueError(
                "memory commands require a configured data directory"
            )
        _handle_memory_command(
            args,
            memory_manager=memory_manager,
            output=output,
        )
        return _NO_REBUILD
    if command == "/skills":
        _handle_skills_command(
            args,
            current_agent=current_agent,
            output=output,
        )
        return _NO_REBUILD
    if command == "/compact":
        _compact_current_session(
            current_agent,
            state=state,
            compression_reporter=compression_reporter,
            output=output,
        )
        return _NO_REBUILD
    write_command_error(output, f"Unknown command: {command}. Type '/help' for commands.")
    return _NO_REBUILD


def _split_repl_command(command_line: str) -> list[str]:
    return [_strip_wrapping_quotes(part) for part in shlex.split(command_line, posix=False)]


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _handle_workspace_command(
    args: list[str],
    *,
    state: ReplState,
    output: TextIO,
) -> ReplCommandResult:
    if state.workspace_root is None:
        write_command_warning(output, "Workspace switching requires a configured workspace root.")
        return _NO_REBUILD
    if not args:
        write_command_heading(output, "Workspace")
        write_command_field(output, "Root", state.workspace_root, subdued_value=True)
        if state.workspace_name is not None:
            write_command_field(output, "Name", state.workspace_name)
        if state.workspace_data_dir is not None:
            write_command_field(output, "Data", state.workspace_data_dir, subdued_value=True)
        output.flush()
        return _NO_REBUILD
    if args == ["list"]:
        _write_workspace_list(state, output)
        return _NO_REBUILD
    target = args[0]
    workspace_root, workspace_name = _resolve_workspace_switch_target(
        target,
        explicit_workspace_name=args[1] if len(args) > 1 else None,
        state=state,
    )
    state.workspace_root = workspace_root
    state.workspace_name = workspace_name
    # Workspace switching also moves the data slot and current session file.
    # The session name is preserved so `/workspace foo` keeps the user's current
    # conversation lane when that lane exists in the target workspace.
    state.workspace_data_dir = resolve_workspace_data_dir(
        workspace_root,
        workspace_name=workspace_name,
        data_dir=state.data_dir,
    )
    write_workspace_metadata(
        workspace_root,
        workspace_name=workspace_name,
        data_dir=state.data_dir,
    )
    state.session_file = _resolve_repl_session_file(state)
    write_command_success(
        output,
        f"Switched workspace to {workspace_root} "
        f"(session: {_session_display(state)}).",
    )
    return _REBUILD_AGENT_AND_REPLAY


def _handle_session_command(
    args: list[str],
    *,
    state: ReplState,
    current_agent: Agent,
    output: TextIO,
) -> ReplCommandResult:
    if state.workspace_root is None:
        write_command_warning(output, "Session switching requires a configured workspace root.")
        return _NO_REBUILD
    if not args:
        write_command_heading(output, "Session")
        write_command_field(output, "Name", _session_display(state))
        if state.session_file is not None:
            write_command_field(output, "File", state.session_file, subdued_value=True)
        output.flush()
        return _NO_REBUILD
    if args == ["list"]:
        _write_session_list(state, output)
        return _NO_REBUILD
    if args == ["clear"]:
        _clear_current_session(current_agent, state=state, output=output)
        return _NO_REBUILD
    if args and args[0].lower() == "rename":
        return _rename_current_session(
            args[1:],
            state=state,
            current_agent=current_agent,
            output=output,
        )
    state.session_name = args[0]
    state.session_file = _resolve_repl_session_file(state)
    write_command_success(output, f"Switched session to {_session_display(state)}.")
    return _REBUILD_AGENT_AND_REPLAY


def _handle_skills_command(
    args: list[str],
    *,
    current_agent: Agent,
    output: TextIO,
) -> None:
    if args not in ([], ["reload"]):
        write_command_warning(output, "Usage: /skills [reload]")
        return
    try:
        tool = current_agent.tools.get("skill")
    except (AttributeError, KeyError):
        write_command_warning(output, "Skills are not configured for this Agent.")
        return
    if not isinstance(tool, SkillTool):
        write_command_error(output, "The registered skill tool is invalid.")
        return

    reloaded = args == ["reload"]
    if reloaded:
        try:
            tool.catalog.reload()
        except (OSError, ValueError) as exc:
            logger.exception("Skill reload failed")
            write_command_error(output, f"Skill reload failed: {exc}")
            return

    skills = tool.catalog.list()
    diagnostics = tool.catalog.diagnostics()
    if reloaded:
        write_command_success(output, f"Reloaded {len(skills)} Skills.")
    if skills:
        write_command_heading(output, f"Skills · {len(skills)}")
        for skill in skills:
            write_command_item(output, f"{skill.reference} · {skill.description}")
    else:
        write_command_warning(output, "No Skills found.")
    if diagnostics:
        write_command_heading(output, f"Discovery diagnostics · {len(diagnostics)}")
        for diagnostic in diagnostics:
            write_command_item(
                output,
                f"{diagnostic.path}: {diagnostic.message}",
                subdued=True,
            )
    output.flush()


def _handle_model_command(
    args: list[str],
    *,
    state: ReplState,
    output: TextIO,
) -> ReplCommandResult:
    if not args:
        write_command_heading(output, "Model")
        write_command_field(output, "Name", state.model)
        if state.data_dir is not None:
            write_command_field(
                output,
                "Config",
                resolve_user_config_path(state.data_dir),
                subdued_value=True,
            )
        else:
            write_command_field(output, "Config", "unavailable in this REPL")
        output.flush()
        return _NO_REBUILD
    if len(args) != 1:
        write_command_warning(output, "Usage: /model [name]")
        return _NO_REBUILD

    model = args[0].strip()
    if not model:
        write_command_warning(output, "Usage: /model [name]")
        return _NO_REBUILD

    state.model = model
    if state.context_compressor is not None:
        state.context_compressor.set_summary_model(model)
    config_path = None
    if state.data_dir is not None:
        config_path = set_configured_model(model, data_dir=state.data_dir)

    write_command_success(output, f"Switched model to {model}.")
    if config_path is not None:
        write_command_field(output, "Config", config_path, subdued_value=True)
    else:
        write_command_note(output, "Model config was not saved because no data directory is configured.")
    output.flush()
    return _REBUILD_AGENT


def _clear_current_session(
    agent: Agent,
    *,
    state: ReplState,
    output: TextIO,
) -> None:
    if state.session_file is None:
        raise ValueError("no session file is configured")
    _clear_agent_messages(agent)
    write_command_success(output, f"Cleared session: {_session_display(state)}")


def _clear_agent_messages(agent: Agent) -> None:
    clear_session = getattr(agent, "clear_session", None)
    if callable(clear_session):
        clear_session()
        return
    agent.session.clear()


def _rename_current_session(
    args: list[str],
    *,
    state: ReplState,
    current_agent: Agent,
    output: TextIO,
) -> ReplCommandResult:
    if len(args) != 1:
        write_command_warning(output, "Usage: /session rename <name>")
        return _NO_REBUILD
    if state.workspace_root is None or state.session_file is None:
        raise ValueError("no session file is configured")
    requested_name = args[0].strip()
    target_file = resolve_session_path(
        state.workspace_root,
        workspace_name=state.workspace_name,
        session_name=requested_name,
        data_dir=state.data_dir,
    )
    target_name = target_file.stem
    if target_file == state.session_file:
        write_command_warning(output, f"Session is already named {target_name}.")
        return _NO_REBUILD
    if target_file.exists():
        raise ValueError(f"session already exists: {target_name}")

    target_trace = resolve_session_trace_path(
        state.workspace_root,
        workspace_name=state.workspace_name,
        session_name=target_name,
        data_dir=state.data_dir,
    )
    if target_trace.exists():
        raise ValueError(f"session trace already exists: {target_name}")

    current_agent.session.move_to(target_file)

    old_name = state.session_name
    move_trace = getattr(current_agent, "move_trace_to", None)
    if callable(move_trace):
        move_trace(
            str(target_trace),
            previous=old_name,
            current=target_name,
        )
    state.session_name = target_name
    state.session_file = target_file
    write_command_success(output, f"Renamed session {old_name} → {target_name}.")
    write_command_field(output, "File", target_file, subdued_value=True)
    output.flush()
    return _REBUILD_AGENT


def _compact_current_session(
    agent: Agent,
    *,
    state: ReplState,
    compression_reporter: Callable[[ContextCompressionNotice], None],
    output: TextIO,
) -> None:
    context_compressor = state.context_compressor
    if context_compressor is None:
        write_command_warning(output, "Context compression is disabled.")
        return

    compression_start: ContextCompressionAnalysis | None = None

    def report_started(analysis: ContextCompressionAnalysis) -> None:
        nonlocal compression_start
        compression_start = analysis
        compression_reporter(
            ContextCompressionNotice(
                stage="started",
                trigger="manual",
                before_count=analysis.request_count,
                before_tokens=analysis.request_tokens,
            ),
        )

    try:
        prepared = agent.compact_context(on_compression_started=report_started)
    except Exception as exc:
        if compression_start is None:
            raise
        diagnostic = getattr(exc, "diagnostic", None)
        compression_reporter(
            ContextCompressionNotice(
                stage="failed",
                trigger="manual",
                before_count=compression_start.request_count,
                before_tokens=compression_start.request_tokens,
                generation_requests=getattr(diagnostic, "generation_requests", None),
                review_performed=getattr(diagnostic, "review_performed", None),
                error=str(exc),
            ),
        )
        raise
    if not prepared.compressed:
        if compression_start is None:
            raise RuntimeError("manual context compression start was not reported")
        compression_reporter(
            ContextCompressionNotice(
                stage="completed",
                trigger="manual",
                before_count=compression_start.request_count,
                before_tokens=compression_start.request_tokens,
                sent_count=prepared.sent_count,
                sent_tokens=prepared.sent_tokens,
                omitted_count=0,
                generation_requests=0,
                review_performed=False,
            )
        )
        return

    if compression_start is None:
        raise RuntimeError("manual context compression start was not reported")
    generation_requests = sum(
        diagnostic.generation_requests or 0
        for diagnostic in prepared.summary_diagnostics
    )
    review_performed = any(
        diagnostic.review_performed is True
        for diagnostic in prepared.summary_diagnostics
    )
    compression_reporter(
        ContextCompressionNotice(
            stage="completed",
            trigger="manual",
            before_count=compression_start.request_count,
            before_tokens=compression_start.request_tokens,
            sent_count=prepared.sent_count,
            sent_tokens=prepared.sent_tokens,
            omitted_count=prepared.omitted_count,
            generation_requests=generation_requests,
            review_performed=review_performed,
            canonical_count=agent.session.message_count(),
            canonical_tokens=prepared.original_tokens,
            breakdown=prepared.compression_breakdown,
        ),
    )


def _write_token_usage(
    agent: Agent,
    *,
    state: ReplState,
    output: TextIO,
) -> None:
    analysis = agent.analyze_context()
    current_tokens = analysis.current_tokens
    write_command_heading(output, "Context usage")
    write_command_field(output, "Messages", analysis.message_count)
    if state.context_compressor is None:
        write_command_field(output, "Context window", "unavailable (compression disabled)")
    else:
        max_tokens = state.context_compressor.max_tokens
        percent = current_tokens.total / max_tokens * 100
        write_command_field(output, "Context window", f"{max_tokens:,} tokens")
        write_command_field(
            output,
            "Current history",
            f"{current_tokens.total:,}/{max_tokens:,} tokens ({percent:.1f}%)",
        )
    write_command_field(output, "Canonical history", f"{analysis.history_tokens:,} tokens")
    write_command_heading(output, "Current history breakdown")
    write_command_field(output, "Summary", f"{current_tokens.summary:,} tokens")
    write_command_field(output, "User input", f"{current_tokens.user_input:,} tokens")
    write_command_field(
        output,
        "Assistant content",
        f"{current_tokens.assistant_content:,} tokens",
    )
    write_command_field(
        output,
        "Tool calls/results/summaries",
        f"{current_tokens.tool_calls:,} tokens",
    )
    write_command_field(output, "Token encoding", "o200k_base")
    write_command_note(
        output,
        "Scope: current provider-facing session history only; excludes dynamic "
        "system prompt and tool schemas. Summary includes its compact wrapper; "
        "bundled transcript is split by message role.",
    )
    rows = agent.session.usage_snapshot()
    totals = {key: sum(row[key] for row in rows) for key in
              (
                  "requests",
                  "prompt_tokens",
                  "cached_prompt_tokens",
                  "completion_tokens",
                  "unreported",
                  "failed",
              )}
    noncached_prompt_tokens = max(
        0, totals["prompt_tokens"] - totals["cached_prompt_tokens"]
    )
    weighted_tokens = (
        noncached_prompt_tokens
        + totals["completion_tokens"]
        + totals["cached_prompt_tokens"] * 0.1
    )
    weighted_prompt_tokens = (
        noncached_prompt_tokens + totals["cached_prompt_tokens"] * 0.1
    )
    cached_share = _percentage(
        totals["cached_prompt_tokens"], totals["prompt_tokens"]
    )
    write_command_heading(output, "API usage · session cumulative")
    write_command_field(
        output,
        "Requests",
        f"{totals['requests']} (failed: {totals['failed']})",
    )
    write_command_field(
        output,
        "Total",
        f"{totals['prompt_tokens'] + totals['completion_tokens']:,} tokens "
        f"({totals['prompt_tokens']:,} prompt + "
        f"{totals['completion_tokens']:,} completion)",
    )
    write_command_field(
        output,
        "Weighted",
        f"{_format_weighted_tokens(weighted_tokens)} tokens "
        f"({noncached_prompt_tokens:,} non-cached prompt + "
        f"{totals['cached_prompt_tokens']:,} cached prompt × 0.1 + "
        f"{totals['completion_tokens']:,} completion)",
    )
    write_command_field(output, "Unreported requests", totals["unreported"])
    write_command_note(
        output,
        "Recorded since usage tracking began; includes subagents and auxiliary calls.",
    )
    for row in rows:
        row_cached_share = _percentage(
            row["cached_prompt_tokens"], row["prompt_tokens"]
        )
        write_command_item(
            output,
            f"{row['purpose']} / {row['model']} · {row['requests']:,} requests · "
            f"{row['prompt_tokens']:,} prompt "
            f"({row['cached_prompt_tokens']:,} cached · {row_cached_share}) + "
            f"{row['completion_tokens']:,} completion tokens",
        )
    write_command_heading(output, "Prompt cache · session cumulative")
    write_command_field(
        output,
        "Cached prompt",
        f"{totals['cached_prompt_tokens']:,} tokens ({cached_share})",
    )
    write_command_field(
        output,
        "Non-cached prompt",
        f"{noncached_prompt_tokens:,} tokens",
    )
    write_command_field(
        output,
        "Weighted prompt",
        f"{_format_weighted_tokens(weighted_prompt_tokens)} tokens",
    )
    write_command_field(
        output,
        "Weighted reduction",
        f"{_format_weighted_tokens(totals['cached_prompt_tokens'] * 0.9)} tokens",
    )
    write_command_note(
        output,
        "Cached share measures reused prompt tokens, not request-level cache hits; "
        "weighted values count cached prompt tokens at × 0.1. Session cumulative "
        "usage does not retain cache-creation token totals.",
    )
    usage = _agent_usage_stats(agent)
    if usage is not None:
        write_command_heading(output, "API usage · current Agent")
        write_command_field(output, "Requests", usage["requests"])
        write_command_field(
            output,
            "Last request",
            f"{usage['last_prompt']:,} prompt + "
            f"{usage['last_completion']:,} completion tokens",
        )
        write_command_field(
            output,
            "Cumulative",
            f"{usage['total_prompt']:,} prompt + "
            f"{usage['total_completion']:,} completion tokens",
        )

    context_compressor = state.context_compressor
    if context_compressor is None:
        write_command_field(output, "Context compression", "disabled")
        output.flush()
        return

    compression = analysis.compression
    if compression is None:
        raise RuntimeError("context compression analysis is unavailable")
    write_command_heading(output, "Context compression")
    write_command_field(output, "Status", "enabled")
    write_command_field(output, "Max tokens", context_compressor.max_tokens)
    write_command_field(output, "Trigger tokens", context_compressor.trigger_tokens)
    write_command_field(output, "Target tokens", context_compressor.target_tokens)
    write_command_field(
        output,
        "Summary initial tokens",
        context_compressor.summary_initial_tokens,
    )
    write_command_field(output, "Summary cap tokens", context_compressor.summary_cap_tokens)
    write_command_field(output, "Recent tokens", context_compressor.recent_tokens)
    write_command_field(
        output,
        "Would compress now",
        "yes" if compression.would_compress else "no",
    )
    if compression.would_compress:
        write_command_field(
            output,
            "Initial messages selected",
            compression.initial_omitted_count,
        )
    output.flush()


def _write_status(
    agent: Agent,
    *,
    state: ReplState,
    output: TextIO,
) -> None:
    analysis = agent.analyze_context()
    message_count = analysis.message_count
    history_tokens = analysis.history_tokens
    write_command_heading(output, "Status")
    write_command_field(
        output,
        "Workspace",
        state.workspace_root if state.workspace_root is not None else "(none)",
        subdued_value=True,
    )
    write_command_field(output, "Session", _session_display(state))
    write_command_field(output, "Model", state.model)
    write_command_field(output, "Messages", message_count)
    context_compressor = state.context_compressor
    if context_compressor is None:
        write_command_field(
            output,
            "Current context",
            f"{analysis.current_tokens.total:,} tokens",
        )
        write_command_field(output, "Canonical history", f"{history_tokens:,} tokens")
        write_command_field(output, "Context compression", "disabled")
    else:
        max_tokens = context_compressor.max_tokens
        current_context_tokens = analysis.current_tokens.total
        percent = (current_context_tokens / max_tokens * 100) if max_tokens else 0
        write_command_field(
            output,
            "Current context",
            f"{current_context_tokens:,}/{max_tokens:,} tokens ({percent:.1f}%)",
        )
        write_command_field(output, "Canonical history", f"{history_tokens:,} tokens")
        write_command_field(
            output,
            "Compression target",
            f"{context_compressor.target_tokens:,} tokens",
        )
    if state.session_file is not None:
        write_command_field(output, "Session file", state.session_file, subdued_value=True)
    output.flush()


def _handle_history_command(
    args: list[str],
    *,
    current_agent: Agent,
    output: TextIO,
) -> None:
    if len(args) < 2 or args[0].lower() != "search":
        write_command_warning(output, "Usage: /history search <query>")
        return
    query = " ".join(args[1:]).strip()
    if not query:
        write_command_warning(output, "Usage: /history search <query>")
        return
    messages = _agent_message_history(current_agent)
    matches: list[tuple[int, str, str]] = []
    query_folded = query.casefold()
    for index, message in enumerate(messages, 1):
        searchable = _message_search_text(message)
        if query_folded not in searchable.casefold():
            continue
        role = str(message.get("role", "unknown"))
        matches.append((index, role, _history_snippet(searchable, query_folded)))
        if len(matches) == 20:
            break

    if not matches:
        write_command_warning(output, f"No messages matched: {query}")
        return
    write_command_heading(output, f"History matches · {query!r} · {len(matches)}")
    for index, role, snippet in matches:
        write_command_item(output, f"#{index} · {role} · {snippet}")
    if len(matches) == 20:
        write_command_note(output, "Results limited to 20 messages.")
    output.flush()


def _handle_memory_command(
    args: list[str],
    *,
    memory_manager: MemoryManager,
    output: TextIO,
) -> None:
    if not args:
        write_command_warning(
            output,
            "Usage: /memory add|list|search|forget "
            "(type /help for command details)",
        )
        return

    action = args[0].lower()
    action_args = args[1:]
    if action == "add":
        _add_memory(
            action_args,
            memory_manager=memory_manager,
            output=output,
        )
    elif action == "list":
        _list_memories(
            action_args,
            memory_manager=memory_manager,
            output=output,
        )
    elif action == "search":
        _search_memories(
            action_args,
            memory_manager=memory_manager,
            output=output,
        )
    elif action == "forget":
        _forget_memory(
            action_args,
            memory_manager=memory_manager,
            output=output,
        )
    else:
        write_command_error(output, f"Unknown memory action: {action}.")


def _add_memory(
    args: list[str],
    *,
    memory_manager: MemoryManager,
    output: TextIO,
) -> None:
    scope, remaining = _memory_scope_from_args(args, required=True)
    content = " ".join(remaining).strip()
    if scope is None or not content:
        write_command_warning(
            output,
            "Usage: /memory add --user|--workspace <text>",
        )
        return
    result = memory_manager.add(scope, content)
    if result.action == "added":
        write_command_success(
            output,
            f"Added {result.record.scope} memory {result.record.id}.",
        )
    elif result.action == "duplicate":
        write_command_warning(
            output,
            f"Not added; duplicates {result.record.scope} memory {result.record.id}.",
        )
    else:
        if result.previous is None:
            raise RuntimeError("replaced memory result has no previous record")
        write_command_success(
            output,
            f"Updated {result.record.scope} memory {result.record.id}.",
        )
        write_command_field(output, "Previous", result.previous.content)
    write_command_field(output, "Content", result.record.content)
    if result.reason:
        write_command_field(output, "Reason", result.reason)
    output.flush()


def _list_memories(
    args: list[str],
    *,
    memory_manager: MemoryManager,
    output: TextIO,
) -> None:
    scope, remaining = _memory_scope_from_args(args, required=False)
    if remaining:
        write_command_warning(
            output,
            "Usage: /memory list [--user|--workspace]",
        )
        return
    records = [
        record
        for selected_scope in _selected_memory_scopes(scope)
        for record in memory_manager.list(selected_scope)
    ]
    if not records:
        write_command_warning(output, "No memories found.")
        return
    write_command_heading(output, f"Memories · {len(records)}")
    for record in records:
        write_command_item(output, f"{record.id} · {record.scope} · {record.content}")
    output.flush()


def _search_memories(
    args: list[str],
    *,
    memory_manager: MemoryManager,
    output: TextIO,
) -> None:
    scope, remaining = _memory_scope_from_args(args, required=False)
    query = " ".join(remaining).strip()
    if not query:
        write_command_warning(
            output,
            "Usage: /memory search [--user|--workspace] <query>",
        )
        return
    records = [
        record
        for selected_scope in _selected_memory_scopes(scope)
        for record in memory_manager.search(selected_scope, query)
    ]
    if not records:
        write_command_warning(output, f"No memories matched: {query}")
        return
    write_command_heading(output, f"Memory matches · {query!r} · {len(records)}")
    for record in records:
        write_command_item(output, f"{record.id} · {record.scope} · {record.content}")
    output.flush()


def _forget_memory(
    args: list[str],
    *,
    memory_manager: MemoryManager,
    output: TextIO,
) -> None:
    if len(args) != 1:
        write_command_warning(output, "Usage: /memory forget <id>")
        return
    memory_id = args[0].strip()
    for scope in _selected_memory_scopes(None):
        deleted = memory_manager.delete(scope, memory_id)
        if deleted is not None:
            write_command_success(
                output,
                f"Forgot {deleted.scope} memory {deleted.id}: {deleted.content}",
            )
            return
    write_command_warning(output, f"Memory not found: {memory_id}")


def _memory_scope_from_args(
    args: list[str],
    *,
    required: bool,
) -> tuple[MemoryCommandScope | None, list[str]]:
    if args and args[0].lower() in {"--user", "--workspace"}:
        scope: MemoryCommandScope = (
            "user" if args[0].lower() == "--user" else "workspace"
        )
        return scope, args[1:]
    if required:
        return None, args
    return None, args


def _selected_memory_scopes(
    scope: MemoryCommandScope | None,
) -> tuple[MemoryCommandScope, ...]:
    return (scope,) if scope is not None else ("user", "workspace")


def _agent_usage_stats(agent: Agent) -> dict[str, int]:
    return {
        "requests": agent.llm_request_count,
        "last_prompt": agent.last_prompt_tokens,
        "last_completion": agent.last_completion_tokens,
        "total_prompt": agent.total_prompt_tokens,
        "total_completion": agent.total_completion_tokens,
    }


def _format_weighted_tokens(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _percentage(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "0.0%"


def _message_search_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        tool_calls = message.get("tool_calls")
        return json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else ""
    return json.dumps(content, ensure_ascii=False)


def _history_snippet(text: str, query_folded: str, *, limit: int = 140) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    center = one_line.casefold().find(query_folded)
    start = max(0, center - limit // 3)
    end = min(len(one_line), start + limit)
    prefix = "..." if start else ""
    suffix = "..." if end < len(one_line) else ""
    return f"{prefix}{one_line[start:end]}{suffix}"


def _agent_message_history(agent: Agent) -> list[dict[str, Any]]:
    return agent.session.snapshot()


def _resolve_repl_session_file(state: ReplState) -> Path | None:
    if state.workspace_root is None:
        return None
    return resolve_session_path(
        state.workspace_root,
        workspace_name=state.workspace_name,
        session_name=state.session_name,
        data_dir=state.data_dir,
    )


def _resolve_workspace_switch_target(
    target: str,
    *,
    explicit_workspace_name: str | None,
    state: ReplState,
) -> tuple[Path, str | None]:
    workspace_dir = _workspace_slot_dir(state, target)
    if explicit_workspace_name is None and workspace_dir is not None:
        metadata = _read_workspace_metadata(workspace_dir)
        root = metadata.get("workspace_root")
        if not isinstance(root, str) or not root:
            raise ValueError(f"workspace slot has no workspace_root: {target}")
        name = metadata.get("workspace_name")
        return resolve_workspace_root(root), name if isinstance(name, str) else None

    root = resolve_workspace_root(target)
    if not root.is_dir():
        raise ValueError(f"workspace root is not a directory: {root}")
    return root, explicit_workspace_name


def _workspace_slot_dir(state: ReplState, workspace_id: str) -> Path | None:
    if state.data_dir is None:
        return None
    candidate = (state.data_dir / "workspaces" / workspace_id).resolve()
    workspaces_dir = (state.data_dir / "workspaces").resolve()
    if not candidate.is_relative_to(workspaces_dir):
        return None
    if not candidate.is_dir():
        return None
    return candidate


def _write_workspace_list(state: ReplState, output: TextIO) -> None:
    workspaces_dir = (
        state.data_dir / "workspaces"
        if state.data_dir is not None
        else None
    )
    if workspaces_dir is None or not workspaces_dir.is_dir():
        write_command_warning(output, "No saved workspaces.")
        return
    workspace_dirs = sorted(path for path in workspaces_dir.iterdir() if path.is_dir())
    if not workspace_dirs:
        write_command_warning(output, "No saved workspaces.")
        return
    write_command_heading(output, f"Workspaces · {len(workspace_dirs)}")
    for workspace_dir in workspace_dirs:
        metadata = _read_workspace_metadata(workspace_dir)
        root = metadata.get("workspace_root") or "(unknown root)"
        name = metadata.get("workspace_name")
        name_part = f" · {name}" if isinstance(name, str) and name else ""
        write_command_item(output, f"{workspace_dir.name}{name_part} · {root}")
    output.flush()


def _write_session_list(state: ReplState, output: TextIO) -> None:
    if state.workspace_data_dir is None:
        write_command_warning(output, "No workspace data directory is configured.")
        return
    sessions_dir = state.workspace_data_dir / "sessions"
    if not sessions_dir.is_dir():
        write_command_warning(output, "No saved sessions in this workspace.")
        return
    session_files = sorted(path for path in sessions_dir.glob("*.json") if path.is_file())
    if not session_files:
        write_command_warning(output, "No saved sessions in this workspace.")
        return
    write_command_heading(output, f"Sessions · {len(session_files)}")
    for session_file in session_files:
        write_command_item(output, session_file.stem)
    output.flush()


def _read_workspace_metadata(workspace_dir: Path) -> dict[str, object]:
    metadata_path = workspace_dir / WORKSPACE_METADATA_FILE_NAME
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)
    if not isinstance(data, dict):
        raise ValueError(f"workspace metadata must be a JSON object: {metadata_path}")
    return data


def _session_display(state: ReplState) -> str:
    return state.session_name
