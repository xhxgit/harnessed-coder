"""Command-line interface for harnessed_coder."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from .agent import Agent
from .agent.system_prompt import SystemPromptContext, build_system_prompt
from .constants import DEFAULT_SESSION_NAME, LOG_DIR_NAME
from .context import (
    ContextCompressor,
    LLMContextSummarizer,
    LLMToolBatchSummarizer,
    ToolBatchSummaryFunction,
)
from .diagnostics import (
    DEFAULT_LOG_LEVEL,
    configure_api_exchange_logging,
    configure_logging,
    configure_jsonl_trace,
    disable_api_exchange_logging,
    make_run_id,
)
from .llm import LLMResponse
from .memory import MemoryManager
from .mcp_client import (
    McpArtifactStore,
    McpConnections,
    load_mcp_server_configs,
)
from .permissions import LLMPermissionReviewer
from .repl.commands import resolve_workspace_display_name, resolve_workspace_root
from .repl.runtime import run_repl as run_repl_runtime
from .session import (
    ConversationSession,
    resolve_data_dir,
    resolve_session_path,
    resolve_session_trace_path,
    resolve_workspace_data_dir,
    write_workspace_metadata,
)
from .shell import select_shell
from .skills import SkillCatalog
from .tools import create_default_registry
from .user_config import (
    create_initial_user_config,
    get_configured_model,
    get_context_max_tokens,
    set_user_config_data_dir,
)
from .workspace_instructions import load_agents_instructions

logger = logging.getLogger(__name__)


def create_agent(
    root: str | Path | None = None,
    *,
    session: ConversationSession | None = None,
    context_compressor: ContextCompressor | None = None,
    memory_manager: MemoryManager | None = None,
    data_dir: str | Path | None = None,
    workspace_name: str | None = None,
    session_name: str = DEFAULT_SESSION_NAME,
    run_id: str | None = None,
    trace_path: str | Path | None = None,
    mcp_connections: McpConnections | None = None,
    mcp_artifact_store: McpArtifactStore | None = None,
    tool_batch_summary_function: ToolBatchSummaryFunction | None = None,
    model: str,
) -> Agent:
    """Create an agent with the default workspace tools."""
    workspace_root = resolve_workspace_root(root)
    resolved_session = session or ConversationSession()
    skill_catalog = SkillCatalog.discover(
        workspace_root,
        data_dir=data_dir,
    )
    agents_instructions, agents_diagnostic = load_agents_instructions(workspace_root)
    if agents_diagnostic is not None:
        logger.warning(
            "AGENTS.md discovery skipped %s: %s",
            agents_diagnostic.path,
            agents_diagnostic.message,
        )

    def provide_system_prompt() -> str:
        return build_system_prompt(
            SystemPromptContext(
                workspace_root=workspace_root,
                workspace_name=resolve_workspace_display_name(
                    workspace_root,
                    workspace_name,
                ),
                session_name=session_name,
            ),
            memories=(
                memory_manager.list_all()
                if memory_manager is not None
                else []
            ),
            skills=skill_catalog.list(),
            agents_instructions=agents_instructions,
        )

    def record_tool_catalog_call(response: LLMResponse | None) -> None:
        resolved_session.record_usage(
            model=model,
            purpose="tool_catalog",
            prompt_tokens=response.prompt_tokens if response is not None else 0,
            cached_prompt_tokens=(
                response.cached_prompt_tokens or 0 if response is not None else 0
            ),
            completion_tokens=(
                response.completion_tokens if response is not None else 0
            ),
            unreported=(
                not response.usage_available if response is not None else True
            ),
            failed=response is None,
        )

    return Agent(
        create_default_registry(
            workspace_root,
            data_dir=data_dir,
            skill_catalog=skill_catalog,
            mcp_connections=mcp_connections,
            mcp_artifact_store=mcp_artifact_store,
            model=model,
            run_id=run_id,
            trace_path=trace_path,
            session_used_tool_names=resolved_session.used_tool_names(),
            on_session_tool_used=resolved_session.record_used_tool,
            on_tool_catalog_call=record_tool_catalog_call,
        ),
        model=model,
        run_id=run_id,
        trace_path=None if trace_path is None else str(Path(trace_path).resolve()),
        session=resolved_session,
        context_compressor=context_compressor,
        permission_reviewer=LLMPermissionReviewer(model=model),
        system_prompt_provider=provide_system_prompt,
        tool_batch_summary_function=tool_batch_summary_function,
    )


def run_repl(
    *,
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
    """Run the REPL using the CLI's default Agent composition."""
    run_repl_runtime(
        agent_factory=create_agent,
        workspace_root=workspace_root,
        workspace_name=workspace_name,
        data_dir=data_dir,
        workspace_data_dir=workspace_data_dir,
        log_file=log_file,
        trace_file=trace_file,
        session_name=session_name,
        run_id=run_id,
        session_file=session_file,
        context_compressor=context_compressor,
        mcp_connections=mcp_connections,
        mcp_artifact_store=mcp_artifact_store,
        automatic_memory_extraction=automatic_memory_extraction,
        tool_batch_summary_factory=tool_batch_summary_factory,
        model=model,
        input_func=input_func,
        output=output,
        error_output=error_output,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="harnessed-coder",
        description="Run an interactive coding agent in a workspace.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root for file tools. Defaults to the current directory.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"File log level. Defaults to {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="User data directory. Defaults to ~/.harnessed-coder.",
    )
    parser.add_argument(
        "--api-log-dir",
        type=Path,
        default=None,
        help=(
            "Write sensitive full model API request/response JSONL logs to this "
            "directory. Disabled by default."
        ),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "Workspace data slot name. Defaults to a stable id derived from --root. "
            "Use this to give the same root a friendly workspace name."
        ),
    )
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION_NAME,
        help="Session name within the selected workspace. Defaults to default.",
    )
    parser.add_argument(
        "--auto-memory-extraction",
        action="store_true",
        help="Enable automatic long-term memory extraction after successful turns.",
    )
    parser.add_argument(
        "--no-tool-batch-summary",
        action="store_true",
        help="Disable provider-facing summaries of completed tool batches.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Application entry point."""
    args = build_arg_parser().parse_args(argv)
    workspace_root = resolve_workspace_root(args.root)
    data_dir = resolve_data_dir(args.data_dir)
    log_dir = data_dir / LOG_DIR_NAME
    set_user_config_data_dir(data_dir)
    created_config_path = create_initial_user_config(data_dir)
    if created_config_path is not None:
        print(f"Created initial LLM config: {created_config_path}")
        print(
            "Replace the openai_api_key placeholder in that file, review the "
            "LLM settings, then run Harnessed Coder again."
        )
        return
    model = get_configured_model(data_dir)
    context_max_tokens = get_context_max_tokens(data_dir)
    workspace_data_dir = resolve_workspace_data_dir(
        workspace_root,
        workspace_name=args.workspace,
        data_dir=data_dir,
    )
    write_workspace_metadata(
        workspace_root,
        workspace_name=args.workspace,
        data_dir=data_dir,
    )
    session_path = resolve_session_path(
        workspace_root,
        workspace_name=args.workspace,
        session_name=args.session,
        data_dir=data_dir,
    )
    run_id = make_run_id()
    log_file = configure_logging(log_dir=log_dir, level=args.log_level, run_id=run_id)
    api_log_file = (
        configure_api_exchange_logging(log_dir=args.api_log_dir, run_id=run_id)
        if args.api_log_dir is not None
        else None
    )
    if api_log_file is not None:
        logger.warning(
            "Sensitive API exchange logging enabled: file=%s",
            api_log_file,
        )
    configure_jsonl_trace()
    trace_file = resolve_session_trace_path(
        workspace_root,
        workspace_name=args.workspace,
        session_name=args.session,
        data_dir=data_dir,
    )
    context_compressor = ContextCompressor(
        max_tokens=context_max_tokens,
        summary_function=LLMContextSummarizer(
            model=model,
            context_max_tokens=context_max_tokens,
        ),
    )
    logger.info(
        "CLI started with workspace=%s bash=enabled shell=%s session=%s "
        "model=%s context_compression=enabled tool_batch_summary=%s "
        "memory_extraction=%s",
        workspace_root,
        select_shell(),
        session_path,
        model,
        "disabled" if args.no_tool_batch_summary else "enabled",
        "enabled" if args.auto_memory_extraction else "disabled",
    )
    mcp_connections: McpConnections | None = None
    status = "failed"
    try:
        mcp_configs = load_mcp_server_configs(data_dir)
        mcp_connections = McpConnections(mcp_configs) if mcp_configs else None
        mcp_artifact_store = (
            McpArtifactStore(data_dir) if mcp_connections else None
        )
        if mcp_connections is not None:
            snapshot = mcp_connections.start()
            logger.info(
                "MCP connected=%s/%s tools=%s",
                snapshot.connected_servers,
                snapshot.configured_servers,
                len(snapshot.tools),
            )
            for diagnostic in snapshot.diagnostics:
                logger.warning(
                    "MCP server %s: %s",
                    diagnostic.server_name,
                    diagnostic.message,
                )
        run_repl(
            workspace_root=workspace_root,
            workspace_name=args.workspace,
            data_dir=data_dir,
            workspace_data_dir=workspace_data_dir,
            log_file=log_file,
            trace_file=trace_file,
            session_file=session_path,
            session_name=args.session,
            run_id=run_id,
            context_compressor=context_compressor,
            mcp_connections=mcp_connections,
            mcp_artifact_store=mcp_artifact_store,
            automatic_memory_extraction=args.auto_memory_extraction,
            tool_batch_summary_factory=(
                None
                if args.no_tool_batch_summary
                else lambda target_model: LLMToolBatchSummarizer(model=target_model)
            ),
            model=model,
        )
        status = "completed"
    except Exception:
        logger.exception(
            "CLI terminated unexpectedly: workspace=%s session=%s",
            workspace_root,
            session_path,
        )
        raise
    finally:
        if mcp_connections is not None:
            mcp_connections.close()
        if api_log_file is not None:
            disable_api_exchange_logging()
        logger.info("CLI stopped: status=%s", status)
