"""Public tool interfaces."""

from collections.abc import Callable
import logging
from pathlib import Path

from harnessed_coder.session.usage import tracked_call
from ..llm import LLMResponse, chat as llm_chat
from ..mcp_client import McpArtifactStore, McpConnections
from ..skills.catalog import SkillCatalog
from .base import Tool, ToolExecutionResult, ToolMetadata
from .bash import BashTool
from .edit import EditFileTool
from .glob import GlobTool
from .grep import GrepTool
from .import_mcp_artifact import ImportMcpArtifactTool
from .mcp_tool import McpTool, create_mcp_tool
from .read import ReadFileTool
from .tool_registry import ToolRegistry
from .session_history import (
    SessionListTool,
    SessionListWorkspacesTool,
    SessionReadTool,
    SessionReadToolResultTool,
    SessionSearchTool,
)
from .skill import SkillTool
from .subagent import SubAgentRunner, SubAgentTool
from .subagent_runner import ConfiguredSubAgentRunner
from .tool_search import LLMToolMatcher, ToolCatalog, ToolMatcher, ToolSearchTool
from .write import WriteFileTool


logger = logging.getLogger(__name__)


def create_default_registry(
    root: str | Path | None = None,
    *,
    include_subagent: bool = True,
    data_dir: str | Path | None = None,
    skill_catalog: SkillCatalog | None = None,
    mcp_connections: McpConnections | None = None,
    mcp_artifact_store: McpArtifactStore | None = None,
    base_url: str | None = None,
    chat_function: Callable[..., LLMResponse] | None = None,
    run_id: str | None = None,
    trace_path: str | Path | None = None,
    session_used_tool_names: tuple[str, ...] = (),
    on_session_tool_used: Callable[[str], None] | None = None,
    on_tool_catalog_call: Callable[[LLMResponse | None], None] | None = None,
    model: str,
) -> ToolRegistry:
    """Return the default workspace tool registry for an agent."""
    registry = ToolRegistry(
        [
            ReadFileTool(root),
            EditFileTool(root),
            WriteFileTool(root),
            GlobTool(root),
            GrepTool(root),
            BashTool(root),
        ],
        session_used_tool_names=session_used_tool_names,
        on_session_tool_used=on_session_tool_used,
    )
    if skill_catalog is not None:
        registry.register(SkillTool(skill_catalog))

    session_history_tools: list[Tool] = [
        SessionListWorkspacesTool(data_dir),
        SessionListTool(data_dir),
        SessionReadTool(data_dir),
        SessionReadToolResultTool(data_dir),
        SessionSearchTool(data_dir),
    ]
    for tool in session_history_tools:
        registry.register(tool)

    deferred_tools: list[Tool] = []
    mcp_tools: list[Tool] = []
    if mcp_connections is not None:
        artifact_store = mcp_artifact_store or McpArtifactStore(data_dir)
        deferred_tools.append(ImportMcpArtifactTool(root, artifact_store))
        mcp_tools = [
            create_mcp_tool(spec, mcp_connections, artifact_store)
            for spec in mcp_connections.tool_specs()
        ]
        deferred_tools.extend(mcp_tools)
    for tool in deferred_tools:
        registry.register(tool, visible=False)

    configured_chat = chat_function or llm_chat
    tool_catalog = ToolCatalog(
        data_dir,
        lambda messages: configured_chat(
            messages,
            model=model,
            base_url=base_url,
            tools=[],
            on_text_delta=None,
            on_activity_delta=None,
            reasoning_effort="none",
        ),
        on_generation_call=on_tool_catalog_call,
    )
    catalog_refresh = tool_catalog.refresh(mcp_tools)
    if mcp_tools:
        logger.info(
            "MCP tool catalog refreshed: cached=%s generated=%s failed=%s requests=%s path=%s",
            catalog_refresh.cached_tools,
            catalog_refresh.generated_tools,
            catalog_refresh.failed_tools,
            catalog_refresh.generation_requests,
            tool_catalog.path,
        )
    registry.register(
        ToolSearchTool(
            registry,
            LLMToolMatcher(
                lambda messages: tracked_call(
                    configured_chat,
                    "tool_search",
                    messages,
                    model=model,
                    base_url=base_url,
                    tools=[],
                    on_text_delta=None,
                    on_activity_delta=None,
                    reasoning_effort="none",
                ),
                tool_catalog,
            ),
        )
    )

    if include_subagent:
        registry.register(
            SubAgentTool(
                ConfiguredSubAgentRunner(
                    root,
                    model=model,
                    base_url=base_url,
                    chat_function=configured_chat,
                    data_dir=data_dir,
                    mcp_connections=mcp_connections,
                    mcp_artifact_store=mcp_artifact_store,
                    run_id=run_id,
                    trace_path=trace_path,
                )
            )
        )
    return registry


__all__ = [
    "BashTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ImportMcpArtifactTool",
    "McpTool",
    "SessionListTool",
    "SessionListWorkspacesTool",
    "SessionReadTool",
    "SessionReadToolResultTool",
    "SessionSearchTool",
    "SkillTool",
    "SubAgentRunner",
    "ReadFileTool",
    "SubAgentTool",
    "Tool",
    "ToolExecutionResult",
    "ToolMetadata",
    "ToolRegistry",
    "ToolMatcher",
    "LLMToolMatcher",
    "ToolCatalog",
    "ToolSearchTool",
    "WriteFileTool",
    "create_default_registry",
]
