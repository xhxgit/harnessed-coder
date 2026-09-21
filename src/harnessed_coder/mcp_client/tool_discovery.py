"""Discover, filter, and adapt tools exposed by one MCP connection."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Protocol, cast

from mcp import types as mcp_types

from .types import McpDiagnostic, McpServerConfig, McpToolSpec


_MAX_LOCAL_TOOL_NAME_CHARS = 64
_UNSAFE_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


class McpToolListingSession(Protocol):
    async def list_tools(
        self,
        *,
        params: mcp_types.PaginatedRequestParams | None = None,
    ) -> mcp_types.ListToolsResult: ...


async def discover_tools(
    session: McpToolListingSession,
    config: McpServerConfig,
) -> tuple[list[McpToolSpec], list[McpDiagnostic]]:
    """Fetch every tools/list page and apply the configured capability filter."""
    remote_tools: list[mcp_types.Tool] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        params = (
            mcp_types.PaginatedRequestParams(cursor=cursor)
            if cursor is not None
            else None
        )
        result = await session.list_tools(params=params)
        remote_tools.extend(result.tools)
        if result.next_cursor is None:
            break
        if result.next_cursor in seen_cursors:
            raise RuntimeError(
                f"MCP tools/list repeated cursor: {result.next_cursor!r}"
            )
        seen_cursors.add(result.next_cursor)
        cursor = result.next_cursor

    remote_names = {tool.name for tool in remote_tools}
    configured_filter = config.include or config.exclude or frozenset()
    selected = (
        config.include
        if config.include is not None
        else remote_names - (config.exclude or frozenset())
    )
    diagnostics = [
        McpDiagnostic(
            config.name,
            f"configured tool name was not returned by server: {name}",
        )
        for name in sorted(configured_filter - remote_names)
    ]

    specs: list[McpToolSpec] = []
    seen_remote_names: set[str] = set()
    seen_local_names: set[str] = set()
    for tool in remote_tools:
        if tool.name not in selected:
            continue
        if tool.name in seen_remote_names:
            diagnostics.append(
                McpDiagnostic(
                    config.name,
                    f"duplicate remote tool name skipped: {tool.name}",
                )
            )
            continue
        seen_remote_names.add(tool.name)
        local_name = _local_tool_name(config.name, tool.name)
        if local_name in seen_local_names:
            diagnostics.append(
                McpDiagnostic(
                    config.name,
                    f"local tool name collision skipped: {tool.name} -> "
                    f"{local_name}",
                )
            )
            continue
        seen_local_names.add(local_name)
        description = tool.description or f"MCP tool {tool.name}"
        specs.append(
            McpToolSpec(
                server_name=config.name,
                remote_name=tool.name,
                local_name=local_name,
                description=(
                    f"MCP server '{config.name}', remote tool '{tool.name}'. "
                    f"{description}"
                ),
                input_schema=cast(
                    dict[str, object],
                    deepcopy(tool.input_schema),
                ),
            )
        )
    return specs, diagnostics


def _local_tool_name(server_name: str, remote_name: str) -> str:
    candidate = (
        f"mcp__{_safe_name_component(server_name)}"
        f"__{_safe_name_component(remote_name)}"
    )
    if len(candidate) <= _MAX_LOCAL_TOOL_NAME_CHARS:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:10]
    prefix_length = _MAX_LOCAL_TOOL_NAME_CHARS - len(digest) - 2
    return f"{candidate[:prefix_length]}__{digest}"


def _safe_name_component(value: str) -> str:
    normalized = _UNSAFE_TOOL_NAME_CHARS.sub("_", value).strip("_")
    return normalized or "unnamed"
