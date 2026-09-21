"""Domain types shared by MCP configuration, runtime, and tool adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


McpTransport = Literal["stdio", "streamable_http"]


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """Validated connection and local capability-filter configuration."""

    name: str
    transport: McpTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    include: frozenset[str] | None = None
    exclude: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class McpToolSpec:
    """One filtered remote MCP tool exposed through a stable local name."""

    server_name: str
    remote_name: str
    local_name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class McpDiagnostic:
    """Non-fatal startup or discovery problem for one MCP server/tool."""

    server_name: str
    message: str


@dataclass(frozen=True, slots=True)
class McpConnectionSnapshot:
    """Immutable result published after MCP connection startup."""

    tools: tuple[McpToolSpec, ...] = ()
    diagnostics: tuple[McpDiagnostic, ...] = ()
    configured_servers: int = 0
    connected_servers: int = 0
