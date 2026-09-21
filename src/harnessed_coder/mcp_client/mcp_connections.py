"""Long-lived outbound MCP connections, tool discovery, and invocation."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    ExitStack,
    asynccontextmanager,
)
from typing import Any, Protocol, cast

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import (
    create_mcp_http_client,  # pyright: ignore[reportPrivateImportUsage]
    streamable_http_client,
)

from .types import (
    McpDiagnostic,
    McpConnectionSnapshot,
    McpServerConfig,
    McpToolSpec,
)
from .tool_discovery import discover_tools


logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_SECONDS = 30.0
_REQUEST_TIMEOUT_SECONDS = 60.0


class McpConnectionError(RuntimeError):
    """Raised when the MCP connections cannot complete an operation."""


class McpSession(Protocol):
    """Subset of the official ClientSession used by the session manager."""

    async def initialize(self) -> Any: ...

    async def list_tools(
        self,
        *,
        params: mcp_types.PaginatedRequestParams | None = None,
    ) -> mcp_types.ListToolsResult: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any: ...


McpSessionConnector = Callable[
    [McpServerConfig],
    AbstractAsyncContextManager[McpSession],
]


class McpConnections:
    """Own outbound MCP connections and bridge synchronous calls to asyncio."""

    def __init__(
        self,
        configs: tuple[McpServerConfig, ...],
        *,
        connector: McpSessionConnector | None = None,
        startup_timeout_seconds: float = _STARTUP_TIMEOUT_SECONDS,
        request_timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._configs = configs
        self._connector = connector or _open_mcp_session
        self._startup_timeout_seconds = startup_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._resources: ExitStack | None = None
        self._portal: BlockingPortal | None = None
        self._sessions: dict[str, McpSession] = {}
        self._snapshot = McpConnectionSnapshot(
            configured_servers=len(configs),
        )
        self._started = False
        self._closed = False

    def start(self) -> McpConnectionSnapshot:
        """Open persistent connections and synchronously publish discovery."""
        if self._closed:
            raise McpConnectionError("MCP connections are closed")
        if self._started:
            return self._snapshot
        self._started = True
        if not self._configs:
            return self._snapshot

        resources = ExitStack()
        self._portal = resources.enter_context(
            start_blocking_portal(
                backend="asyncio",
                name="harnessed-coder-mcp",
            )
        )
        self._resources = resources
        try:
            self._snapshot = self._open_all(resources, self._portal)
        except Exception:
            self.close()
            raise
        return self._snapshot

    async def _initialize_and_discover(
        self,
        session: McpSession,
        config: McpServerConfig,
    ) -> tuple[list[McpToolSpec], list[McpDiagnostic]]:
        try:
            await asyncio.wait_for(
                session.initialize(),
                timeout=self._startup_timeout_seconds,
            )
        except TimeoutError as exc:
            raise McpConnectionError(
                "MCP initialization timed out after "
                f"{self._startup_timeout_seconds:g} seconds"
            ) from exc
        try:
            return await asyncio.wait_for(
                discover_tools(session, config),
                timeout=self._request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise McpConnectionError(
                "MCP tool discovery timed out after "
                f"{self._request_timeout_seconds:g} seconds"
            ) from exc

    async def _call_tool_with_timeout(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:
        try:
            return await asyncio.wait_for(
                self._call_tool(server_name, tool_name, arguments),
                timeout=self._request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise McpConnectionError(
                "MCP tool call timed out after "
                f"{self._request_timeout_seconds:g} seconds: "
                f"{server_name}/{tool_name}"
            ) from exc

    def tool_specs(self) -> tuple[McpToolSpec, ...]:
        """Return the immutable filtered tool snapshot."""
        return self._snapshot.tools

    def snapshot(self) -> McpConnectionSnapshot:
        """Return startup discovery and diagnostic information."""
        return self._snapshot

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:
        """Call one remote tool from any synchronous caller thread."""
        if self._closed:
            raise McpConnectionError("MCP connections are closed")
        portal = self._portal
        if portal is None:
            raise McpConnectionError(
                "MCP connections have not been started"
            )

        try:
            return portal.call(
                self._call_tool_with_timeout,
                server_name,
                tool_name,
                arguments,
            )
        except McpConnectionError:
            raise
        except Exception as exc:
            raise McpConnectionError(
                f"MCP tool call failed: {server_name}/{tool_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close sessions, transports, subprocesses, and the portal."""
        if self._closed:
            return
        self._closed = True

        portal = self._portal
        resources = self._resources
        if portal is None or resources is None:
            return
        try:
            resources.close()
        except Exception:
            logger.exception("Failed to close MCP resources cleanly")
        self._portal = None
        self._resources = None
        self._sessions.clear()

    async def _call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:
        try:
            session = self._sessions[server_name]
        except KeyError as exc:
            raise McpConnectionError(
                f"MCP server is unavailable: {server_name}"
            ) from exc
        result = await session.call_tool(
            tool_name,
            arguments,
            read_timeout_seconds=self._request_timeout_seconds,
        )
        if not isinstance(result, mcp_types.CallToolResult):
            raise McpConnectionError(
                f"unsupported MCP result type: {type(result).__name__}"
            )
        return result

    def _open_all(
        self,
        resources: ExitStack,
        portal: BlockingPortal,
    ) -> McpConnectionSnapshot:
        specs: list[McpToolSpec] = []
        diagnostics: list[McpDiagnostic] = []
        connected_servers = 0
        for config in self._configs:
            server_resources = ExitStack()
            try:
                session = server_resources.enter_context(
                    portal.wrap_async_context_manager(
                        self._connector(config)
                    )
                )
                filtered, tool_diagnostics = portal.call(
                    self._initialize_and_discover,
                    session,
                    config,
                )
            except Exception as exc:
                logger.exception(
                    "MCP server connection or discovery failed: %s",
                    config.name,
                )
                try:
                    server_resources.close()
                except Exception:
                    logger.exception(
                        "Failed to close MCP server after startup failure: %s",
                        config.name,
                    )
                diagnostics.append(
                    McpDiagnostic(
                        config.name,
                        f"connection or discovery failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            resources.enter_context(server_resources.pop_all())
            self._sessions[config.name] = session
            connected_servers += 1
            specs.extend(filtered)
            diagnostics.extend(tool_diagnostics)

        return McpConnectionSnapshot(
            tools=tuple(specs),
            diagnostics=tuple(diagnostics),
            configured_servers=len(self._configs),
            connected_servers=connected_servers,
        )


@asynccontextmanager
async def _open_mcp_session(
    config: McpServerConfig,
) -> AsyncIterator[McpSession]:
    if config.transport == "stdio":
        assert config.command is not None
        async with stdio_client(
            StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=config.env or None,
            ),
            errlog=sys.stderr,
        ) as streams:
            async with ClientSession(
                streams[0],
                streams[1],
                read_timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
            ) as session:
                yield cast(McpSession, session)
        return

    assert config.url is not None
    async with create_mcp_http_client(
        headers=config.headers or None,
    ) as http_client:
        async with streamable_http_client(
            config.url,
            http_client=http_client,
        ) as streams:
            async with ClientSession(
                streams[0],
                streams[1],
                read_timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
            ) as session:
                yield cast(McpSession, session)


