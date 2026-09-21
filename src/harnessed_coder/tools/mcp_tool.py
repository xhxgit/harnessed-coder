"""Dynamic Tool adapters and MCP result conversion."""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any, ClassVar, cast

from mcp import types as mcp_types

from harnessed_coder.mcp_client.artifacts import McpArtifactStore
from harnessed_coder.mcp_client.mcp_connections import (
    McpConnectionError,
    McpConnections,
)
from harnessed_coder.mcp_client.types import McpToolSpec

from .base import Tool, ToolMetadata


class McpTool(Tool):
    """Call one filtered remote MCP tool through shared MCP connections."""

    name: ClassVar[str] = "mcp__unbound"
    description: ClassVar[str] = "Unbound MCP tool."
    parameters: ClassVar[dict[str, Any]] = {"type": "object"}
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=False,
        result_size_hint="large",
    )

    def __init__(
        self,
        spec: McpToolSpec,
        connections: McpConnections,
        artifact_store: McpArtifactStore,
    ) -> None:
        self._spec = spec
        self._connections = connections
        self._artifact_store = artifact_store

    def execute(self, **kwargs: Any) -> str:
        try:
            result = self._connections.call_tool(
                self._spec.server_name,
                self._spec.remote_name,
                kwargs,
            )
            return format_mcp_result(
                result,
                artifact_store=self._artifact_store,
                server_name=self._spec.server_name,
                tool_name=self._spec.remote_name,
            )
        except (McpConnectionError, OSError, ValueError) as exc:
            return f"Error: {exc}"


def create_mcp_tool(
    spec: McpToolSpec,
    connections: McpConnections,
    artifact_store: McpArtifactStore,
) -> McpTool:
    """Create a Tool subclass whose class-level definition matches one spec."""
    dynamic_class = cast(
        type[McpTool],
        type(
            f"McpTool_{spec.local_name}",
            (McpTool,),
            {
                "name": spec.local_name,
                "description": spec.description,
                "parameters": cast(dict[str, Any], spec.input_schema),
            },
        ),
    )
    return dynamic_class(spec, connections, artifact_store)


def format_mcp_result(
    result: mcp_types.CallToolResult,
    *,
    artifact_store: McpArtifactStore,
    server_name: str,
    tool_name: str,
) -> str:
    """Convert MCP content blocks into model-readable text and artifact IDs."""
    parts: list[str] = []
    for content in result.content:
        if isinstance(content, mcp_types.TextContent):
            parts.append(content.text)
        elif isinstance(content, (mcp_types.ImageContent, mcp_types.AudioContent)):
            parts.append(
                _save_base64_artifact(
                    content.data,
                    mime_type=content.mime_type,
                    artifact_store=artifact_store,
                    server_name=server_name,
                    tool_name=tool_name,
                )
            )
        elif isinstance(content, mcp_types.ResourceLink):
            details = [f"MCP resource link: {content.name}", f"URI: {content.uri}"]
            if content.mime_type:
                details.append(f"MIME type: {content.mime_type}")
            if content.description:
                details.append(f"Description: {content.description}")
            parts.append("\n".join(details))
        elif isinstance(content, mcp_types.EmbeddedResource):
            resource = content.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                parts.append(f"MCP embedded resource ({resource.uri}):\n{resource.text}")
            else:
                parts.append(
                    _save_base64_artifact(
                        resource.blob,
                        mime_type=resource.mime_type,
                        artifact_store=artifact_store,
                        server_name=server_name,
                        tool_name=tool_name,
                        suggested_name=Path(str(resource.uri)).name or None,
                    )
                )
        else:
            parts.append(
                json.dumps(
                    content.model_dump(by_alias=True),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

    if result.structured_content is not None:
        parts.append(
            "Structured content:\n"
            + json.dumps(
                result.structured_content,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    if not parts:
        parts.append("MCP tool returned no content.")
    body = "\n\n".join(parts)
    if result.is_error:
        return f"Error: MCP server reported a tool failure.\n{body}"
    return body


def _save_base64_artifact(
    encoded: str,
    *,
    mime_type: str | None,
    artifact_store: McpArtifactStore,
    server_name: str,
    tool_name: str,
    suggested_name: str | None = None,
) -> str:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"MCP returned invalid base64 content: {exc}") from exc
    record = artifact_store.save(
        data,
        mime_type=mime_type,
        server_name=server_name,
        tool_name=tool_name,
        suggested_name=suggested_name,
    )
    return (
        "Saved non-text MCP content as an artifact.\n"
        f"Artifact ID: {record.artifact_id}\n"
        f"Filename: {record.filename}\n"
        f"MIME type: {record.mime_type}\n"
        f"Size: {record.size} bytes\n"
        f"SHA-256: {record.sha256}\n"
        "Use import_mcp_artifact with this artifact ID to copy it into "
        "the workspace."
    )
