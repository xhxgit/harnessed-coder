"""Import a verified MCP artifact into the active workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from harnessed_coder.mcp_client.artifacts import (
    McpArtifactError,
    McpArtifactStore,
)

from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata


class ImportMcpArtifactTool(WorkspaceTool, Tool):
    """Copy a user-data MCP artifact into a workspace path."""

    name: ClassVar[str] = "import_mcp_artifact"
    description: ClassVar[str] = (
        "Copy a non-text artifact returned by an MCP tool into the workspace. "
        "The source must be referenced by its opaque artifact ID."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=True,
        result_size_hint="small",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Opaque artifact ID returned by an MCP tool.",
            },
            "destination": {
                "type": "string",
                "description": "Destination file path inside the workspace.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to replace an existing destination file.",
                "default": False,
            },
        },
        "required": ["artifact_id", "destination"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        root: str | Path | None,
        artifact_store: McpArtifactStore,
    ) -> None:
        super().__init__(root)
        self._artifact_store = artifact_store

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"artifact_id", "destination", "overwrite"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"
        artifact_id = kwargs.get("artifact_id")
        destination = kwargs.get("destination")
        overwrite = kwargs.get("overwrite", False)
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            return "Error: artifact_id must be a non-empty string"
        if not isinstance(destination, str) or not destination.strip():
            return "Error: destination must be a non-empty string"
        if not isinstance(overwrite, bool):
            return "Error: overwrite must be a boolean"

        temp_path: Path | None = None
        try:
            record = self._artifact_store.get(artifact_id)
            target = self._resolve_path(destination, allow_root=False)
            if not target.parent.is_dir():
                return (
                    "Error: parent directory not found: "
                    f"{self._relative_name(target.parent)}"
                )
            if target.exists():
                if not target.is_file():
                    return f"Error: destination is not a file: {destination}"
                if not overwrite:
                    return f"Error: destination already exists: {destination}"

            data = record.path.read_bytes()
            if overwrite:
                temp_path = target.parent / f".{target.name}.{uuid4().hex}.tmp"
                with temp_path.open("xb") as output_file:
                    output_file.write(data)
                temp_path.replace(target)
            else:
                with target.open("xb") as output_file:
                    output_file.write(data)
        except (McpArtifactError, OSError, ValueError) as exc:
            return f"Error: {exc}"
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return (
            f"Imported MCP artifact {record.artifact_id} to "
            f"{self._relative_name(target)} ({record.size} bytes, "
            f"SHA-256 {record.sha256})."
        )
