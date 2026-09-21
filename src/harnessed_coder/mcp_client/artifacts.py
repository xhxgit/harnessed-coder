"""Persistent storage for non-text content returned by MCP tools."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from harnessed_coder.session import resolve_data_dir


_ARTIFACT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_KNOWN_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class McpArtifactError(RuntimeError):
    """Raised when an MCP artifact is missing, corrupt, or unsafe."""


@dataclass(frozen=True, slots=True)
class McpArtifactRecord:
    """Verified metadata for one locally stored MCP artifact."""

    artifact_id: str
    filename: str
    mime_type: str
    size: int
    sha256: str
    server_name: str
    tool_name: str
    created_at: str
    path: Path


class McpArtifactStore:
    """Store opaque MCP bytes outside the workspace and retrieve them by ID."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.root = (resolve_data_dir(data_dir) / "mcp" / "artifacts").resolve()

    def save(
        self,
        data: bytes,
        *,
        mime_type: str | None,
        server_name: str,
        tool_name: str,
        suggested_name: str | None = None,
    ) -> McpArtifactRecord:
        """Persist bytes plus integrity metadata and return their opaque ID."""
        normalized_mime = (mime_type or "application/octet-stream").strip()
        artifact_id = uuid4().hex
        filename = _artifact_filename(
            artifact_id,
            normalized_mime,
            suggested_name,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        artifact_path = (self.root / filename).resolve()
        metadata_path = (self.root / f"{artifact_id}.json").resolve()
        if (
            not artifact_path.is_relative_to(self.root)
            or not metadata_path.is_relative_to(self.root)
        ):
            raise McpArtifactError("resolved artifact path escaped storage root")

        digest = hashlib.sha256(data).hexdigest()
        created_at = datetime.now(UTC).isoformat()
        with artifact_path.open("xb") as artifact_file:
            artifact_file.write(data)
        metadata = {
            "artifact_id": artifact_id,
            "filename": filename,
            "mime_type": normalized_mime,
            "size": len(data),
            "sha256": digest,
            "server_name": server_name,
            "tool_name": tool_name,
            "created_at": created_at,
        }
        try:
            with metadata_path.open("x", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
                metadata_file.write("\n")
        except Exception:
            artifact_path.unlink(missing_ok=True)
            raise
        return McpArtifactRecord(**metadata, path=artifact_path)

    def get(self, artifact_id: str) -> McpArtifactRecord:
        """Load an artifact by ID and verify its path, length, and hash."""
        normalized_id = artifact_id.strip().lower()
        if not _ARTIFACT_ID_PATTERN.fullmatch(normalized_id):
            raise McpArtifactError("artifact_id must be 32 lowercase hex characters")
        metadata_path = (self.root / f"{normalized_id}.json").resolve()
        if not metadata_path.is_relative_to(self.root):
            raise McpArtifactError("artifact metadata path escaped storage root")
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise McpArtifactError(f"MCP artifact not found: {normalized_id}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise McpArtifactError(
                f"cannot read MCP artifact metadata: {normalized_id}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise McpArtifactError("MCP artifact metadata must be an object")

        required = {
            "artifact_id": str,
            "filename": str,
            "mime_type": str,
            "size": int,
            "sha256": str,
            "server_name": str,
            "tool_name": str,
            "created_at": str,
        }
        for key, expected_type in required.items():
            if not isinstance(raw.get(key), expected_type):
                raise McpArtifactError(f"invalid MCP artifact metadata field: {key}")
        if raw["artifact_id"] != normalized_id:
            raise McpArtifactError("MCP artifact metadata ID mismatch")

        artifact_path = (self.root / raw["filename"]).resolve()
        if not artifact_path.is_relative_to(self.root):
            raise McpArtifactError("artifact file path escaped storage root")
        try:
            data = artifact_path.read_bytes()
        except OSError as exc:
            raise McpArtifactError(
                f"cannot read MCP artifact: {normalized_id}: {exc}"
            ) from exc
        if len(data) != raw["size"]:
            raise McpArtifactError("MCP artifact size verification failed")
        if hashlib.sha256(data).hexdigest() != raw["sha256"]:
            raise McpArtifactError("MCP artifact hash verification failed")
        return McpArtifactRecord(
            artifact_id=raw["artifact_id"],
            filename=raw["filename"],
            mime_type=raw["mime_type"],
            size=raw["size"],
            sha256=raw["sha256"],
            server_name=raw["server_name"],
            tool_name=raw["tool_name"],
            created_at=raw["created_at"],
            path=artifact_path,
        )


def _artifact_filename(
    artifact_id: str,
    mime_type: str,
    suggested_name: str | None,
) -> str:
    if suggested_name:
        safe_name = _SAFE_FILENAME_CHARS.sub("_", Path(suggested_name).name)
        safe_name = safe_name.strip("._")
        if safe_name:
            return f"{artifact_id}_{safe_name[:100]}"
    extension = _KNOWN_EXTENSIONS.get(mime_type)
    if extension is None:
        guessed = mimetypes.guess_extension(mime_type, strict=False)
        extension = guessed if guessed and re.fullmatch(r"\.[A-Za-z0-9]+", guessed) else ".bin"
    return f"{artifact_id}{extension}"
