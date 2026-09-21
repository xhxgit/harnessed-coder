"""Resolve and maintain the on-disk layout used by session storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from ..constants import DATA_DIR_NAME, DEFAULT_SESSION_NAME, WORKSPACE_METADATA_FILE_NAME
from .conversation_session import SESSION_VERSION


_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_session_path(
    workspace_root: str | Path,
    *,
    workspace_name: str | None = None,
    session_name: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve the session file path for a workspace/session pair.

    By default sessions live under:

    ``~/.harnessed-coder/workspaces/<workspace-id>/sessions/<session>.json``

    Session names are normalized to safe path segments so application data stays
    under the selected workspace data directory.
    """
    workspace_data_dir = resolve_workspace_data_dir(
        workspace_root,
        workspace_name=workspace_name,
        data_dir=data_dir,
    )
    session_id = _safe_segment(session_name or DEFAULT_SESSION_NAME)
    return (workspace_data_dir / "sessions" / f"{session_id}.json").resolve()


def resolve_session_trace_path(
    workspace_root: str | Path,
    *,
    workspace_name: str | None = None,
    session_name: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve the persistent JSONL business trace for one session."""
    workspace_data_dir = resolve_workspace_data_dir(
        workspace_root,
        workspace_name=workspace_name,
        data_dir=data_dir,
    )
    session_id = _safe_segment(session_name or DEFAULT_SESSION_NAME)
    return (workspace_data_dir / "traces" / f"{session_id}.jsonl").resolve()


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the root directory used for user-level harnessed-coder data."""
    return Path(data_dir or Path.home() / DATA_DIR_NAME).expanduser().resolve()


def resolve_workspace_data_dir(
    workspace_root: str | Path,
    *,
    workspace_name: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Return the per-workspace data directory in the user data root."""
    workspace_id = _workspace_id(workspace_root, workspace_name=workspace_name)
    return (resolve_data_dir(data_dir) / "workspaces" / workspace_id).resolve()


def write_workspace_metadata(
    workspace_root: str | Path,
    *,
    workspace_name: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Persist metadata describing the workspace slot and return its path."""
    workspace = Path(workspace_root).resolve()
    workspace_data_dir = resolve_workspace_data_dir(
        workspace,
        workspace_name=workspace_name,
        data_dir=data_dir,
    )
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = workspace_data_dir / WORKSPACE_METADATA_FILE_NAME
    data = {
        "version": SESSION_VERSION,
        "workspace_root": str(workspace),
        "workspace_name": workspace_name,
    }
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=workspace_data_dir,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_file.write("\n")
    temp_path.replace(metadata_path)
    return metadata_path


def _workspace_id(workspace_root: str | Path, *, workspace_name: str | None = None) -> str:
    if workspace_name:
        return _safe_segment(workspace_name)
    workspace = Path(workspace_root).resolve()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:12]
    stem = _safe_segment(workspace.name or "workspace")
    return f"{stem}-{digest}"


def _safe_segment(value: str) -> str:
    normalized = _SAFE_NAME_PATTERN.sub("-", value.strip())
    normalized = normalized.strip(".-_")
    if not normalized:
        raise ValueError("workspace/session names must contain at least one safe character")
    return normalized[:80]
