"""Shared path handling for workspace-scoped tools."""

from __future__ import annotations

from pathlib import Path


class WorkspaceTool:
    """Mixin for tools that may only access a configured workspace root."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    def _resolve_path(self, value: str, *, allow_root: bool = True) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("path must be a non-empty string")

        raw_path = Path(value)
        target = (raw_path if raw_path.is_absolute() else self.root / raw_path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"path is outside workspace: {value}")
        if not allow_root and target == self.root:
            raise ValueError("path must identify a file or subdirectory")
        return target

    def _relative_name(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
