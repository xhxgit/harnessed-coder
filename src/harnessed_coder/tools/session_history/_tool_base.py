"""Base class for model-visible saved-session history tools."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from harnessed_coder.tools.base import Tool, ToolMetadata

from ._shared import SessionHistoryReader


class SessionHistoryToolBase(Tool):
    """Common setup for read-only saved-session tools."""

    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=True,
        is_parallel_safe=True,
        skip_permission_review=True,
        result_size_hint="paged",
    )

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.history = SessionHistoryReader(data_dir)

    @property
    def data_dir(self) -> Path:
        return self.history.data_dir
