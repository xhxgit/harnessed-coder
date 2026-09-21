"""Domain types for workspace ``AGENTS.md`` instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AgentsInstructions:
    """The fully loaded workspace-root ``AGENTS.md`` snapshot."""

    path: Path
    content: str


@dataclass(frozen=True, slots=True)
class AgentsInstructionDiagnostic:
    """A non-fatal problem encountered while loading ``AGENTS.md``."""

    path: Path
    message: str
