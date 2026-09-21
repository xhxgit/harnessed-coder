"""Domain types for file-based Skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SkillScope = Literal["user", "workspace"]


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Validated metadata discovered without loading a Skill body."""

    name: str
    description: str
    scope: SkillScope
    directory: Path
    skill_file: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    allowed_tools: str | None = None

    @property
    def reference(self) -> str:
        """Return the unambiguous model- and user-visible Skill identifier."""
        return f"{self.scope}:{self.name}"


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """A validated Skill loaded completely from one ``SKILL.md`` file."""

    metadata: SkillMetadata
    instructions: str


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    """A non-fatal discovery problem for one Skill candidate."""

    path: Path
    message: str
