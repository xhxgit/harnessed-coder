"""In-memory catalog for discovered file-based Skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .file_discovery import discover_skills
from .types import SkillDefinition, SkillDiagnostic, SkillMetadata


@dataclass(frozen=True, slots=True)
class _SkillSnapshot:
    definitions: tuple[SkillDefinition, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


class SkillCatalog:
    """Own and resolve an immutable snapshot of fully loaded Skills."""

    def __init__(
        self,
        skills: list[SkillDefinition] | None = None,
        *,
        diagnostics: list[SkillDiagnostic] | None = None,
        workspace_root: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self._snapshot = self._make_snapshot(skills or [], diagnostics or [])
        self._workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root is not None
            else None
        )
        self._data_dir = (
            Path(data_dir).resolve()
            if data_dir is not None
            else None
        )

    @classmethod
    def discover(
        cls,
        workspace_root: str | Path,
        *,
        data_dir: str | Path | None = None,
    ) -> SkillCatalog:
        skills, diagnostics = discover_skills(
            workspace_root,
            data_dir=data_dir,
        )
        return cls(
            skills,
            diagnostics=diagnostics,
            workspace_root=workspace_root,
            data_dir=data_dir,
        )

    def reload(self) -> None:
        """Replace the current snapshot with a fresh complete disk load."""
        if self._workspace_root is None:
            raise ValueError("Skill Catalog has no discovery source to reload")
        skills, diagnostics = discover_skills(
            self._workspace_root,
            data_dir=self._data_dir,
        )
        snapshot = self._make_snapshot(skills, diagnostics)
        self._snapshot = snapshot

    def list(self) -> list[SkillMetadata]:
        """Return metadata from the current in-memory snapshot."""
        return [
            definition.metadata
            for definition in self._snapshot.definitions
        ]

    def diagnostics(self) -> list[SkillDiagnostic]:
        """Return non-fatal discovery diagnostics."""
        return list(self._snapshot.diagnostics)

    def resolve(self, reference: str) -> SkillDefinition:
        """Resolve a qualified reference or an unambiguous short Skill name."""
        normalized = reference.strip()
        if not normalized:
            raise ValueError("Skill name must not be empty")
        qualified = [
            definition
            for definition in self._snapshot.definitions
            if definition.metadata.reference == normalized
        ]
        if qualified:
            return qualified[0]
        short_matches = [
            definition
            for definition in self._snapshot.definitions
            if definition.metadata.name == normalized
        ]
        if not short_matches:
            raise ValueError(f"Unknown Skill: {normalized}")
        if len(short_matches) > 1:
            choices = ", ".join(
                definition.metadata.reference
                for definition in short_matches
            )
            raise ValueError(
                f"Ambiguous Skill {normalized!r}; use one of: {choices}"
            )
        return short_matches[0]

    def load(self, reference: str, *, arguments: str = "") -> SkillDefinition:
        """Render one Skill from the current in-memory snapshot."""
        definition = self.resolve(reference)
        metadata = definition.metadata
        instructions = (
            definition.instructions
            .replace("${SKILL_DIR}", metadata.directory.as_posix())
            .replace("$ARGUMENTS", arguments)
        )
        return SkillDefinition(metadata=metadata, instructions=instructions)

    @staticmethod
    def _make_snapshot(
        skills: list[SkillDefinition],
        diagnostics: list[SkillDiagnostic],
    ) -> _SkillSnapshot:
        definitions = tuple(
            sorted(
                skills,
                key=lambda item: (
                    item.metadata.name,
                    item.metadata.scope,
                ),
            )
        )
        references = [
            definition.metadata.reference
            for definition in definitions
        ]
        if len(references) != len(set(references)):
            raise ValueError("Duplicate Skill reference discovered")
        return _SkillSnapshot(definitions, tuple(diagnostics))
