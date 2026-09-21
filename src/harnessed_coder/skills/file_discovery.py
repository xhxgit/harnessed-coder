"""Discover user- and workspace-scoped file-based Skills."""

from __future__ import annotations

from pathlib import Path

from .skill_file import parse_skill_file
from .types import SkillDefinition, SkillDiagnostic, SkillScope


_PROJECT_SKILLS_RELATIVE_PATH = Path(".harnessed-coder") / "skills"
_USER_SKILLS_DIR_NAME = "skills"


def discover_skills(
    workspace_root: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> tuple[list[SkillDefinition], list[SkillDiagnostic]]:
    """Return fully loaded Skills plus non-fatal discovery diagnostics."""
    sources: list[tuple[SkillScope, Path]] = []
    if data_dir is not None:
        sources.append(("user", Path(data_dir).resolve() / _USER_SKILLS_DIR_NAME))
    sources.append(
        (
            "workspace",
            Path(workspace_root).resolve() / _PROJECT_SKILLS_RELATIVE_PATH,
        )
    )

    skills: list[SkillDefinition] = []
    diagnostics: list[SkillDiagnostic] = []
    for scope, root in sources:
        discovered, source_diagnostics = _discover_source(root, scope=scope)
        skills.extend(discovered)
        diagnostics.extend(source_diagnostics)
    return skills, diagnostics


def _discover_source(
    root: Path,
    *,
    scope: SkillScope,
) -> tuple[list[SkillDefinition], list[SkillDiagnostic]]:
    if not root.exists():
        return [], []
    if not root.is_dir():
        return [], [SkillDiagnostic(root, "Skills path is not a directory")]

    resolved_root = root.resolve()
    skills: list[SkillDefinition] = []
    diagnostics: list[SkillDiagnostic] = []
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return [], [SkillDiagnostic(root, f"Cannot list Skills directory: {exc}")]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            resolved_directory = candidate.resolve(strict=True)
        except OSError as exc:
            diagnostics.append(
                SkillDiagnostic(candidate, f"Cannot resolve Skill directory: {exc}")
            )
            continue
        if not resolved_directory.is_relative_to(resolved_root):
            diagnostics.append(
                SkillDiagnostic(candidate, "Skill directory escapes its source root")
            )
            continue

        skill_file = resolved_directory / "SKILL.md"
        if not skill_file.is_file():
            diagnostics.append(
                SkillDiagnostic(candidate, "Skill directory has no SKILL.md")
            )
            continue
        try:
            resolved_skill_file = skill_file.resolve(strict=True)
        except OSError as exc:
            diagnostics.append(
                SkillDiagnostic(skill_file, f"Cannot resolve SKILL.md: {exc}")
            )
            continue
        if not resolved_skill_file.is_relative_to(resolved_directory):
            diagnostics.append(
                SkillDiagnostic(skill_file, "SKILL.md escapes its Skill directory")
            )
            continue
        try:
            definition = parse_skill_file(
                resolved_skill_file,
                scope=scope,
                directory=resolved_directory,
            )
        except ValueError as exc:
            diagnostics.append(SkillDiagnostic(skill_file, str(exc)))
            continue
        skills.append(definition)

    return skills, diagnostics
