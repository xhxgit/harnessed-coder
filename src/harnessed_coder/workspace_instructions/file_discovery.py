"""Load the workspace-root ``AGENTS.md`` snapshot."""

from __future__ import annotations

from pathlib import Path

from .types import AgentsInstructionDiagnostic, AgentsInstructions


_MAX_FILE_BYTES = 64 * 1024


def load_agents_instructions(
    workspace_root: str | Path,
) -> tuple[AgentsInstructions | None, AgentsInstructionDiagnostic | None]:
    """Load ``<workspace-root>/AGENTS.md`` when it exists and is valid."""
    root = Path(workspace_root).resolve()
    path = root / "AGENTS.md"
    if not path.exists():
        return None, None
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        return None, AgentsInstructionDiagnostic(
            path,
            f"Cannot resolve AGENTS.md: {exc}",
        )
    if not resolved_path.is_relative_to(root):
        return None, AgentsInstructionDiagnostic(
            path,
            "AGENTS.md escapes the workspace",
        )
    if not resolved_path.is_file():
        return None, AgentsInstructionDiagnostic(
            path,
            "AGENTS.md is not a file",
        )
    try:
        size = resolved_path.stat().st_size
    except OSError as exc:
        return None, AgentsInstructionDiagnostic(
            path,
            f"Cannot inspect AGENTS.md: {exc}",
        )
    if size > _MAX_FILE_BYTES:
        return None, AgentsInstructionDiagnostic(
            path,
            f"AGENTS.md exceeds {_MAX_FILE_BYTES} bytes",
        )
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, AgentsInstructionDiagnostic(
            path,
            f"Cannot read AGENTS.md: {exc}",
        )
    if not content.strip():
        return None, AgentsInstructionDiagnostic(path, "AGENTS.md is empty")
    return AgentsInstructions(path=resolved_path, content=content.strip()), None
