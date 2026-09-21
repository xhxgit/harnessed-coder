"""Process-wide command shell selection."""

from __future__ import annotations

from functools import cache
import shutil


@cache
def select_shell() -> str:
    """Select the PowerShell executable once for the current process."""
    if shutil.which("pwsh") is not None:
        return "pwsh"
    return "powershell"
