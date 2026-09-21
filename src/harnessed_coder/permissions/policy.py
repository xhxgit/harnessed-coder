"""Deterministic permission policy for tool execution."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .types import PermissionAction, PermissionDecision


class DefaultPermissionPolicy:
    """Conservative first-pass policy for workspace tool execution."""

    def evaluate(self, tool: Any, arguments: dict[str, Any]) -> PermissionDecision:
        path_decision = _workspace_path_decision(tool, arguments)
        if path_decision is not None:
            return path_decision

        if getattr(tool, "name", "") == "bash":
            return evaluate_bash_command(arguments.get("command"))

        metadata = getattr(tool, "metadata", None)
        if metadata is not None and getattr(
            metadata,
            "skip_permission_review",
            False,
        ):
            return PermissionDecision(
                PermissionAction.ALLOW,
                "tool is configured to skip permission review",
            )
        return PermissionDecision(
            PermissionAction.ASK,
            "tool has no explicit permission classification",
        )


def evaluate_bash_command(command: object) -> PermissionDecision:
    """Apply the conservative policy gate used before PowerShell execution.

    This classifier is deliberately not described as process containment. It
    mechanically allows a small, inspectable read-only subset plus common
    dependency-management commands explicitly trusted for daily development.
    Commands outside those subsets require review, while capabilities that
    would bypass review or cross the declared boundary are denied.
    """
    if not isinstance(command, str) or not command.strip():
        return PermissionDecision(
            PermissionAction.ALLOW,
            "command argument validation is handled by the bash tool",
        )

    raw_command = command.strip().lower()
    # A newline is a PowerShell statement boundary. Preserve that meaning so a
    # second command cannot become an ordinary argument during normalization.
    normalized = " ".join(re.sub(r"[\r\n]+", " ; ", raw_command).split())
    if (
        "remove-item" in normalized
        and "-recurse" in normalized
        and ("c:\\" in normalized or "~" in normalized)
    ):
        return PermissionDecision(PermissionAction.DENY, "dangerous commands are disabled")
    if (
        (normalized.startswith("del ") or normalized.startswith("rd "))
        and "/s" in normalized
        and "c:\\" in normalized
    ):
        return PermissionDecision(PermissionAction.DENY, "dangerous commands are disabled")
    if re.search(r"\brm\b", normalized) and "-rf" in normalized and (
        " /" in normalized or " ~" in normalized
    ):
        return PermissionDecision(PermissionAction.DENY, "dangerous commands are disabled")

    if _matches_any(normalized, _DANGEROUS_PATTERNS):
        return PermissionDecision(PermissionAction.DENY, "dangerous commands are disabled")

    if normalized in {
        "cmd",
        "cmd.exe",
        "pwsh",
        "pwsh.exe",
        "powershell",
        "powershell.exe",
        "bash",
        "sh",
        "wsl",
        "node",
        "python",
        "py",
    }:
        return PermissionDecision(PermissionAction.DENY, "interactive shells are disabled")
    if normalized.startswith("python -i") or normalized.startswith("py -i"):
        return PermissionDecision(PermissionAction.DENY, "interactive commands are disabled")

    if _matches_any(normalized, _DYNAMIC_EXECUTION_PATTERNS):
        return PermissionDecision(
            PermissionAction.DENY,
            "dynamic, encoded, or nested shell execution is disabled",
        )

    if _matches_any(normalized, _WORKSPACE_ESCAPE_PATTERNS):
        return PermissionDecision(
            PermissionAction.DENY,
            "command contains an explicit workspace escape",
        )

    if _is_common_dependency_management_command(normalized):
        return PermissionDecision(
            PermissionAction.ALLOW,
            "common dependency management command is explicitly trusted",
        )

    if _matches_any(normalized, _NETWORK_REVIEW_PATTERNS):
        return PermissionDecision(
            PermissionAction.ASK,
            "network command requires permission review",
        )

    if _matches_any(normalized, _CONFIRMATION_PATTERNS):
        return PermissionDecision(
            PermissionAction.ASK,
            "shell command may modify or delete files and requires confirmation",
        )

    if _is_mechanically_read_only_command(normalized):
        return PermissionDecision(
            PermissionAction.ALLOW,
            "command is in the mechanically verified local read-only subset",
        )

    return PermissionDecision(
        PermissionAction.ASK,
        "command is outside the mechanically verified local read-only subset",
    )


def permission_block_message(decision: PermissionDecision) -> str:
    """Return the tool result used when a permission check blocks execution."""
    if decision.action == PermissionAction.DENY:
        if decision.resolution in {"user_denied", "user_denied_after_abstain"}:
            return f"Permission denied by user.\nReason: {decision.reason}."
        if decision.resolution == "auto_denied":
            return (
                "Permission denied by automatic review.\n"
                f"Reason: {decision.reason}.\n"
                f"Review: {decision.resolution_reason or 'no reason provided'}."
            )
        if decision.resolution == "auto_abstained":
            return (
                "Permission denied because automatic review abstained and "
                "interactive approval is unavailable.\n"
                f"Reason: {decision.reason}."
            )
        if decision.resolution == "approval_unavailable":
            return (
                "Permission denied because interactive approval is unavailable.\n"
                f"Reason: {decision.reason}."
            )
        if decision.resolution == "approval_interrupted":
            return (
                "Permission denied because approval was interrupted.\n"
                f"Reason: {decision.reason}."
            )
        if decision.resolution == "approval_error":
            return (
                "Permission denied because approval failed.\n"
                f"Reason: {decision.reason}."
            )
        return f"Permission denied before execution.\nReason: {decision.reason}."
    if decision.action == PermissionAction.ASK:
        return (
            "Permission confirmation required before execution.\n"
            f"Reason: {decision.reason}.\n"
            "This call must be resolved by an interactive permission approver."
        )
    raise ValueError("allow decisions do not produce a block message")


def _workspace_path_decision(
    tool: Any,
    arguments: dict[str, Any],
) -> PermissionDecision | None:
    root = getattr(tool, "root", None)
    path = arguments.get("path")
    if root is None or not isinstance(path, str) or not path.strip():
        return None

    try:
        workspace_root = Path(root).resolve()
        raw_path = Path(path)
        target = (raw_path if raw_path.is_absolute() else workspace_root / raw_path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return PermissionDecision(PermissionAction.DENY, f"invalid workspace path: {exc}")

    if not target.is_relative_to(workspace_root):
        return PermissionDecision(
            PermissionAction.DENY,
            f"path is outside workspace: {path}",
        )
    return None


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def _is_mechanically_read_only_command(command: str) -> bool:
    """Recognize a narrow PowerShell pipeline without executable syntax."""
    if _matches_any(command, _REVIEW_REQUIRED_SYNTAX_PATTERNS):
        return False

    segments = [segment.strip() for segment in command.split("|")]
    if not segments or any(not segment for segment in segments):
        return False

    for segment in segments:
        words = segment.split()
        if not words:
            return False
        executable = words[0]
        if executable not in _READ_ONLY_POWERSHELL_COMMANDS:
            return False
    return True


def _is_common_dependency_management_command(command: str) -> bool:
    """Allow a direct dependency command, never a compound shell statement."""
    if _matches_any(command, _REVIEW_REQUIRED_SYNTAX_PATTERNS):
        return False
    return _matches_any(command, _DEPENDENCY_MANAGEMENT_PATTERNS)


_DEPENDENCY_MANAGEMENT_PATTERNS = [
    r"^(?:\s*[^\s;|&]*[\\/])?(?:pip|pip3)(?:\.exe)?\s+(?:install|download)\b",
    r"^(?:(?:[^\s;|&]*[\\/])?python(?:\.exe)?|py(?:\.exe)?)\s+-m\s+pip\s+(?:install|download)\b",
    r"^(?:[^\s;|&]*[\\/])?uv(?:\.exe)?\s+(?:sync|add|pip\s+install)\b",
    r"^(?:[^\s;|&]*[\\/])?(?:npm|pnpm|yarn)(?:\.cmd|\.exe)?\s+(?:install|add|ci)\b",
    r"^(?:[^\s;|&]*[\\/])?dotnet(?:\.exe)?\s+(?:restore|add\b.*\bpackage)\b",
    r"^(?:[^\s;|&]*[\\/])?cargo(?:\.exe)?\s+(?:install|fetch)\b",
    r"^(?:[^\s;|&]*[\\/])?go(?:\.exe)?\s+(?:get|install|mod\s+download)\b",
    r"^(?:[^\s;|&]*[\\/])?nuget(?:\.exe)?\s+(?:install|restore)\b",
]

_NETWORK_REVIEW_PATTERNS = [
    r"https?://",
    r"ftp://",
    r"\bcurl\b",
    r"\bwget\b",
    r"\binvoke-webrequest\b",
    r"\biwr\b",
    r"\binvoke-restmethod\b",
    r"(?:^|[;|&]\s*)irm(?:\s|$)",
    r"\b(system\.net|net\.webclient|httpclient|downloadstring|downloadfile)\b",
    r"(?:^|[;|&]\s*)(ssh|scp|sftp|ftp|telnet|nc|ncat|bitsadmin)(?:\s|$)",
    r"\b(pip|pip3|uv)\s+(install|download|sync|add)\b",
    r"\b(?:python(?:\.exe)?|py(?:\.exe)?)\s+-m\s+pip\s+(?:install|download)\b",
    r"\b(npm|pnpm|yarn)\s+(install|add|ci)\b",
    r"\b(dotnet\s+restore|dotnet\s+add\b.*\bpackage)\b",
    r"\b(cargo\s+install|go\s+(get|install))\b",
    r"\b(winget|choco|scoop|nuget)\b",
    r"\bgit\b[^;|&]*\b(clone|pull|fetch)\b",
    r"\bgit\s+submodule\s+(add|update)\b",
]

_DYNAMIC_EXECUTION_PATTERNS = [
    r"(?:^|[;|&]\s*)(invoke-expression|iex)(?:\s|$)",
    r"\b(encodedcommand|encodedarguments)\b",
    r"\b(start-process|start-job|invoke-command)\b",
    r"\b(enter-pssession|new-pssession|register-objectevent)\b",
    r"\badd-type\b",
    r"\b(python(?:\.exe)?|py|node(?:\.exe)?|ruby(?:\.exe)?|perl(?:\.exe)?)\s+(-c|-e|--eval|-)(?:\s|$)",
    r"\[scriptblock\]\s*::\s*create\b",
    r"\.invoke\s*\(",
    r"(?:^|\s)&\s*(?:\$|\()",
    r"(?:^|\s)\.\s*(?:\$|\()",
    r"(?:^|[;|(&]\s*)&?\s*['\"]?(cmd(?:\.exe)?|pwsh(?:\.exe)?|powershell(?:\.exe)?|bash|sh|wsl)['\"]?(?:\s|$)",
]

_WORKSPACE_ESCAPE_PATTERNS = [
    r"(?:^|[\s\"'=,(])(?:[a-z]:[\\/])",
    r"(?:^|[\s\"'=,(])\\\\[^\\]",
    r"(?:^|[\s\"'=,(])[\\/](?![\\/])[^\\/\s]",
    r"(?:^|[\s\"'=,(\\/])\.\.(?:[\\/]|$)",
    r"(?:^|[\s\"'=,(])~(?:[\\/]|$)",
    r"\$(home|env:(userprofile|homedrive|homepath))\b",
    r"\$pwd\.parent\b",
    r"\[environment\]\s*::\s*getfolderpath\b",
    r"\b(env|registry|cert|wsman|function|variable|alias):",
    r"\b(hkcu|hklm):",
    r"(?:^|[;|&]\s*)(set-location|push-location|pop-location|cd|chdir|sl)(?:\s|$)",
]

_DANGEROUS_PATTERNS = [
    r"\bformat\b",
    r"\bdiskpart\b",
    r"\bmkfs\b",
    r"\bstart-process\b.*\b-verb\s+runas\b",
    r"\brunas\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bgit\b[^;|&]*\breset\b[^;|&]*--hard\b",
    r"\bgit\b[^;|&]*\b(clean|push)\b",
]

_CONFIRMATION_PATTERNS = [
    r"\bremove-item\b",
    r"(?:^|[;|&]\s*)(ri|erase|rmdir)(?:\s|$)",
    r"\bdel\b",
    r"\brd\b",
    r"\brm\b",
    r"\bmove-item\b",
    r"\b(copy-item|rename-item)\b",
    r"\bset-content\b",
    r"\b(add-content|clear-content)\b",
    r"\bout-file\b",
    r"\bnew-item\b",
    r"\b(set-item|set-itemproperty|new-itemproperty|remove-itemproperty)\b",
    r"(?:^|[;|&]\s*)(ni|sc|ac|clc|cp|mv|ren)(?:\s|$)",
    r"\b(git\s+(add|commit|checkout|switch|merge|rebase|restore|tag|stash))\b",
    r"(^|[^<])>(?!>)|>>",
]

_REVIEW_REQUIRED_SYNTAX_PATTERNS = [
    r"[;&{}()\[\]]",
    r"&&|\|\|",
    r"[<>]",
    r"`",
    r"\$\(",
    r"@\(",
    r"\$",
    r"\$[a-z_][\w:]*\s*=",
]

_READ_ONLY_POWERSHELL_COMMANDS = {
    "compare-object",
    "convertfrom-json",
    "convertto-json",
    "format-list",
    "format-table",
    "get-acl",
    "get-childitem",
    "get-command",
    "get-content",
    "get-date",
    "get-filehash",
    "get-item",
    "get-itemproperty",
    "get-location",
    "group-object",
    "measure-object",
    "out-string",
    "resolve-path",
    "select-object",
    "select-string",
    "sort-object",
    "test-path",
    "where-object",
    "write-output",
}
