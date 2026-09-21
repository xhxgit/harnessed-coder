"""Run Windows PowerShell commands in the workspace."""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from ..shell import select_shell
from ._paths import WorkspaceTool
from .base import Tool, ToolMetadata

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


class BashTool(WorkspaceTool, Tool):
    """Execute a policy-gated, non-interactive PowerShell command."""

    name: ClassVar[str] = "bash"
    description: ClassVar[str] = (
        "Run a policy-gated PowerShell command with the workspace root as cwd. "
        "Use this for tests, builds, and local project checks. Approved commands "
        "are not isolated from the current user account by an OS sandbox."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=False,
        result_size_hint="bounded",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "PowerShell command to run in the workspace root.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
                "description": "Maximum seconds to allow the command to run.",
                "default": DEFAULT_TIMEOUT_SECONDS,
            },
            "strip_ansi": {
                "type": "boolean",
                "description": (
                    "Strip ANSI escape sequences and common control characters "
                    "from stdout/stderr before returning output. Set false when "
                    "diagnosing terminal colors or control-code rendering."
                ),
                "default": True,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__(root)
        self.shell = select_shell()

    def definition(self) -> dict[str, Any]:
        definition = super().definition()
        function = definition["function"]
        function["description"] = (
            f"{self.description} The selected shell for this session is "
            f"{self.shell}. Generate commands specifically for {self.shell}. "
            "Common direct dependency-management commands are allowed. Other "
            "recognized network access and unknown commands require permission "
            "review. Explicit workspace escape, nested shells, and dynamic or "
            "encoded execution are denied. "
            "Commands that may execute project code also require permission "
            "review. "
            "Do not use syntax, modules, or behavior that only work in the other "
            "PowerShell runtime. Do not prefix commands with pwsh or powershell "
            "because this tool adds the shell."
        )
        command_schema = function["parameters"]["properties"]["command"]
        command_schema["description"] = (
            f"Command body to run in the workspace root using {self.shell}. "
            "Use workspace-relative paths and direct command names. "
            f"Write it for {self.shell} compatibility, not for the other "
            "PowerShell runtime. Do not include a pwsh or powershell executable "
            "prefix."
        )
        return definition

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {
            "command",
            "timeout_seconds",
            "strip_ansi",
        }
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"

        command = kwargs.get("command")
        timeout_seconds = kwargs.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        strip_ansi = kwargs.get("strip_ansi", True)
        if not isinstance(command, str) or not command.strip():
            return "Error: command must be a non-empty string"
        if (
            not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            return f"Error: timeout_seconds must be an integer between 1 and {MAX_TIMEOUT_SECONDS}"
        if not isinstance(strip_ansi, bool):
            return "Error: strip_ansi must be a boolean"

        try:
            process = subprocess.Popen(
                [
                    self.shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_process_group_options(),
            )
        except OSError as exc:
            return f"Error: failed to run PowerShell: {exc}"

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_and_collect(process)
            return _format_command_result(
                heading=f"Command timed out after {timeout_seconds} seconds.",
                shell=self.shell,
                stdout=stdout,
                stderr=stderr,
                strip_ansi=strip_ansi,
            )
        except KeyboardInterrupt:
            stdout, stderr = _terminate_and_collect(process)
            return _format_command_result(
                heading="Cancelled: command interrupted by Ctrl+C.",
                shell=self.shell,
                stdout=stdout,
                stderr=stderr,
                strip_ansi=strip_ansi,
            )

        return _format_command_result(
            heading=f"Command exited with code {process.returncode}.",
            shell=self.shell,
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            strip_ansi=strip_ansi,
        )


def _process_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_and_collect(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    _terminate_process_tree(process)
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def _format_command_result(
    *,
    heading: str,
    shell: str,
    stdout: str,
    stderr: str,
    exit_code: int | None = None,
    strip_ansi: bool = True,
) -> str:
    stdout_stripped = False
    stderr_stripped = False
    if strip_ansi:
        stdout, stdout_stripped = _strip_ansi_and_controls(stdout)
        stderr, stderr_stripped = _strip_ansi_and_controls(stderr)

    stdout = stdout.rstrip("\r\n")
    stderr = stderr.rstrip("\r\n")

    lines = [heading, f"Shell: {shell}"]
    if exit_code is not None:
        lines.append("Timed out: false")
    if stdout_stripped or stderr_stripped:
        lines.append(
            "ANSI/control sequences stripped: "
            f"stdout={str(stdout_stripped).lower()}, "
            f"stderr={str(stderr_stripped).lower()}."
        )
    lines.extend(["", "STDOUT:", stdout, "", "STDERR:", stderr])
    return "\n".join(lines)


def _strip_ansi_and_controls(value: str) -> tuple[str, bool]:
    stripped = ANSI_SEQUENCE_RE.sub("", value)
    stripped = CONTROL_CHARS_RE.sub("", stripped)
    return stripped, stripped != value


ANSI_SEQUENCE_RE = re.compile(
    r"""
    \x1B
    (?:
        \[[0-?]*[ -/]*[@-~]      # CSI sequences, including SGR colors.
        | \][^\x07]*(?:\x07|\x1B\\) # OSC sequences.
        | [@-Z\\-_]              # Two-byte escape sequences.
    )
    """,
    re.VERBOSE,
)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
