"""Subagent tool with runtime orchestration supplied by a runner."""

from __future__ import annotations

from itertools import count
from typing import Any, ClassVar, Protocol

from harnessed_coder.permissions import PermissionApprover

from .base import Tool, ToolMetadata


class SubAgentRunner(Protocol):
    """Run one isolated subagent without exposing LLM configuration to the tool."""

    def run(
        self,
        prompt: str,
        *,
        needed_tools: list[str],
        permission_approver: PermissionApprover | None,
        subagent_id: str,
        task_summary: str,
    ) -> str: ...


class SubAgentTool(Tool):
    """Run an isolated agent turn for a focused subtask."""

    name: ClassVar[str] = "subagent"
    description: ClassVar[str] = (
        "Delegate a focused workspace task to an isolated subagent. "
        "Use this for investigation, search, or a small implementation step when "
        "a separate short-lived context is useful. The subagent does not inherit "
        "the parent agent's system prompt, conversation, memory, loaded Skills, "
        "or AGENTS.md instructions. Therefore, provide all necessary task information "
        "in context, including relevant memory, AGENTS.md constraints, background, "
        "user requirements and constraints, known findings, relevant file paths, "
        "and the expected output. Starting a subagent is not a permission bypass: "
        "the delegation call itself needs no separate approval, but every tool call "
        "inside the subagent goes through the same workspace permission pipeline as "
        "the parent. Hard-denied calls remain denied; calls classified as ask receive "
        "automatic review, and an abstention uses the host's human approval prompt. "
        "If human approval is unavailable or rejected, that call is denied. Do not "
        "delegate work to avoid permission review. Optionally pass needed_tools to "
        "expose registered deferred tools inside the subagent."
    )
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
        skip_permission_review=True,
        result_size_hint="large",
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific task for the subagent to complete.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Information the isolated subagent needs to complete the task. "
                    "Include relevant memory, AGENTS.md constraints, background, user "
                    "requirements and constraints, known findings, relevant file paths, "
                    "and expected output because it does not inherit the parent system "
                    "prompt, conversation, memory, loaded Skills, or AGENTS.md instructions."
                ),
                "default": "",
            },
            "needed_tools": {
                "type": "array",
                "description": (
                    "Optional model-visible tool names to expose inside the subagent, "
                    "such as session_search. Use only when the subtask is likely to "
                    "need a specific registered deferred tool."
                ),
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    def __init__(self, runner: SubAgentRunner) -> None:
        self._runner = runner
        self._subagent_counter = count(1)
        self._permission_approver: PermissionApprover | None = None

    def set_permission_approver(
        self,
        permission_approver: PermissionApprover | None,
    ) -> None:
        """Use the host's interactive approver for future delegated agents."""
        self._permission_approver = permission_approver

    def execute(self, **kwargs: Any) -> str:
        unexpected = set(kwargs) - {"task", "context", "needed_tools"}
        if unexpected:
            return f"Error: unexpected arguments: {', '.join(sorted(unexpected))}"

        task = kwargs.get("task")
        context = kwargs.get("context", "")
        needed_tools = kwargs.get("needed_tools", [])
        if not isinstance(task, str) or not task.strip():
            return "Error: task must be a non-empty string"
        if not isinstance(context, str):
            return "Error: context must be a string"
        if not isinstance(needed_tools, list):
            return "Error: needed_tools must be a list of tool names"
        needed_tool_names: list[str] = []
        for tool_name in needed_tools:
            if not isinstance(tool_name, str) or not tool_name.strip():
                return "Error: needed_tools must contain non-empty strings"
            needed_tool_names.append(tool_name.strip())

        prompt = task.strip()
        if context.strip():
            prompt = f"Context:\n{context.strip()}\n\nTask:\n{prompt}"
        return self._runner.run(
            prompt,
            needed_tools=_unique_tool_names(needed_tool_names),
            permission_approver=self._permission_approver,
            subagent_id=f"subagent-{next(self._subagent_counter)}",
            task_summary=_task_summary(task),
        )


def _task_summary(task: str, *, limit: int = 160) -> str:
    summary = " ".join(task.strip().split())
    if len(summary) <= limit:
        return summary
    omitted = len(summary) - limit
    return f"{summary[:limit]}... [truncated {omitted} chars]"


def _unique_tool_names(tool_names: list[str]) -> list[str]:
    return list(dict.fromkeys(tool_names))
