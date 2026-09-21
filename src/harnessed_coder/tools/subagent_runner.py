"""Configured runtime for isolated subagent execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from harnessed_coder.llm import LLMResponse, chat as llm_chat
from harnessed_coder.mcp_client import McpArtifactStore, McpConnections
from harnessed_coder.permissions import PermissionApprover
from harnessed_coder.response_style import PLAIN_TEXT_RESPONSE_RULES
from harnessed_coder.agent.trace_recorder import (
    AgentTraceRecorder,
    current_trace_turn_id,
)


class ConfiguredSubAgentRunner:
    """Own LLM/runtime configuration needed to construct isolated agents."""

    def __init__(
        self,
        root: str | Path | None,
        *,
        model: str,
        base_url: str | None = None,
        chat_function: Callable[..., LLMResponse] | None = None,
        max_tool_rounds: int = 10,
        data_dir: str | Path | None = None,
        mcp_connections: McpConnections | None = None,
        mcp_artifact_store: McpArtifactStore | None = None,
        run_id: str | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self._root = root
        self._model = model
        self._base_url = base_url
        self._chat_function = chat_function or llm_chat
        self._max_tool_rounds = max_tool_rounds
        self._data_dir = data_dir
        self._mcp_connections = mcp_connections
        self._mcp_artifact_store = mcp_artifact_store
        self._run_id = run_id
        self._trace_path = trace_path

    def run(
        self,
        prompt: str,
        *,
        needed_tools: list[str],
        permission_approver: PermissionApprover | None,
        subagent_id: str,
        task_summary: str,
    ) -> str:
        # Imported lazily to avoid a package-load cycle: Agent imports tools,
        # while this configured runner creates an Agent only when invoked.
        from harnessed_coder.agent import Agent
        from harnessed_coder.permissions import LLMPermissionReviewer
        from harnessed_coder.tools import create_default_registry

        parent_turn_id = current_trace_turn_id()
        parent_trace = AgentTraceRecorder(
            run_id=self._run_id,
            trace_path=self._trace_path,
            metadata={"scope": "main", "turn_id": parent_turn_id},
        )
        started_at = perf_counter()
        subagent_tools = create_default_registry(
            self._root,
            include_subagent=False,
            data_dir=self._data_dir,
            mcp_connections=self._mcp_connections,
            mcp_artifact_store=self._mcp_artifact_store,
            base_url=self._base_url,
            chat_function=self._chat_function,
            model=self._model,
            run_id=self._run_id,
            trace_path=self._trace_path,
        )
        try:
            subagent_tools.expose(needed_tools)
        except KeyError as exc:
            parent_trace.subagent(
                status="failed",
                subagent_id=subagent_id,
                duration_ms=round((perf_counter() - started_at) * 1000),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            return f"Error: {exc.args[0]}"

        parent_trace.subagent(
            status="started",
            subagent_id=subagent_id,
            task=task_summary,
            needed_tools=needed_tools,
        )

        subagent = Agent(
            subagent_tools,
            model=self._model,
            base_url=self._base_url,
            chat_function=self._chat_function,
            permission_reviewer=LLMPermissionReviewer(
                model=self._model,
                base_url=self._base_url,
                chat_function=self._chat_function,
            ),
            permission_approver=permission_approver,
            max_tool_rounds=self._max_tool_rounds,
            trace_metadata={
                "scope": "subagent",
                "subagent_id": subagent_id,
                "subagent_task": task_summary,
                "turn_id": parent_turn_id,
            },
            run_id=self._run_id,
            trace_path=None if self._trace_path is None else str(Path(self._trace_path).resolve()),
            system_prompt_provider=lambda: PLAIN_TEXT_RESPONSE_RULES,
        )
        try:
            result = subagent.chat(prompt).content
        except Exception:
            parent_trace.subagent(
                status="failed",
                subagent_id=subagent_id,
                duration_ms=round((perf_counter() - started_at) * 1000),
            )
            raise
        parent_trace.subagent(
            status="completed",
            subagent_id=subagent_id,
            duration_ms=round((perf_counter() - started_at) * 1000),
        )
        return result
