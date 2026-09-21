"""Tool registration, visibility, and dispatch."""

from __future__ import annotations

from contextlib import suppress
from threading import RLock
from typing import Any, Callable, Iterable

from harnessed_coder.permissions import (
    DefaultPermissionPolicy,
    PermissionAction,
    PermissionApprover,
    PermissionDecision,
    PermissionPolicy,
    permission_block_message,
)

from .base import Tool, ToolExecutionResult


class ToolRegistry:
    """Collection of registered tools and the subset currently visible to a model."""

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        permission_policy: PermissionPolicy | None = None,
        session_used_tool_names: Iterable[str] = (),
        on_session_tool_used: Callable[[str], None] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._persistent_visible_tool_names: set[str] = set()
        self._turn_visible_tool_names: set[str] = set()
        self._cacheable_tool_names: set[str] = set()
        self._session_used_tool_names: set[str] = set()
        for name in session_used_tool_names:
            if not isinstance(name, str) or not name:
                raise ValueError("session used tool names must be non-empty strings")
            self._session_used_tool_names.add(name)
        self._on_session_tool_used = on_session_tool_used
        self._lock = RLock()
        self._permission_policy = permission_policy or DefaultPermissionPolicy()
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool, *, visible: bool = True) -> None:
        """Add one tool, rejecting accidental name collisions."""
        with self._lock:
            if not tool.name or not tool.description:
                raise ValueError("Tool name and description must not be empty")
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool
            if visible:
                self._persistent_visible_tool_names.add(tool.name)
            else:
                self._cacheable_tool_names.add(tool.name)

    def get(self, name: str) -> Tool:
        """Look up a registered tool by its model-visible name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        """Return currently visible tool schemas suitable for ``llm.chat(tools=...)``."""
        with self._lock:
            visible_names = self._visible_names()
            return [
                tool.definition()
                for name, tool in self._tools.items()
                if name in visible_names
            ]

    def visible_names(self) -> set[str]:
        """Return names currently exposed to the model."""
        with self._lock:
            return self._visible_names()

    def deferred_tools(self) -> list[Tool]:
        """Return registered tools that are not currently exposed to the model."""
        with self._lock:
            visible_names = self._visible_names()
            return [
                tool
                for name, tool in self._tools.items()
                if name not in visible_names
            ]

    def expose(self, names: Iterable[str]) -> list[str]:
        """Persistently expose tools and return names that became newly visible."""
        with self._lock:
            validated_names = self._validated_names(names)
            visible_names = self._visible_names()
            newly_visible: list[str] = []
            for name in validated_names:
                if name not in visible_names:
                    newly_visible.append(name)
                self._persistent_visible_tool_names.add(name)
                self._turn_visible_tool_names.discard(name)
            return newly_visible

    def expose_for_turn(self, names: Iterable[str]) -> list[str]:
        """Expose tools until the next user turn begins."""
        with self._lock:
            validated_names = self._validated_names(names)
            visible_names = self._visible_names()
            newly_visible: list[str] = []
            for name in validated_names:
                if name not in visible_names:
                    newly_visible.append(name)
                    visible_names.add(name)
                if (
                    name not in self._persistent_visible_tool_names
                    and name not in self._session_used_tool_names
                ):
                    self._turn_visible_tool_names.add(name)
            return newly_visible

    def clear_turn_exposures(self) -> None:
        """Hide tools exposed by searches in the previous user turn."""
        with self._lock:
            self._turn_visible_tool_names.clear()

    def record_tool_call(self, name: str) -> None:
        """Keep one actually dispatched deferred tool visible for the session."""
        with self._lock:
            self._validated_names([name])
            if (
                name not in self._cacheable_tool_names
                or name in self._persistent_visible_tool_names
            ):
                return
            if name in self._session_used_tool_names:
                return
            self._session_used_tool_names.add(name)
            self._turn_visible_tool_names.discard(name)
            if self._on_session_tool_used is not None:
                self._on_session_tool_used(name)

    def session_used_names(self) -> list[str]:
        """Return used deferred tools in stable registry order."""
        with self._lock:
            return [
                name
                for name in self._tools
                if name in self._session_used_tool_names
                and name in self._cacheable_tool_names
            ]

    def _visible_names(self) -> set[str]:
        return (
            self._persistent_visible_tool_names
            | self._turn_visible_tool_names
            | (self._session_used_tool_names & self._cacheable_tool_names)
        )

    def _validated_names(self, names: Iterable[str]) -> list[str]:
        validated_names = list(names)
        for name in validated_names:
            if name not in self._tools:
                raise KeyError(f"Unknown tool: {name}")
        return validated_names

    def set_permission_approver(
        self,
        permission_approver: PermissionApprover | None,
    ) -> None:
        """Propagate the host approver to tools that run nested agents."""
        for tool in self._tools.values():
            setter = getattr(tool, "set_permission_approver", None)
            if callable(setter):
                setter(permission_approver)

    def metadata(self) -> dict[str, dict[str, Any]]:
        """Return runtime metadata for all registered tools keyed by tool name."""
        return {
            name: tool.metadata.to_dict()
            for name, tool in self._tools.items()
        }

    def metadata_by_name(self, name: str) -> dict[str, Any]:
        """Return runtime metadata for one registered tool."""
        return self.get(name).metadata.to_dict()

    def can_execute_in_parallel(self, name: str) -> bool:
        """Return whether a registered tool is read-only and parallel-safe."""
        with suppress(KeyError):
            metadata = self.get(name).metadata
            return metadata.is_read_only and metadata.is_parallel_safe
        return False

    def evaluate_permission(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> PermissionDecision:
        """Return the permission decision for one registered tool call."""
        return self._permission_policy.evaluate(self.get(name), arguments)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a parsed LLM tool call to its registered implementation."""
        decision = self.evaluate_permission(name, arguments)
        result = self._apply_permission_decision(name, arguments, decision)
        return result.model_content if isinstance(result, ToolExecutionResult) else result

    def _apply_permission_decision(
        self,
        name: str,
        arguments: dict[str, Any],
        decision: PermissionDecision,
    ) -> str | ToolExecutionResult:
        """Apply the decision produced for an in-flight internal tool call."""
        if decision.action != PermissionAction.ALLOW:
            return permission_block_message(decision)
        self.record_tool_call(name)
        return self.get(name).execute(**arguments)
