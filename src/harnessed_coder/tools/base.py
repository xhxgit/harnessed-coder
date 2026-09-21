"""Base contract for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from ..constants.tool_protocol import (
    HISTORICAL_COMPRESSION_ARGUMENT,
    HISTORICAL_COMPRESSION_DESCRIPTION,
)

@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Separate model-visible tool output from canonical host state."""

    model_content: str
    internal_data: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Runtime-only properties used by the agent, not exposed to the model."""

    is_read_only: bool
    is_parallel_safe: bool
    skip_permission_review: bool = False
    result_size_hint: str = "small"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logging/tests."""
        return asdict(self)


class Tool(ABC):
    """An executable function that can be exposed to the language model."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        is_read_only=False,
        is_parallel_safe=False,
    )

    def definition(self) -> dict[str, Any]:
        """Return this tool in the function-calling schema expected by the LLM."""
        parameters = deepcopy(self.parameters)
        if parameters.get("type") == "object":
            properties = parameters.setdefault("properties", {})
            if isinstance(properties, dict):
                properties[HISTORICAL_COMPRESSION_ARGUMENT] = {
                    "type": "boolean",
                    "description": HISTORICAL_COMPRESSION_DESCRIPTION,
                }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> str | ToolExecutionResult:
        """Run the tool with model-provided keyword arguments."""
        raise NotImplementedError
