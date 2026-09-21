"""User configuration helpers for harnessed_coder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .constants import DEFAULT_CONTEXT_TRIGGER_RATIO, USER_CONFIG_FILE_NAME
from .session import resolve_data_dir

USER_CONFIG_VERSION = 1
DEFAULT_OPENAI_API_KEY = "sk-"
DEFAULT_OPENAI_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_CONTEXT_MAX_TOKENS = 300_000
DEFAULT_REASONING_EFFORT = "low"
_ACTIVE_DATA_DIR: Path | None = None


@dataclass(frozen=True)
class ReasoningConfig:
    reasoning_effort: str | None
    thinking_budget: int | None


def set_user_config_data_dir(data_dir: str | Path | None) -> None:
    """Select the user data directory used by process-wide LLM config reads."""
    global _ACTIVE_DATA_DIR
    _ACTIVE_DATA_DIR = resolve_data_dir(data_dir)


def resolve_user_config_path(data_dir: str | Path | None = None) -> Path:
    """Return the user-level JSON config path."""
    effective_data_dir = data_dir if data_dir is not None else _ACTIVE_DATA_DIR
    return (resolve_data_dir(effective_data_dir) / USER_CONFIG_FILE_NAME).resolve()


def load_user_config(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load user-level CLI configuration from the data directory."""
    config_path = resolve_user_config_path(data_dir)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"user config must contain a JSON object: {config_path}")
    version = data.get("version", USER_CONFIG_VERSION)
    if version != USER_CONFIG_VERSION:
        raise ValueError(f"unsupported user config version: {version}")
    return data


def save_user_config(
    config: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Atomically write user-level CLI configuration."""
    config_path = resolve_user_config_path(data_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {**config, "version": USER_CONFIG_VERSION}
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=config_path.parent,
        prefix=f".{config_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_file.write("\n")
    temp_path.replace(config_path)
    return config_path


def create_initial_user_config(
    data_dir: str | Path | None = None,
) -> Path | None:
    """Create the editable first-run LLM config without overwriting user data."""
    config_path = resolve_user_config_path(data_dir)
    if config_path.exists():
        return None

    trigger_tokens = int(DEFAULT_CONTEXT_MAX_TOKENS * DEFAULT_CONTEXT_TRIGGER_RATIO)
    trigger_percentage = DEFAULT_CONTEXT_TRIGGER_RATIO * 100
    return save_user_config(
        {
            "_comment": (
                "Automatic context compression starts at "
                f"{trigger_tokens} tokens "
                f"({trigger_percentage:g}% of context_max_tokens). "
                "Replace openai_api_key before the first model request."
            ),
            "openai_api_key": DEFAULT_OPENAI_API_KEY,
            "openai_base_url": DEFAULT_OPENAI_BASE_URL,
            "model": DEFAULT_MODEL,
            "context_max_tokens": DEFAULT_CONTEXT_MAX_TOKENS,
            "responses_full_history": True,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        },
        data_dir=data_dir,
    )


def get_openai_api_key() -> str:
    """Return the OpenAI API key from the user config file."""
    api_key = load_user_config().get("openai_api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("openai_api_key is not set in user config")
    return api_key.strip()


def get_openai_base_url() -> str | None:
    """Return the OpenAI base URL from the user config file, if any."""
    base_url = load_user_config().get("openai_base_url")
    if base_url is None:
        return None
    if not isinstance(base_url, str):
        raise ValueError("openai_base_url must be a string")
    normalized_base_url = base_url.strip()
    return normalized_base_url or None


def get_configured_model(data_dir: str | Path | None = None) -> str:
    """Return the required chat model from user config."""
    model = load_user_config(data_dir).get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is not set in user config")
    return model.strip()


def get_context_max_tokens(data_dir: str | Path | None = None) -> int:
    """Return the required context compression trigger budget from user config."""
    max_tokens = load_user_config(data_dir).get("context_max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("context_max_tokens must be a positive integer in user config")
    return max_tokens


def get_responses_full_history(data_dir: str | Path | None = None) -> bool:
    """Return whether Responses requests must resend the complete history."""
    value = load_user_config(data_dir).get("responses_full_history", False)
    if not isinstance(value, bool):
        raise ValueError("responses_full_history must be a boolean in user config")
    return value


def get_reasoning_config(
    data_dir: str | Path | None = None,
) -> ReasoningConfig:
    """Return optional provider reasoning settings from user config."""
    config = load_user_config(data_dir)
    reasoning_effort = config.get("reasoning_effort")
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or not reasoning_effort.strip():
            raise ValueError("reasoning_effort must be a non-empty string in user config")
        reasoning_effort = reasoning_effort.strip()

    thinking_budget = config.get("thinking_budget")
    if thinking_budget is not None and (
        isinstance(thinking_budget, bool)
        or not isinstance(thinking_budget, int)
        or thinking_budget < 1
    ):
        raise ValueError("thinking_budget must be a positive integer in user config")

    return ReasoningConfig(
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )


def set_configured_model(model: str, *, data_dir: str | Path | None = None) -> Path:
    """Persist the selected chat model in user-level config."""
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must be a non-empty string")
    config = load_user_config(data_dir)
    config["model"] = normalized_model
    return save_user_config(config, data_dir=data_dir)
