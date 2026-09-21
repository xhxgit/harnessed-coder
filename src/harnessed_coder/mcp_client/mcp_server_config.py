"""Load and validate remote MCP server configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harnessed_coder.user_config import load_user_config

from .types import McpServerConfig


_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_COMMON_KEYS = {"transport", "include", "exclude"}
_STDIO_KEYS = _COMMON_KEYS | {"command", "args", "env"}
_HTTP_KEYS = _COMMON_KEYS | {"url", "headers"}


def load_mcp_server_configs(
    data_dir: str | Path | None = None,
) -> tuple[McpServerConfig, ...]:
    """Return validated MCP server configuration from ``config.json``."""
    raw_servers = load_user_config(data_dir).get("mcp_servers", {})
    if not isinstance(raw_servers, dict):
        raise ValueError("mcp_servers must be an object")

    configs: list[McpServerConfig] = []
    for name, raw_config in raw_servers.items():
        if not isinstance(name, str) or not _SERVER_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "MCP server names must contain only letters, digits, '_' or '-'"
            )
        if not isinstance(raw_config, dict):
            raise ValueError(f"mcp_servers.{name} must be an object")
        configs.append(_parse_server(name, raw_config))
    return tuple(configs)


def _parse_server(name: str, raw: dict[str, Any]) -> McpServerConfig:
    transport = raw.get("transport")
    if transport == "stdio":
        _reject_unknown_keys(name, raw, _STDIO_KEYS)
        command = _required_string(raw.get("command"), f"mcp_servers.{name}.command")
        args = _string_list(raw.get("args", []), f"mcp_servers.{name}.args")
        env = _string_mapping(raw.get("env", {}), f"mcp_servers.{name}.env")
        include, exclude = _tool_filters(name, raw)
        return McpServerConfig(
            name=name,
            transport="stdio",
            command=command,
            args=tuple(args),
            env=_expand_mapping(env, f"mcp_servers.{name}.env"),
            include=include,
            exclude=exclude,
        )
    if transport == "streamable_http":
        _reject_unknown_keys(name, raw, _HTTP_KEYS)
        url = _required_string(raw.get("url"), f"mcp_servers.{name}.url")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"mcp_servers.{name}.url must use http or https")
        headers = _string_mapping(
            raw.get("headers", {}),
            f"mcp_servers.{name}.headers",
        )
        include, exclude = _tool_filters(name, raw)
        return McpServerConfig(
            name=name,
            transport="streamable_http",
            url=url,
            headers=_expand_mapping(
                headers,
                f"mcp_servers.{name}.headers",
            ),
            include=include,
            exclude=exclude,
        )
    raise ValueError(
        f"mcp_servers.{name}.transport must be 'stdio' or 'streamable_http'"
    )


def _tool_filters(
    name: str,
    raw: dict[str, Any],
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    if "include" in raw and "exclude" in raw:
        raise ValueError(
            f"mcp_servers.{name}.include and exclude are mutually exclusive"
        )
    include = (
        frozenset(
            _string_list(raw["include"], f"mcp_servers.{name}.include")
        )
        if "include" in raw
        else None
    )
    exclude = (
        frozenset(
            _string_list(raw["exclude"], f"mcp_servers.{name}.exclude")
        )
        if "exclude" in raw
        else None
    )
    return include, exclude


def _reject_unknown_keys(
    name: str,
    raw: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"unexpected keys in mcp_servers.{name}: {', '.join(unknown)}"
        )


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array of non-empty strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _required_string(item, path)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _string_mapping(value: object, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object of string values")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path} keys must be non-empty strings")
        if not isinstance(item, str):
            raise ValueError(f"{path}.{key} must be a string")
        result[key] = item
    return result


def _expand_mapping(values: Mapping[str, str], path: str) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, value in values.items():
        match = _ENV_REFERENCE_PATTERN.fullmatch(value)
        if match is None:
            expanded[key] = value
            continue
        variable = match.group(1)
        resolved = os.environ.get(variable)
        if resolved is None:
            raise ValueError(
                f"{path}.{key} references missing environment variable {variable}"
            )
        expanded[key] = resolved
    return expanded
