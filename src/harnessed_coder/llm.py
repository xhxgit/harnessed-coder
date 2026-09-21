"""OpenAI chat helpers."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from openai import OpenAI

from .diagnostics.api_exchange import (
    api_request_completed,
    api_request_failed,
    api_request_started,
)
from .session.usage import report_usage
from .user_config import (
    get_openai_api_key,
    get_openai_base_url,
    get_reasoning_config,
    get_responses_full_history,
)


REQUEST_TOO_LARGE_MESSAGE = (
    "The model request was rejected because the payload or context window was too large."
)
class LLMRequestTooLargeError(RuntimeError):
    """Raised when an LLM request is rejected for payload/context size."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_available: bool = True
    first_event_kind: str | None = None
    time_to_first_event_ms: int | None = None
    request_duration_ms: int | None = None
    cached_prompt_tokens: int | None = None
    cache_creation_prompt_tokens: int | None = None
    protocol: str = "chat_completions"
    provider_response_id: str | None = None
    previous_response_id: str | None = None
    response_state_reuse: str | None = None
    response_matched_messages: int | None = None
    response_input_items: int | None = None

    def to_history_message(self) -> dict[str, Any]:
        """Convert the provider response to a canonical assistant message."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass(frozen=True)
class _ResponseNode:
    response_id: str
    message_count: int
    fingerprint: str


class ResponseContextHandle:
    """Opaque, process-local Responses API context owned by the LLM layer."""

    __slots__ = ("_id", "_lock", "_nodes")

    def __init__(self) -> None:
        self._id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._nodes: list[_ResponseNode] = []

    @property
    def id(self) -> str:
        """Return a trace-safe local identifier, not a provider response ID."""
        return self._id


def open_response_context() -> ResponseContextHandle:
    """Open an empty, non-persistent Responses API context."""
    return ResponseContextHandle()


def reset_response_context(handle: ResponseContextHandle) -> None:
    """Forget every provider response node associated with a local context."""
    with handle._lock:
        handle._nodes.clear()


@lru_cache(maxsize=8)
def _build_client(base_url: str | None = None) -> OpenAI:
    api_key = get_openai_api_key()

    if base_url is None:
        base_url = get_openai_base_url()

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def _resolve_base_url(base_url: str | None = None) -> str | None:
    if base_url is not None:
        return base_url

    return get_openai_base_url()


def is_request_too_large_error(exc: BaseException) -> bool:
    """Return True for common OpenAI-compatible payload/context-size errors."""
    if isinstance(exc, LLMRequestTooLargeError):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 413:
        return True

    payload = _extract_error_payload(exc)
    haystack = " ".join(
        str(value)
        for value in [
            type(exc).__name__,
            str(exc),
            payload.get("code"),
            payload.get("type"),
            payload.get("message"),
        ]
        if value
    ).lower()
    return any(
        marker in haystack
        for marker in [
            "payload too large",
            "request too large",
            "context length exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
            "413",
        ]
    )


def _extract_error_payload(exc: BaseException) -> dict[str, Any]:
    for attr in ("body", "response", "error"):
        value = getattr(exc, attr, None)
        if isinstance(value, dict):
            nested = value.get("error")
            if isinstance(nested, dict):
                return nested
            return value
    return {}


def chat(
    messages: list[dict],
    *,
    model: str,
    base_url: str | None = None,
    tools: list[dict] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_activity_delta: Callable[[str, int], None] | None = None,
    **kwargs,
) -> LLMResponse:
    """Stream a chat completion and return a structured response."""
    resolved_base_url = _resolve_base_url(base_url)
    client = _build_client(base_url=resolved_base_url)

    params: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        **kwargs,
    }
    _apply_reasoning_config(params, model=model, protocol="chat_completions")
    if tools:
        params["tools"] = tools

    api_call = api_request_started(
        protocol="chat_completions",
        base_url=resolved_base_url,
        params=params,
    )
    request_started_at = perf_counter()
    try:
        stream = client.chat.completions.create(**params)
    except Exception as exc:
        api_request_failed(api_call, error=exc)
        if is_request_too_large_error(exc):
            raise LLMRequestTooLargeError(REQUEST_TOO_LARGE_MESSAGE) from exc
        raise

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_map: dict[int, dict] = {}
    prompt_tok = 0
    completion_tok = 0
    usage_available = False
    cached_prompt_tokens: int | None = None
    cache_creation_prompt_tokens: int | None = None
    first_event_kind: str | None = None
    first_event_at: float | None = None

    # Iterating this stream calls next() on a network-backed iterator, so it may
    # block while waiting for the model or the network to deliver the next chunk.
    for chunk in stream:
        if chunk.usage:
            usage_available = True
            prompt_tok = chunk.usage.prompt_tokens
            completion_tok = chunk.usage.completion_tokens
            reported_cached_tokens = _usage_token_detail(
                chunk.usage,
                "cached_tokens",
            )
            if reported_cached_tokens is not None:
                cached_prompt_tokens = reported_cached_tokens
            reported_cache_creation_tokens = _usage_token_detail(
                chunk.usage,
                "cache_creation_input_tokens",
            )
            if reported_cache_creation_tokens is not None:
                cache_creation_prompt_tokens = reported_cache_creation_tokens
            report_usage(prompt_tok, completion_tok, cached_prompt_tokens or 0)

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning_content = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            if first_event_at is None:
                first_event_kind = "reasoning"
                first_event_at = perf_counter()
            reasoning_parts.append(reasoning_content)
            if on_activity_delta:
                on_activity_delta("reasoning", len(reasoning_content))

        if delta.content:
            if first_event_at is None:
                first_event_kind = "content"
                first_event_at = perf_counter()
            content_parts.append(delta.content)
            if on_text_delta:
                # The callback is invoked synchronously while processing the
                # stream; slow callback work delays application-level handling
                # of later chunks, but does not normally stop lower-level
                # network receiving.
                on_text_delta(delta.content)

        if delta.tool_calls:
            if first_event_at is None:
                first_event_kind = "tool_call"
                first_event_at = perf_counter()
            if on_activity_delta:
                on_activity_delta("tool_call", 0)
            # OpenAI-compatible streaming sends function arguments in fragments.
            # Accumulate by delta.index, then parse JSON once the stream ends.
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tc_map:
                    tc_map[idx] = {"id": "", "name": "", "args": ""}
                if tc_delta.id:
                    tc_map[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc_map[idx]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    tc_map[idx]["args"] += tc_delta.function.arguments

    request_finished_at = perf_counter()
    parsed: list[ToolCall] = []
    for idx in sorted(tc_map):
        raw = tc_map[idx]
        try:
            args = json.loads(raw["args"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

    response = LLMResponse(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=parsed,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        usage_available=usage_available,
        first_event_kind=first_event_kind,
        time_to_first_event_ms=(
            round((first_event_at - request_started_at) * 1000)
            if first_event_at is not None
            else None
        ),
        request_duration_ms=round((request_finished_at - request_started_at) * 1000),
        cached_prompt_tokens=cached_prompt_tokens,
        cache_creation_prompt_tokens=cache_creation_prompt_tokens,
    )
    api_request_completed(api_call, response=_api_response_payload(response))
    return response


def responses(
    messages: list[dict],
    *,
    response_context: ResponseContextHandle,
    model: str,
    base_url: str | None = None,
    tools: list[dict] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_activity_delta: Callable[[str, int], None] | None = None,
    **kwargs,
) -> LLMResponse:
    """Stream a stateful or full-history Responses API call."""
    resolved_base_url = _resolve_base_url(base_url)
    client = _build_client(base_url=resolved_base_url)
    instructions, conversation = _split_instructions(messages)
    fingerprints = _prefix_fingerprints(conversation)
    full_history = get_responses_full_history()

    # Serialize selection and insertion for a handle. Agent calls are normally
    # sequential, but the lock makes accidental concurrent use deterministic.
    with response_context._lock:
        matched = (
            None
            if full_history
            else _deepest_response_node(response_context._nodes, fingerprints)
        )
        matched_count = matched.message_count if matched is not None else 0
        input_items = _responses_input(
            conversation if full_history else conversation[matched_count:],
            deepseek_reasoning_replay=_is_official_deepseek_url(resolved_base_url),
        )
        params: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            "store": True,
            **kwargs,
        }
        if instructions:
            params["instructions"] = instructions
        if matched is not None:
            params["previous_response_id"] = matched.response_id
        converted_tools = _responses_tools(tools or [])
        if converted_tools:
            params["tools"] = converted_tools
        _apply_reasoning_config(params, model=model, protocol="responses")

        api_call = api_request_started(
            protocol="responses",
            base_url=resolved_base_url,
            params=params,
        )
        request_started_at = perf_counter()
        try:
            stream = client.responses.create(**params)
        except Exception as exc:
            api_request_failed(api_call, error=exc)
            if is_request_too_large_error(exc):
                raise LLMRequestTooLargeError(REQUEST_TOO_LARGE_MESSAGE) from exc
            raise

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[str, ToolCall] = {}
        provider_response_id: str | None = None
        prompt_tok = 0
        completion_tok = 0
        usage_available = False
        cached_prompt_tokens: int | None = None
        cache_creation_prompt_tokens: int | None = None
        first_event_kind: str | None = None
        first_event_at: float | None = None

        for event in stream:
            event_type = _field_value(event, "type")
            event_response = _field_value(event, "response")
            event_error_code = _field_value(event, "code")
            event_error_message = _field_value(event, "message")
            if not event_type and (event_error_code or event_error_message):
                details = ": ".join(
                    str(value)
                    for value in (event_error_code, event_error_message)
                    if value
                )
                error = RuntimeError(
                    f"Responses API stream ended unsuccessfully: {details}"
                )
                api_request_failed(api_call, error=error)
                raise error
            event_response_id = _field_value(event_response, "id")
            if not isinstance(event_response_id, str):
                event_response_id = _field_value(event, "response_id")
            if isinstance(event_response_id, str) and event_response_id:
                provider_response_id = event_response_id

            if event_type in {
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                delta = _field_value(event, "delta")
                if isinstance(delta, str) and delta:
                    if first_event_at is None:
                        first_event_kind = "reasoning"
                        first_event_at = perf_counter()
                    reasoning_parts.append(delta)
                    if on_activity_delta:
                        on_activity_delta("reasoning", len(delta))
            elif event_type == "response.output_text.delta":
                delta = _field_value(event, "delta")
                if isinstance(delta, str) and delta:
                    if first_event_at is None:
                        first_event_kind = "content"
                        first_event_at = perf_counter()
                    content_parts.append(delta)
                    if on_text_delta:
                        on_text_delta(delta)
            elif event_type == "response.function_call_arguments.delta":
                if first_event_at is None:
                    first_event_kind = "tool_call"
                    first_event_at = perf_counter()
                if on_activity_delta:
                    on_activity_delta("tool_call", 0)
            elif event_type == "response.output_item.done":
                item = _field_value(event, "item")
                if _field_value(item, "type") == "function_call":
                    call = _tool_call_from_response_item(item)
                    tool_calls[call.id] = call
                    if first_event_at is None:
                        first_event_kind = "tool_call"
                        first_event_at = perf_counter()
                    if on_activity_delta:
                        on_activity_delta("tool_call", 0)
            elif event_type in {"response.failed", "response.incomplete"}:
                error = _field_value(event_response, "error") or event_type
                stream_error = RuntimeError(
                    f"Responses API stream ended unsuccessfully: {error}"
                )
                api_request_failed(api_call, error=stream_error)
                raise stream_error

            usage = _field_value(event_response, "usage")
            if usage is not None:
                usage_available = True
                prompt_tok = _nonnegative_int(_field_value(usage, "input_tokens"))
                completion_tok = _nonnegative_int(_field_value(usage, "output_tokens"))
                cached_prompt_tokens = _usage_token_detail(usage, "cached_tokens")
                cache_creation_prompt_tokens = _usage_token_detail(
                    usage, "cache_creation_input_tokens"
                )

        request_finished_at = perf_counter()
        if usage_available:
            report_usage(prompt_tok, completion_tok, cached_prompt_tokens or 0)
        if not provider_response_id:
            error = RuntimeError("Responses API stream did not return a response ID")
            api_request_failed(api_call, error=error)
            raise error

        response = LLMResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=list(tool_calls.values()),
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            usage_available=usage_available,
            first_event_kind=first_event_kind,
            time_to_first_event_ms=(
                round((first_event_at - request_started_at) * 1000)
                if first_event_at is not None
                else None
            ),
            request_duration_ms=round((request_finished_at - request_started_at) * 1000),
            cached_prompt_tokens=cached_prompt_tokens,
            cache_creation_prompt_tokens=cache_creation_prompt_tokens,
            protocol="responses",
            provider_response_id=provider_response_id,
            previous_response_id=(matched.response_id if matched is not None else None),
            response_state_reuse=(
                "full_history"
                if full_history
                else "root"
                if matched is None
                else "latest"
                if matched is response_context._nodes[-1]
                else "branch"
            ),
            response_matched_messages=matched_count,
            response_input_items=len(input_items),
        )
        post_messages = [*conversation, _provider_history_message(response)]
        if not full_history:
            response_context._nodes.append(
                _ResponseNode(
                    response_id=provider_response_id,
                    message_count=len(post_messages),
                    fingerprint=_messages_fingerprint(post_messages),
                )
            )
        api_request_completed(api_call, response=_api_response_payload(response))
        return response


def _api_response_payload(response: LLMResponse) -> dict[str, Any]:
    """Return the full aggregate response written to the opt-in API log."""
    return {
        "content": response.content,
        "reasoning_content": response.reasoning_content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in response.tool_calls
        ],
        "usage": {
            "available": response.usage_available,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cached_prompt_tokens": response.cached_prompt_tokens,
            "cache_creation_prompt_tokens": response.cache_creation_prompt_tokens,
        },
        "performance": {
            "first_event": response.first_event_kind,
            "ttft_ms": response.time_to_first_event_ms,
            "request_duration_ms": response.request_duration_ms,
        },
        "transport": {
            "protocol": response.protocol,
            "response_id": response.provider_response_id,
            "previous_response_id": response.previous_response_id,
            "state_reuse": response.response_state_reuse,
            "matched_messages": response.response_matched_messages,
            "input_items": response.response_input_items,
        },
    }


def _split_instructions(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, str):
            return content, messages[1:]
    return None, messages


def _prefix_fingerprints(messages: list[dict[str, Any]]) -> dict[int, str]:
    digest = hashlib.sha256()
    result: dict[int, str] = {}
    for count, message in enumerate(messages, start=1):
        serialized = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(serialized).to_bytes(8, "big"))
        digest.update(serialized)
        result[count] = digest.hexdigest()
    return result


def _messages_fingerprint(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return hashlib.sha256().hexdigest()
    return _prefix_fingerprints(messages)[len(messages)]


def _deepest_response_node(
    nodes: list[_ResponseNode], fingerprints: dict[int, str]
) -> _ResponseNode | None:
    matches = [
        node
        for node in nodes
        if fingerprints.get(node.message_count) == node.fingerprint
    ]
    return max(matches, key=lambda node: node.message_count, default=None)


def _is_official_deepseek_url(base_url: str | None) -> bool:
    """Return whether Responses history uses DeepSeek's reasoning extension."""
    if base_url is None:
        return False
    return (urlsplit(base_url).hostname or "").lower() == "api.deepseek.com"


def _responses_input(
    messages: list[dict[str, Any]], *, deepseek_reasoning_replay: bool = False
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        if role == "assistant" and isinstance(reasoning_content, str) and reasoning_content:
            if deepseek_reasoning_replay:
                items.append(
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": reasoning_content}
                        ],
                    }
                )
            else:
                items.append(
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": reasoning_content}
                        ],
                    }
                )
        if content is not None:
            items.append({"role": role, "content": content})
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tool_call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "arguments": arguments,
                    }
                )
    return items


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        result.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
                "strict": function.get("strict", False),
            }
        )
    return result


def _tool_call_from_response_item(item: Any) -> ToolCall:
    raw_arguments = _field_value(item, "arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = {}
    call_id = _field_value(item, "call_id") or _field_value(item, "id") or ""
    return ToolCall(
        id=str(call_id),
        name=str(_field_value(item, "name") or ""),
        arguments=arguments,
    )


def _provider_history_message(response: LLMResponse) -> dict[str, Any]:
    message = response.to_history_message()
    message.pop("reasoning_content", None)
    return message


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _usage_token_detail(usage: Any, name: str) -> int | None:
    """Read optional cache counters across compatible provider response shapes."""
    for source in (
        _field_value(usage, "prompt_tokens_details"),
        _field_value(usage, "input_tokens_details"),
        usage,
    ):
        value = _field_value(source, name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _field_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    value = getattr(source, name, None)
    if value is not None:
        return value
    model_extra = getattr(source, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None


def _apply_reasoning_config(
    params: dict[str, Any],
    *,
    model: str,
    protocol: str,
) -> None:
    """Merge user reasoning settings while preserving request-specific overrides."""
    is_qwen3_8 = model.strip().lower().startswith("qwen3.8")
    extra_body = dict(params.get("extra_body") or {})
    explicit_effort = params.pop("reasoning_effort", None)
    extra_body_effort = extra_body.pop("reasoning_effort", None)
    if explicit_effort is None:
        explicit_effort = extra_body_effort
    explicit_reasoning = (
        explicit_effort is not None
        or "reasoning" in params
        or any(
            name in extra_body for name in ("thinking_budget", "enable_thinking")
        )
    )

    if explicit_reasoning:
        if is_qwen3_8 and "thinking_budget" in extra_body:
            explicit_effort = None
        if explicit_effort is not None and "reasoning" not in params:
            if protocol == "responses":
                params["reasoning"] = {"effort": explicit_effort}
            else:
                params["reasoning_effort"] = explicit_effort
        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)
        return

    config = get_reasoning_config()
    if config.thinking_budget is not None:
        extra_body["thinking_budget"] = config.thinking_budget
        params["extra_body"] = extra_body
    if config.reasoning_effort is not None and not (
        is_qwen3_8 and config.thinking_budget is not None
    ):
        if protocol == "responses":
            params["reasoning"] = {"effort": config.reasoning_effort}
        else:
            params["reasoning_effort"] = config.reasoning_effort
