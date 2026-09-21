"""Opt-in, sensitive JSONL logging for model API exchanges."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any


_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "proxy-authorization",
    "x-api-key",
}
_lock = threading.Lock()
_active_path: Path | None = None
_request_ids = count(1)


@dataclass(frozen=True)
class ApiExchangeCall:
    """Correlation data for one logged provider request."""

    request_id: str
    protocol: str
    started_at: float


def configure_api_exchange_logging(
    *,
    log_dir: str | Path,
    run_id: str,
) -> Path:
    """Enable per-run API exchange logging and return its JSONL path."""
    global _active_path, _request_ids
    resolved_dir = Path(log_dir).expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    path = resolved_dir / f"api-exchange-{run_id}.jsonl"
    path.write_text("", encoding="utf-8")
    with _lock:
        _active_path = path
        _request_ids = count(1)
    return path


def disable_api_exchange_logging() -> None:
    """Disable API exchange logging for later calls in this process."""
    global _active_path
    with _lock:
        _active_path = None


def api_request_started(
    *,
    protocol: str,
    base_url: str | None,
    params: dict[str, Any],
) -> ApiExchangeCall | None:
    """Persist the complete SDK call parameters when logging is enabled."""
    with _lock:
        if _active_path is None:
            return None
        request_id = f"api-request-{next(_request_ids)}"
    call = ApiExchangeCall(
        request_id=request_id,
        protocol=protocol,
        started_at=perf_counter(),
    )
    _write(
        {
            "timestamp": _timestamp(),
            "event": "request",
            "request_id": request_id,
            "protocol": protocol,
            "base_url": base_url,
            "params": params,
        }
    )
    return call


def api_request_completed(
    call: ApiExchangeCall | None,
    *,
    response: dict[str, Any],
) -> None:
    """Persist the complete aggregate response for a logged request."""
    if call is None:
        return
    _write(
        {
            "timestamp": _timestamp(),
            "event": "response",
            "request_id": call.request_id,
            "protocol": call.protocol,
            "duration_ms": round((perf_counter() - call.started_at) * 1000),
            "response": response,
        }
    )


def api_request_failed(
    call: ApiExchangeCall | None,
    *,
    error: BaseException,
) -> None:
    """Persist a terminal failure for a logged request."""
    if call is None:
        return
    _write(
        {
            "timestamp": _timestamp(),
            "event": "error",
            "request_id": call.request_id,
            "protocol": call.protocol,
            "duration_ms": round((perf_counter() - call.started_at) * 1000),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "status_code": getattr(error, "status_code", None),
            },
        }
    )


def _write(payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with _lock:
        path = _active_path
        if path is None:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.strip().lower() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(item_key): _json_safe(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    return repr(value)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")
