"""Process-wide structured trace channel and JSONL handler."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

TRACE_LOGGER_NAME = __name__
TRACE_EVENT_ATTRIBUTE = "harnessed_coder_trace_event"
TRACE_BODY_ATTRIBUTE = "harnessed_coder_trace_body"
TRACE_METADATA_ATTRIBUTE = "harnessed_coder_trace_metadata"
TRACE_PATH_ATTRIBUTE = "harnessed_coder_trace_path"

_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
_trace_logger.propagate = False
_trace_logger.addHandler(logging.NullHandler())


class JsonlTraceHandler(logging.Handler):
    """Route structured events to their caller-selected session JSONL file."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        """Append one structured trace record as a JSON line."""
        event = getattr(record, TRACE_EVENT_ATTRIBUTE)
        body = getattr(record, TRACE_BODY_ATTRIBUTE)
        metadata = getattr(record, TRACE_METADATA_ATTRIBUTE)
        trace_path_value = getattr(record, TRACE_PATH_ATTRIBUTE, None)
        if trace_path_value is None:
            return
        trace_path = Path(trace_path_value).resolve()
        trace_record = {
            "timestamp": _local_timestamp(),
            "event": event,
        }
        if metadata:
            for key, value in metadata.items():
                if key in trace_record or key == "data":
                    raise ValueError(
                        f"trace metadata key conflicts with record field: {key}"
                    )
                trace_record[key] = value
        trace_record["data"] = body
        line = json.dumps(trace_record, ensure_ascii=False, default=str) + "\n"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(line)


def configure_jsonl_trace() -> None:
    """Configure the process channel for lazily-created session trace files."""
    trace_handler = JsonlTraceHandler()
    for handler in _trace_logger.handlers[:]:
        _trace_logger.removeHandler(handler)
        handler.close()
    _trace_logger.setLevel(logging.INFO)
    _trace_logger.addHandler(trace_handler)
    logging.getLogger("harnessed_coder.diagnostics").info(
        "Session JSONL tracing enabled"
    )


def write_trace_event(
    event: str,
    body: dict[str, Any],
    *,
    trace_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Publish one structured event to the configured process trace channel."""
    _trace_logger.info(
        event,
        extra={
            TRACE_EVENT_ATTRIBUTE: event,
            TRACE_BODY_ATTRIBUTE: body,
            TRACE_METADATA_ATTRIBUTE: metadata,
            TRACE_PATH_ATTRIBUTE: trace_path,
        },
    )


def _local_timestamp() -> str:
    """Return an ISO timestamp in the machine's current local timezone."""
    return datetime.now().astimezone().isoformat()
