"""Application logging configuration."""

from __future__ import annotations

from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from ..constants import DEFAULT_LOG_FILE


DEFAULT_LOG_LEVEL = "INFO"

# These dependencies emit one INFO record for nearly every HTTP/model request.
# Keep normal logs operational; detailed transport logs remain available at DEBUG.
_NOISY_DEPENDENCY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3")


def configure_logging(
    *,
    log_dir: str | Path,
    level: str = DEFAULT_LOG_LEVEL,
    run_id: str | None = None,
) -> Path:
    """Configure application logging and return the active log file path."""
    resolved_log_dir = Path(log_dir).resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_log_dir / _run_file_name(
        DEFAULT_LOG_FILE,
        run_id or make_run_id(),
    )

    numeric_level = _parse_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    dependency_level = (
        logging.DEBUG if numeric_level <= logging.DEBUG else logging.WARNING
    )
    for logger_name in _NOISY_DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(dependency_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(thread)d] %(levelname)s - %(name)s - %(message)s",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info(
        "Logging initialized: file=%s level=%s",
        log_file,
        logging.getLevelName(numeric_level),
    )
    return log_file


def make_run_id() -> str:
    """Return a filesystem-friendly identifier for one CLI startup."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _parse_log_level(level: str) -> int:
    normalized = level.strip().upper()
    numeric_level = logging.getLevelName(normalized)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    return numeric_level


def _run_file_name(base_name: str, run_id: str) -> str:
    path = Path(base_name)
    return f"{path.stem}-{run_id}{path.suffix}"
