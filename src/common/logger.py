"""Logging utilities for the application."""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a minimal, safe default format.

    This function is idempotent: repeated calls will not add duplicate
    handlers, but will update the root logger level to the provided value.

    Args:
        level: Logging level name (e.g., "INFO", "DEBUG").
    """
    root_logger = logging.getLogger()
    level_value = _resolve_level(level)

    if root_logger.handlers:
        root_logger.setLevel(level_value)
        return

    logging.basicConfig(
        level=level_value,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger by name, defaulting to this module's logger.

    Args:
        name: Optional logger name. If None, use this module's name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name or __name__)


def _resolve_level(level: str) -> int:
    level_key = level.upper()
    level_map = logging.getLevelNamesMapping()
    if level_key in level_map:
        return int(level_map[level_key])
    if level_key.isdigit():
        return int(level_key)
    return logging.INFO
