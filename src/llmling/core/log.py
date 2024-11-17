"""Logging configuration for llmling."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import logfire


if TYPE_CHECKING:
    from collections.abc import Sequence


def setup_logging(
    *,
    level: int = logging.INFO,
    handlers: Sequence[logging.Handler] | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for llmling.

    Args:
        level: The logging level to use
        handlers: Optional sequence of handlers to add
        format_string: Optional custom format string
    """
    # Configure logfire first
    logfire.configure()
    logger = logging.getLogger("llmling")
    logger.setLevel(level)

    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    if not handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handlers = [handler]

    for handler in handlers:
        if not handler.formatter:
            handler.setFormatter(formatter)
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with 'llmling.'

    Returns:
        A logger instance
    """
    return logging.getLogger(f"llmling.{name}")
