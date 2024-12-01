"""Logging configuration for llmling."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import TYPE_CHECKING

import logfire
import platformdirs
from upath import UPath


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import LoggingLevel


# Map Python logging levels to MCP logging levels
LEVEL_MAP: dict[int, LoggingLevel] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}

# Get platform-specific log directory
LOG_DIR = UPath(platformdirs.user_log_dir("llmling", "llmling"))
LOG_FILE = LOG_DIR / "llmling.log"

# Maximum log file size in bytes (10MB)
MAX_LOG_SIZE = 10 * 1024 * 1024
# Number of backup files to keep
BACKUP_COUNT = 5


def setup_logging(
    *,
    level: int | str = logging.INFO,
    handlers: Sequence[logging.Handler] | None = None,
    format_string: str | None = None,
    log_to_file: bool = True,
) -> None:
    """Configure logging for llmling.

    Args:
        level: The logging level for console output
        handlers: Optional sequence of handlers to add
        format_string: Optional custom format string
        log_to_file: Whether to log to file in addition to stdout
    """
    logfire.configure()
    logger = logging.getLogger("llmling")
    logger.setLevel(logging.DEBUG)  # Always set root logger to DEBUG

    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    if not handlers:
        handlers = []
        # Add stdout handler with user-specified level
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(level)
        handlers.append(stdout_handler)

        # Add file handler if requested (always DEBUG level)
        if log_to_file:
            try:
                # Create log directory if it doesn't exist
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    LOG_FILE,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
                handlers.append(file_handler)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Failed to setup file logging at {LOG_FILE}: {exc}",
                    file=sys.stderr,
                )

    for handler in handlers:
        if not handler.formatter:
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Log startup info with both handlers' levels
    logger.info("Logging initialized")
    if log_to_file:
        logger.debug(
            "Console logging level: %s, File logging level: DEBUG (%s)",
            logging.getLevelName(level),
            LOG_FILE,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with 'llmling.'

    Returns:
        A logger instance
    """
    return logging.getLogger(f"llmling.{name}")
