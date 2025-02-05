"""Logging configuration for llmling."""

from __future__ import annotations

from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import platformdirs


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
LOG_DIR = Path(platformdirs.user_log_dir("llmling", "llmling"))
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
LOG_FILE = LOG_DIR / f"llmling_{TIMESTAMP}.log"

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
    logger = logging.getLogger("llmling")
    logger.setLevel(logging.DEBUG)

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
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    LOG_FILE,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                handlers.append(file_handler)
            except Exception as exc:  # noqa: BLE001
                msg = f"Failed to create log file: {exc}"
                print(msg, file=sys.stderr)

    for handler in handlers:
        if not handler.formatter:
            handler.setFormatter(formatter)
        logger.addHandler(handler)

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
