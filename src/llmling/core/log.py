"""Core logging configuration for llmling."""

from __future__ import annotations

from datetime import datetime
import logging
import sys
from typing import TYPE_CHECKING, Literal

import platformdirs
from upath import UPath


if TYPE_CHECKING:
    from collections.abc import Sequence

# Get platform-specific log directory
LOG_DIR = UPath(platformdirs.user_log_dir("llmling", "llmling"))
# Include date in log filename for parallel runs
LOG_FILE = LOG_DIR / f"llmling_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

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
    mode: Literal["client", "server"] = "client",
) -> list[logging.Handler]:
    """Configure core logging for llmling.

    Args:
        level: The logging level for console output
        handlers: Optional sequence of handlers to add
        format_string: Optional custom format string
        log_to_file: Whether to log to file in addition to stdout
        mode: Whether running as client or server (affects stdout usage)

    Returns:
        List of configured handlers
    """
    # Configure logfire first
    import logfire

    logfire.configure()

    logger = logging.getLogger("llmling")
    logger.setLevel(logging.DEBUG)  # Always set root logger to DEBUG

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    configured_handlers: list[logging.Handler] = []

    if not handlers:
        # Add stdout handler only in client mode
        if mode == "client":
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            stdout_handler.setLevel(level)
            configured_handlers.append(stdout_handler)

        # Add file handler if requested (always DEBUG level)
        if log_to_file:
            try:
                # Create log directory if it doesn't exist
                LOG_DIR.mkdir(parents=True, exist_ok=True)

                # Use RotatingFileHandler for log rotation
                from logging.handlers import RotatingFileHandler

                file_handler = RotatingFileHandler(
                    LOG_FILE,
                    maxBytes=MAX_LOG_SIZE,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
                configured_handlers.append(file_handler)
            except Exception as exc:  # noqa: BLE001
                if mode == "client":  # Only print to stderr in client mode
                    print(
                        f"Failed to setup file logging at {LOG_FILE}: {exc}",
                        file=sys.stderr,
                    )
    else:
        configured_handlers.extend(handlers)
        for handler in configured_handlers:
            if not handler.formatter:
                handler.setFormatter(formatter)

    # Add all handlers to logger
    for handler in configured_handlers:
        logger.addHandler(handler)

    # Log startup info (will only go to appropriate handlers)
    logger.info("Logging initialized")
    if log_to_file:
        logger.debug(
            "Mode: %s, Console logging level: %s, File logging level: DEBUG (%s)",
            mode,
            logging.getLevelName(level),
            LOG_FILE,
        )

    return configured_handlers


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with 'llmling.'

    Returns:
        A logger instance
    """
    return logging.getLogger(f"llmling.{name}")
