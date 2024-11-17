# llmling/core/utils.py
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Generator

    from llmling.core.exceptions import LLMLingError


T = TypeVar("T")

@contextmanager
def error_handler(
    error_class: type[LLMLingError],
    message: str,
) -> Generator[None, None, None]:
    """Handle exceptions and wrap them with custom error.

    Args:
        error_class: The error class to raise
        message: The error message format

    Example:
        with error_handler(ConfigError, "Failed to load config"):
            config = load_config()
    """
    try:
        yield
    except Exception as exc:
        msg = f"{message}: {exc}"
        raise error_class(msg) from exc
