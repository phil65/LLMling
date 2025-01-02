from __future__ import annotations

import asyncio


def reverse_text(text: str) -> str:
    """Reverse input text."""
    return text[::-1]


def uppercase_text(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def multiply(text: str, times: int = 2) -> str:
    """Convert text to uppercase."""
    return text * times


def append_text(text: str, suffix: str = "!") -> str:
    """Append suffix to text."""
    return f"{text}{suffix}"


async def async_reverse_text(text: str) -> str:
    """Reverse text asynchronously."""
    await asyncio.sleep(0.1)  # Simulate async work
    return text[::-1]


async def failing_processor(text: str) -> str:
    """Test helper that fails."""
    msg = "Test failure"
    raise ValueError(msg)
