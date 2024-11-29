from __future__ import annotations

from collections.abc import Callable


type CompletionCallable = Callable[[str], list[str]]
"""Type for completion functions. Takes current value, returns possible completions."""

CompletionFunction = CompletionCallable | None
"""A completion function or None."""
