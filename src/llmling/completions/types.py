from __future__ import annotations

from collections.abc import Callable


type CompletionFunction = Callable[[str], list[str]]
"""Type for completion functions. Takes current value, returns possible completions."""
