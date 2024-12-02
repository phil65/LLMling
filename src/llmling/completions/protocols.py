from __future__ import annotations

from typing import Any, Protocol


class CompletionProvider(Protocol):
    """Protocol for completion providers."""

    async def get_completions(
        self,
        current_value: str,
        argument_name: str | None = None,
        **options: Any,
    ) -> list[str]: ...
