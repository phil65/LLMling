from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class ImmediateAction:
    """Represents an action that should be executed immediately."""

    action: Callable[..., Awaitable[Any]]
    args: tuple[Any]
    kwargs: dict[str, Any]
    requires_confirmation: bool = False
    description: str = ""  # Human readable description of what the action will do

    async def execute(self) -> Any:
        """Execute the immediate action."""
        if self.requires_confirmation and not await self.get_confirmation():
            return {"status": "cancelled", "reason": "User declined"}

        try:
            result = await self.action(*self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}
        else:
            return {"status": "success", "result": result}

    async def get_confirmation(self) -> bool:
        """Get user confirmation for the action."""
        print(f"\nRequested action: {self.description}")
        response = input("Execute this action? (y/n): ")
        return response.lower() == "y"
