from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from mcp.server import Server


class TransportBase(ABC):
    """Base class for transport implementations."""

    def __init__(self, server: Server) -> None:
        """Initialize transport with server instance."""
        self.server = server

    @abstractmethod
    async def serve(self, *, raise_exceptions: bool = False) -> None:
        """Start serving the transport.

        Args:
            raise_exceptions: Whether to raise exceptions for debugging
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the transport."""

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()
