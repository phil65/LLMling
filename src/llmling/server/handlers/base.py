"""Base handler functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from llmling.server import LLMLingServer

T = TypeVar("T", bound="HandlerBase")


class HandlerBase(ABC):
    """Base class for protocol handlers."""

    server: LLMLingServer  # Type hint for the server attribute

    def __init__(self, server: LLMLingServer) -> None:
        """Initialize handler.

        Args:
            server: Server instance
        """
        self.server = server

    @abstractmethod
    def register(self) -> None:
        """Register handlers with server."""
