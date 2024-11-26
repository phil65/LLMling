"""MCP protocol handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.server.handlers.base import HandlerBase
from llmling.server.handlers.logger import LoggingHandlers
from llmling.server.handlers.prompts import PromptHandlers
from llmling.server.handlers.resources import ResourceHandlers
from llmling.server.handlers.tools import ToolHandlers

if TYPE_CHECKING:
    from llmling.server import LLMLingServer

__all__ = [
    "HandlerBase",
    "LoggingHandlers",
    "PromptHandlers",
    "ResourceHandlers",
    "ToolHandlers",
]
