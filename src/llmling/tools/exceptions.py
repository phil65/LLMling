"""Exceptions for the tool system."""

from __future__ import annotations

from llmling.core.exceptions import LLMLingError


class ToolError(LLMLingError):
    """Base exception for tool-related errors."""


class ToolExecutionError(ToolError):
    """Error during tool execution."""


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""


class ToolValidationError(ToolError):
    """Tool parameter validation error."""
