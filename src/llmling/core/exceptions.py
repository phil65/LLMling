"""Core exceptions for the llmling package."""

from __future__ import annotations


class LLMLingError(Exception):
    """Base exception for all llmling errors."""


class ConfigError(LLMLingError):
    """Configuration related errors."""


class ResourceError(LLMLingError):
    """Base class for context-related errors."""


class LoaderError(ResourceError):
    """Error during context loading."""


class ProcessorError(LLMLingError):
    """Base class for processor-related errors."""


class ProcessorNotFoundError(ProcessorError):
    """Raised when a processor cannot be found."""


class ValidationError(LLMLingError):
    """Validation related errors."""


class LLMError(LLMLingError):
    """LLM related errors."""


class ResourceResolutionError(ResourceError):
    """Raised when resource resolution fails."""
