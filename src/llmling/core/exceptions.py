"""Core exceptions for the llmling package."""

from __future__ import annotations


class LLMLingError(Exception):
    """Base exception for all llmling errors."""


class ConfigError(LLMLingError):
    """Configuration related errors."""


class ContextError(LLMLingError):
    """Base class for context-related errors."""


class LoaderError(ContextError):
    """Error during context loading."""


class ProcessorError(LLMLingError):
    """Base class for processor-related errors."""


class ProcessorNotFoundError(ProcessorError):
    """Raised when a processor cannot be found."""


class ValidationError(LLMLingError):
    """Validation related errors."""


class ProviderError(LLMLingError):
    """Provider related errors."""


class TaskError(LLMLingError):
    """Task execution related errors."""


class LLMError(LLMLingError):
    """LLM related errors."""


class ImageProcessingError(ProcessorError):
    """Error during image processing."""


class UnsupportedImageFormatError(ImageProcessingError):
    """Image format not supported."""


class ImageValidationError(ValidationError):
    """Image validation failed."""
