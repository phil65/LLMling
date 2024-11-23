"""Collection of content processors for text transformation pipelines."""

from __future__ import annotations

from llmling.processors.base import (
    AsyncProcessor,
    BaseProcessor,
    ChainableProcessor,
    ProcessorConfig,
    ProcessorResult,
)
from llmling.processors.implementations.function import FunctionProcessor
from llmling.processors.implementations.template import TemplateProcessor
from llmling.processors.registry import ProcessorRegistry


__all__ = [
    "AsyncProcessor",
    "BaseProcessor",
    "ChainableProcessor",
    "FunctionProcessor",
    "ProcessorConfig",
    "ProcessorRegistry",
    "ProcessorResult",
    "TemplateProcessor",
]
