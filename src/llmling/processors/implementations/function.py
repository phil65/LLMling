"""Function-based processor implementation."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import TYPE_CHECKING, Any

import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import ChainableProcessor, ProcessorConfig, ProcessorResult


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.context.models import ProcessingContext


logger = get_logger(__name__)


class FunctionProcessor(ChainableProcessor):
    """Processor that executes a Python function."""

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize processor."""
        super().__init__(config)
        self.func: Callable[..., Any] | None = None

    async def startup(self) -> None:
        """Load function during startup."""
        if not self.config.import_path:
            msg = "Import path not configured"
            raise exceptions.ProcessorError(msg)

        try:
            self.func = self._load_function()
        except Exception as exc:
            msg = f"Failed to load function: {exc}"
            raise exceptions.ProcessorError(msg) from exc
        else:
            if not callable(self.func):
                msg = f"Loaded object {self.config.import_path} is not callable"
                raise exceptions.ProcessorError(msg)

    def _load_function(self) -> Callable[..., Any]:
        """Load function from import path."""
        try:
            module_path, func_name = self.config.import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            msg = f"Cannot import {self.config.import_path}"
            raise exceptions.ProcessorError(msg) from exc
        except Exception as exc:
            msg = f"Unexpected error loading {self.config.import_path}"
            raise exceptions.ProcessorError(msg) from exc

    @logfire.instrument("Executing function processor")
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Execute function with content."""
        if not self.func:
            msg = "Processor not initialized"
            raise exceptions.ProcessorError(msg)

        try:
            # Execute function
            result = self.func(context.current_content, **context.parameters)

            # Handle async functions
            if inspect.iscoroutine(result):
                result = await result

            # Convert result to string
            content = str(result)

            return ProcessorResult(
                content=content,
                original_content=context.original_content,
                metadata={
                    "function": self.config.import_path,
                    "is_async": asyncio.iscoroutinefunction(self.func),
                },
            )
        except Exception as exc:
            msg = f"Function execution failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
