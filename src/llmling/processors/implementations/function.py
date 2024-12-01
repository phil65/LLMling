"""Function-based processor implementation."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import ChainableProcessor, ProcessorConfig, ProcessorResult
from llmling.utils import importing


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling.resources.models import ProcessingContext


logger = get_logger(__name__)


class FunctionProcessor(ChainableProcessor):
    """Processor that executes a Python function."""

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize processor."""
        super().__init__(config)
        self.func_config = config.get_function_config()
        self.func: Callable[..., Any] | None = None

    async def startup(self) -> None:
        """Load function during startup."""
        if not self.config.import_path:
            msg = "Import path not configured"
            raise exceptions.ProcessorError(msg)

        try:
            self.func = importing.import_callable(self.config.import_path)
        except ValueError as exc:
            msg = f"Failed to load function: {exc}"
            raise exceptions.ProcessorError(msg) from exc

    @logfire.instrument("Executing function processor")
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Execute function with content."""
        if not self.func:
            msg = "Processor not initialized"
            raise exceptions.ProcessorError(msg)

        try:
            # Execute function
            result = self.func(context.current_content, **context.kwargs)

            # Handle async functions
            if inspect.iscoroutine(result):
                result = await result

            # Convert result to string
            content = str(result)
            is_coro = asyncio.iscoroutinefunction(self.func)
            return ProcessorResult(
                content=content,
                original_content=context.original_content,
                metadata={"function": self.config.import_path, "is_async": is_coro},
            )
        except Exception as exc:
            msg = f"Function execution failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
