"""Registry for content processors."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.processors.base import (
    BaseProcessor,
    Processor,
    ProcessorConfig,
    ProcessorResult,
)
from llmling.resources.models import ProcessingContext


if TYPE_CHECKING:
    from llmling.core.typedefs import ProcessingStep


logger = get_logger(__name__)


class ProcessorRegistry(BaseRegistry[str, BaseProcessor]):
    """Registry for content processors."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize registry with builtin processors."""
        super().__init__(*args, **kwargs)

    @property
    def _error_class(self) -> type[exceptions.ProcessorError]:
        return exceptions.ProcessorError

    def _validate_item(self, item: Any) -> BaseProcessor:
        """Validate and transform items."""
        match item:
            case BaseProcessor():
                return item
            case ProcessorConfig():
                return Processor(item)  # Creates function-based processor
            case _ if callable(item):
                path = f"{item.__module__}.{item.__qualname__}"
                is_coro = asyncio.iscoroutinefunction(item)
                config = ProcessorConfig(import_path=path, async_execution=is_coro)
                return Processor(config)
            case _:
                msg = f"Invalid processor type: {type(item)}"
                raise exceptions.ProcessorError(msg)

    async def process(
        self,
        content: str,
        steps: list[ProcessingStep],
        metadata: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        """Process content through steps."""
        if not self._initialized:
            await self.startup()

        current_context = ProcessingContext(
            original_content=content,
            current_content=content,
            metadata=metadata or {},
            kwargs={},
        )

        result = None
        for step in steps:
            step_context = ProcessingContext(
                original_content=current_context.original_content,
                current_content=current_context.current_content,
                metadata=current_context.metadata,
                kwargs=step.kwargs or {},
            )

            processor = await self.get_processor(step.name)
            try:
                result = await processor.process(step_context)
            except Exception as exc:
                if step.required:
                    msg = f"Required step {step.name} failed: {exc}"
                    raise exceptions.ProcessorError(msg) from exc

                # Optional step failed, continue with current context
                logger.warning("Optional step %s failed: %s", step.name, exc)
                result = ProcessorResult(
                    content=current_context.current_content,
                    original_content=current_context.original_content,
                    metadata=current_context.metadata,
                )

            # Update context for next step
            if result:
                current_context = ProcessingContext(
                    original_content=content,
                    current_content=result.content,
                    metadata={**current_context.metadata, **result.metadata},
                    kwargs={},
                )

        return (
            result
            if result
            else ProcessorResult(
                content=content,
                original_content=content,
                metadata=current_context.metadata,
            )
        )

    async def get_processor(self, name: str) -> BaseProcessor:
        """Get a processor by name."""
        processor = self.get(name)
        if not getattr(processor, "_initialized", False):
            await processor.startup()
            processor._initialized = True
        return processor
