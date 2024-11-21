"""Registry for content processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import logfire

from llmling.context.models import ProcessingContext
from llmling.core import exceptions
from llmling.core.baseregistry import BaseRegistry
from llmling.core.log import get_logger
from llmling.processors.base import BaseProcessor, ProcessorConfig, ProcessorResult
from llmling.processors.implementations.function import FunctionProcessor
from llmling.processors.implementations.template import TemplateProcessor


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.core.typedefs import ProcessingStep


logger = get_logger(__name__)


class ProcessorRegistry(BaseRegistry[BaseProcessor, str]):
    """Registry and execution manager for processors."""

    @property
    def _error_class(self) -> type[exceptions.ProcessorError]:
        return exceptions.ProcessorError

    def _validate_item(self, item: Any) -> BaseProcessor:
        """Validate processor configuration and create instance."""
        match item:
            case ProcessorConfig():
                return self._create_processor(item)
            case BaseProcessor():
                return item
            case _:
                msg = f"Invalid processor type: {type(item)}"
                raise exceptions.ProcessorError(msg)

    def _create_processor(self, config: ProcessorConfig) -> BaseProcessor:
        """Create processor instance from configuration."""
        try:
            match config.type:
                case "function":
                    return FunctionProcessor(config)
                case "template":
                    return TemplateProcessor(config)
                case _:
                    msg = f"Unknown processor type: {config.type}"
                    raise exceptions.ProcessorError(msg)  # noqa: TRY301
        except Exception as exc:
            msg = f"Failed to create processor for config {config}"
            raise exceptions.ProcessorError(msg) from exc

    async def _initialize_item(self, item: BaseProcessor) -> None:
        """Initialize processor during startup."""
        await item.startup()

    async def _cleanup_item(self, item: BaseProcessor) -> None:
        """Clean up processor during shutdown."""
        await item.shutdown()

    # Backward compatibility methods
    def register(self, key: str, item: ProcessorConfig) -> None:  # type: ignore
        """Register a new processor configuration."""
        super().register(key, item)

    async def get_processor(self, name: str) -> BaseProcessor:
        """Get a processor by name (backward compatibility)."""
        processor = self.get(name)
        if not getattr(processor, "_initialized", False):
            await processor.startup()
            processor._initialized = True  # type: ignore
        return processor

    @logfire.instrument("Processing content")
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

        # Group parallel steps together
        parallel_groups: list[list[ProcessingStep]] = [[]]
        for step in steps:
            if step.parallel and parallel_groups[-1]:
                parallel_groups[-1].append(step)
            else:
                if parallel_groups[-1]:
                    parallel_groups.append([])
                parallel_groups[-1].append(step)

        # Process each group
        result = None
        for group in parallel_groups:
            if len(group) > 1:  # Parallel processing
                try:
                    result = await self.process_parallel_steps(group, current_context)
                except Exception as exc:
                    # All parallel steps failed
                    msg = f"All parallel steps failed: {exc}"
                    raise exceptions.ProcessorError(msg) from exc
            else:  # Sequential processing
                step = group[0]
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
                    logger.warning(
                        "Optional step %s failed: %s",
                        step.name,
                        exc,
                    )
                    result = ProcessorResult(
                        content=current_context.current_content,
                        original_content=current_context.original_content,
                        metadata=current_context.metadata,
                    )

            # Update context for next group
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

    async def process_stream(
        self,
        content: str,
        steps: list[ProcessingStep],
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ProcessorResult]:
        """Process content through steps, yielding intermediate results."""
        if not self._initialized:
            await self.startup()

        current_context = ProcessingContext(
            original_content=content,
            current_content=content,
            metadata=metadata or {},
            kwargs={},
        )

        for step in steps:
            processor = await self.get_processor(step.name)

            # Create new context with step kwargs
            step_context = ProcessingContext(
                original_content=current_context.original_content,
                current_content=current_context.current_content,
                metadata=current_context.metadata,
                kwargs=step.kwargs or {},
            )

            try:
                result = await processor.process(step_context)

                # Update context for next step
                current_context = ProcessingContext(
                    original_content=content,
                    current_content=result.content,
                    metadata={**current_context.metadata, **result.metadata},
                    kwargs={},
                )

                yield result
            except Exception as exc:
                if not step.required:
                    continue
                msg = f"Step {step.name} failed"
                raise exceptions.ProcessorError(msg) from exc

    async def process_parallel_steps(
        self,
        steps: list[ProcessingStep],
        context: ProcessingContext,
    ) -> ProcessorResult:
        """Process steps in parallel."""
        results: list[ProcessorResult] = []

        for step in steps:
            step_context = ProcessingContext(
                original_content=context.original_content,
                current_content=context.current_content,
                metadata=context.metadata,
                kwargs=step.kwargs or {},
            )

            processor = await self.get_processor(step.name)
            try:
                result = await processor.process(step_context)
                results.append(result)
            except Exception as exc:
                if step.required:
                    raise
                logger.warning(
                    "Optional parallel step %s failed: %s",
                    step.name,
                    exc,
                )

        if not results:
            # If all steps failed and were optional, return original context
            return ProcessorResult(
                content=context.current_content,
                original_content=context.original_content,
                metadata=context.metadata,
            )

        # Combine successful results
        combined_content = "\n".join(r.content for r in results)
        combined_metadata: dict[str, Any] = {}
        for result in results:
            combined_metadata.update(result.metadata)

        return ProcessorResult(
            content=combined_content,
            original_content=context.original_content,
            metadata=combined_metadata,
        )
