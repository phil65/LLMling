"""Processor registry and execution management."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import logfire

from llmling.context.models import ProcessingContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import (
    BaseProcessor,
    ProcessorConfig,
    ProcessorResult,
)
from llmling.processors.implementations.function import FunctionProcessor
from llmling.processors.implementations.template import TemplateProcessor


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from llmling.core.typedefs import ProcessingStep


logger = get_logger(__name__)


class ProcessorRegistry:
    """Registry and execution manager for processors."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._processors: dict[str, BaseProcessor] = {}
        self._configs: dict[str, ProcessorConfig] = {}
        self._active = False
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        """Initialize all registered processors."""
        if self._active:
            return

        async with self._lock:
            if self._active:
                return

            try:
                for name, config in self._configs.items():
                    processor = self._create_processor(config)
                    await processor.startup()
                    self._processors[name] = processor

                self._active = True
                logger.info("Processor registry started successfully")

            except Exception as exc:
                logger.exception("Failed to start processor registry")
                await self.shutdown()
                msg = "Registry startup failed"
                raise exceptions.ProcessorError(msg) from exc

    def _create_processor(self, config: ProcessorConfig) -> BaseProcessor:
        """Create processor instance from configuration."""
        try:
            match config.type:
                case "function":
                    return FunctionProcessor(config)
                case "template":
                    return TemplateProcessor(config)
        except Exception as exc:
            msg = f"Failed to create processor for config {config}"
            raise exceptions.ProcessorError(msg) from exc
        msg = f"Unknown processor type: {config.type}"
        raise exceptions.ProcessorError(msg)

    def register(self, name: str, config: ProcessorConfig) -> None:
        """Register a new processor configuration."""
        if name in self._configs:
            msg = f"Processor {name} already registered"
            raise exceptions.ProcessorError(msg)

        self._configs[name] = config

        # If registry is already active, create and initialize the processor immediately
        if self._active:
            processor = self._create_processor(config)
            # We can't await here since this is a sync method,
            # so we'll initialize in get_processor instead
            self._processors[name] = processor

    @logfire.instrument("Processing content")
    async def process(
        self,
        content: str,
        steps: list[ProcessingStep],
        metadata: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        """Process content through steps."""
        if not self._active:
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
                    kwargs=step.parameters or {},
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

    async def shutdown(self) -> None:
        """Shutdown all processors."""
        if not self._active:
            return

        async with self._lock:
            if not self._active:
                return

            logger.info("Shutting down processor registry")
            errors: list[tuple[str, Exception]] = []

            for name, processor in self._processors.items():
                try:
                    await processor.shutdown()
                except Exception as exc:
                    logger.exception("Error shutting down processor %s", name)
                    errors.append((name, exc))

            self._processors.clear()
            self._active = False

            if errors:
                error_msgs = [f"{name}: {exc}" for name, exc in errors]
                msg = f"Errors during shutdown: {', '.join(error_msgs)}"
                raise exceptions.ProcessorError(msg)

    async def get_processor(self, name: str) -> BaseProcessor:
        """Get a processor by name."""
        if not self._active:
            await self.startup()

        try:
            processor = self._processors[name]
        except KeyError as exc:
            msg = f"Processor not found: {name}"
            raise exceptions.ProcessorError(msg) from exc
        else:
            # Initialize if not already initialized
            if not getattr(processor, "_initialized", False):
                await processor.startup()
                processor._initialized = True
            return processor

    async def process_stream(
        self,
        content: str,
        steps: list[ProcessingStep],
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ProcessorResult]:
        """Process content through steps, yielding intermediate results."""
        if not self._active:
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
                kwargs=step.parameters or {},
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

    def _group_parallel_steps(
        self,
        steps: list[ProcessingStep],
    ) -> list[list[ProcessingStep]]:
        """Group steps by parallel execution.

        Args:
            steps: Steps to group

        Returns:
            List of step groups for parallel execution
        """
        groups: list[list[ProcessingStep]] = []
        current_group: list[ProcessingStep] = []

        for step in steps:
            if step.parallel and current_group:
                current_group.append(step)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [step]

        if current_group:
            groups.append(current_group)

        return groups

    async def _process_step(
        self,
        step: ProcessingStep,
        context: ProcessingContext,
    ) -> ProcessingContext:
        """Process a single step.

        Args:
            step: Step to process
            context: Current processing context

        Returns:
            Updated context

        Raises:
            ProcessorError: If processing fails
        """
        try:
            processor = await self.get_processor(step.name)
            context.parameters = step.parameters
            result = await processor.process(context)

            return ProcessingContext(
                original_content=context.original_content,
                current_content=result.content,
                metadata={**context.metadata, **result.metadata},
            )

        except Exception as exc:
            if step.required:
                msg = f"Step {step.name} failed"
                raise exceptions.ProcessorError(msg) from exc
            logger.warning("Optional step %s failed: %s", step.name, exc)
            return context

    async def process_parallel_steps(
        self,
        steps: list[ProcessingStep],
        context: ProcessingContext,
    ) -> ProcessorResult:
        """Process steps in parallel."""
        results = []

        for step in steps:
            step_context = ProcessingContext(
                original_content=context.original_content,
                current_content=context.current_content,
                metadata=context.metadata,
                kwargs=step.parameters or {},
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
        combined_metadata = {}
        for result in results:
            combined_metadata.update(result.metadata)

        return ProcessorResult(
            content=combined_content,
            original_content=context.original_content,
            metadata=combined_metadata,
        )
