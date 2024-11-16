"""Context processing implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import importlib
from typing import Any

import jinja2
from pydantic import BaseModel, field_validator


class ProcessorNotFoundError(Exception):
    """Raised when a processor cannot be found."""


class ProcessorConfig(BaseModel):
    """Configuration for a processor."""

    type: str
    import_path: str | None = None
    template: str | None = None

    @field_validator("*")
    @classmethod
    def validate_processor_type(cls, v: Any, info: Any) -> Any:
        """Validate processor configuration based on type."""
        if info.field_name == "type" and v not in ["function", "template"]:
            msg = f"Invalid processor type: {v}"
            raise ValueError(msg)

        if v is None:
            if info.field_name == "import_path" and info.data.get("type") == "function":
                msg = "import_path is required for function processors"
                raise ValueError(msg)
            if info.field_name == "template" and info.data.get("type") == "template":
                msg = "template is required for template processors"
                raise ValueError(msg)
        return v


class ProcessingStep(BaseModel):
    """Configuration for a processing step."""

    name: str
    keyword_args: dict[str, Any] = {}

    model_config = {
        "frozen": True,
    }


class Processor:
    """Processor that can be either a function or template."""

    def __init__(self, config: ProcessorConfig):
        """Initialize processor from config."""
        self.config = config

        if config.type == "function" and config.import_path:
            self.processor = self._load_function(config.import_path)
        elif config.template:
            self.processor = self._create_template(config.template)
        else:
            msg = "Invalid processor configuration"
            raise ProcessorError(msg)

    def _load_function(self, import_path: str) -> Callable[[str], str]:
        """Load function from import path."""
        try:
            module_path, func_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            msg = f"Failed to load processor function: {import_path}"
            raise ProcessorNotFoundError(msg) from exc

    def _create_template(self, template_str: str) -> Callable[[str], str]:
        """Create template processor."""
        env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=True)
        template = env.from_string(template_str)

        def process(content: str, **kwargs: Any) -> str:
            return template.render(content=content, **kwargs)

        return process

    def __call__(self, content: str, **kwargs: Any) -> str:
        """Process the content."""
        return self.processor(content, **kwargs)


class ProcessorRegistry:
    """Registry for processors."""

    def __init__(self) -> None:
        self.processors: dict[str, Processor] = {}

    def register(self, name: str, config: ProcessorConfig) -> None:
        """Register a new processor."""
        self.processors[name] = Processor(config)

    def get(self, name: str) -> Processor:
        """Get a processor by name."""
        if name not in self.processors:
            msg = f"Processor not found: {name}"
            raise ProcessorNotFoundError(msg)
        return self.processors[name]


async def process_context(
    content: str,
    steps: list[ProcessingStep],
    registry: ProcessorRegistry,
    *,
    parallel: bool = False,
) -> str:
    """Process content through a chain of processors.

    Args:
        content: The input content to process
        steps: List of processing steps to apply
        registry: Registry containing available processors
        parallel: Whether to process steps in parallel when possible

    Returns:
        Processed content after applying all steps

    Raises:
        ProcessorNotFoundError: If a processor is not found in the registry
    """
    if not steps:
        return content

    if parallel:
        return await _process_parallel(content, steps, registry)

    return await _process_sequential(content, steps, registry)


async def _process_sequential(
    content: str,
    steps: list[ProcessingStep],
    registry: ProcessorRegistry,
) -> str:
    """Process steps sequentially."""
    processed = content
    for step in steps:
        processor = registry.get(step.name)
        try:
            result = processor(processed, **step.keyword_args)
            if isinstance(result, Awaitable):
                processed = await result
            else:
                processed = result
        except Exception as exc:
            msg = f"Error in processor {step.name}"
            raise ProcessorError(msg) from exc
    return processed


async def _process_parallel(
    content: str,
    steps: list[ProcessingStep],
    registry: ProcessorRegistry,
) -> str:
    """Process independent steps in parallel when possible."""

    async def _process_step(step: ProcessingStep) -> str:
        processor = registry.get(step.name)
        try:
            result = processor(content, **step.keyword_args)
            if isinstance(result, Awaitable):
                return await result
        except Exception as exc:
            msg = f"Error in processor {step.name}"
            raise ProcessorError(msg) from exc
        else:
            return result

    results = await asyncio.gather(
        *[_process_step(step) for step in steps],
        return_exceptions=True,
    )

    # Check for exceptions
    for result in results:
        if isinstance(result, Exception):
            raise result

    # Combine results (this is just one way - you might want a different strategy)
    return "\n".join(str(r) for r in results)


class ProcessorError(Exception):
    """Raised when a processor fails to process content."""


async def _main() -> None:
    """Example usage of process_context."""
    # Create registry and register processors
    registry = ProcessorRegistry()

    # Register a simple uppercase processor
    registry.register(
        "uppercase",
        ProcessorConfig(
            type="function",
            import_path="reprlib.repr",
        ),
    )

    # Register a template processor
    registry.register(
        "wrap",
        ProcessorConfig(
            type="template",
            template="[{{ content }}]",
        ),
    )

    # Define processing steps
    steps = [
        ProcessingStep(name="uppercase"),
        ProcessingStep(name="wrap"),
    ]

    # Process content sequentially
    content = "Hello, world!"
    result = await process_context(content, steps, registry)
    print("Sequential processing:", result)

    # Process content in parallel
    result = await process_context(content, steps, registry, parallel=True)
    print("Parallel processing:", result)


if __name__ == "__main__":
    asyncio.run(_main())
