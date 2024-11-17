"""Collection of content processors for text transformation pipelines.

This package provides a flexible framework for defining and executing content processing
pipelines. It includes base classes for processors, a registry system for managing
processor instances, and implementations for function-based and template-based
processing.

Key Components:
- ProcessorConfig: Configuration model for processors
- BaseProcessor: Abstract base class for all processors
- ChainableProcessor: Base class for processors that can be chained
- ProcessorRegistry: Central registry and execution manager
- FunctionProcessor: Executes Python functions on content
- TemplateProcessor: Applies Jinja2 templates to content

Interface Examples:

Creating and registering a processor:
```python
config = ProcessorConfig(
    type="function",
    name="my_processor",
    import_path="my_module.process_func",
    async_execution=True
)
registry = ProcessorRegistry()
registry.register("my_processor", config)
```

Processing content:
```python
steps = [
    ProcessingStep(name="my_processor", required=True),
    ProcessingStep(name="template_proc", parallel=True),
]
result = await registry.process("input text", steps)
```

Streaming processing:
```python
async for result in registry.process_stream("input text", steps):
    print(result.content)
```

Creating a custom processor:
```python
class MyProcessor(ChainableProcessor):
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        # Process content here
        return ProcessorResult(
            content="processed content",
            original_content=context.current_content
        )
```

The package supports both sequential and parallel processing steps, error handling,
and result validation. Processors can be configured via dependency injection and
support async operations.
"""

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
    # Base classes
    "AsyncProcessor",
    "BaseProcessor",
    "ChainableProcessor",
    "ProcessorConfig",
    "ProcessorResult",
    # Implementations
    "FunctionProcessor",
    "TemplateProcessor",
    # Registry
    "ProcessorRegistry",
]
