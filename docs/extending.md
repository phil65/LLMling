# Extension Points

LLMling provides several ways to extend its functionality. This section covers the main extension points and how to use them.

## Entry Points

LLMling uses Python's entry point system for extensions. The main entry points are:

```toml
# pyproject.toml
[project.entry-points.llmling]
tools = "myapp.tools:get_mcp_tools"
processors = "myapp.processors:get_processors"
prompts = "myapp.prompts:get_prompts"
```

## Custom Tools

The most common extension point is adding custom tools:

```python
# myapp/tools.py
from typing import Any, Callable
from llmling.tools.base import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Custom tool description"

    async def execute(self, **params: Any) -> Any:
        """Tool implementation."""
        return "Result"

def get_mcp_tools() -> list[Callable[..., Any]]:
    """Tool entry point."""
    return [
        CustomTool,  # Class-based tool
        my_function_tool,  # Function-based tool
    ]
```

## Custom Processors

Add custom content processors:

```python
# myapp/processors.py
from typing import Any
from llmling.processors.base import BaseProcessor, ProcessorResult
from llmling.resources.models import ProcessingContext

class CustomProcessor(BaseProcessor):
    async def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process content with given context."""
        result = context.current_content.upper()
        return ProcessorResult(
            content=result,
            original_content=context.original_content,
            metadata={"processed_by": "custom"}
        )

def get_processors() -> dict[str, Any]:
    """Processor entry point."""
    return {
        "custom": CustomProcessor(),
        "simple": my_processor_function,
    }
```

## Custom Resource Types

Create new resource types by extending the base loader:

```python
# myapp/resources.py
from typing import AsyncIterator
from llmling.resources.base import ResourceLoader
from llmling.resources.models import LoadedResource
from llmling.config.models import BaseResource
from llmling.processors.registry import ProcessorRegistry

class CustomResource(BaseResource):
    """Custom resource configuration."""
    resource_type: str = "custom"
    # Add custom fields...

class CustomResourceLoader(ResourceLoader[CustomResource]):
    """Custom resource loader implementation."""

    context_class = CustomResource
    uri_scheme = "custom"

    async def _load_impl(
        self,
        resource: CustomResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Implementation of loading logic."""
        # Load content...
        yield create_loaded_resource(
            content="content",
            source_type="custom",
            uri=self.create_uri(name=name),
            # ...
        )
```

## Custom Prompt Types

Implement custom prompt types:

```python
# myapp/prompts.py
from typing import Any
from llmling.prompts.models import BasePrompt, PromptMessage

class CustomPrompt(BasePrompt):
    """Custom prompt implementation."""

    async def format(
        self,
        arguments: dict[str, Any] | None = None
    ) -> list[PromptMessage]:
        """Format prompt with arguments."""
        # Format messages...
        return [
            PromptMessage(
                role="user",
                content="Formatted content"
            )
        ]
```

## Event Handling

Subscribe to runtime events:

```python
from llmling.core.events import Event, EventHandler

class CustomEventHandler(EventHandler):
    async def handle_event(self, event: Event) -> None:
        """Handle runtime events."""
        match event.type:
            case "RESOURCE_MODIFIED":
                print(f"Resource modified: {event.name}")
            case "TOOL_ADDED":
                print(f"Tool added: {event.name}")
```

## Registry Extensions

Create custom registries for new component types:

```python
from typing import Any
from llmling.core.baseregistry import BaseRegistry
from llmling.core.exceptions import LLMLingError

class CustomRegistry(BaseRegistry[str, Any]):
    """Custom component registry."""

    @property
    def _error_class(self) -> type[LLMLingError]:
        return LLMLingError

    def _validate_item(self, item: Any) -> Any:
        """Validate registry items."""
        if not isinstance(item, CustomComponent):
            msg = f"Invalid item type: {type(item)}"
            raise self._error_class(msg)
        return item
```

## Best Practices

### Entry Points
- Use descriptive names
- Document return types
- Handle import errors gracefully
- Validate components before returning

### Custom Components
- Follow base class patterns
- Document public interfaces
- Use type hints consistently
- Handle errors appropriately
- Clean up resources properly

### Testing
- Write unit tests for components
- Test error conditions
- Verify resource cleanup
- Check type hint correctness
- Test async behavior

## Next Steps

The next section covers [Programmatic Usage](#programmatic-usage), which explains how to use LLMling in your Python code.
