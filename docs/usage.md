# Programmatic Usage

While LLMling is designed for declarative configuration, it provides a clean Python API for programmatic usage.

## RuntimeConfig

The main interface for using LLMling programmatically is the `RuntimeConfig` class:

```python
from llmling import Config, RuntimeConfig

# Create from YAML file
async with RuntimeConfig.open("config.yml") as runtime:
    # Use runtime...
    pass

```

> **Important**
> Always use RuntimeConfig as a context manager to ensure proper resource cleanup.

## Resource Operations

```python
async with RuntimeConfig.open("config.yml") as runtime:
    # Load a resource
    resource = await runtime.load_resource("my_resource")
    print(resource.content)
    print(resource.metadata)

    # List available resources
    resources = runtime.list_resource_names()

    # Get resource URI
    uri = runtime.get_resource_uri("my_resource")

    # Load by URI
    resource = await runtime.load_resource_by_uri(uri)

    # Register new resource
    from llmling.config.models import TextResource

    resource = TextResource(content="Hello, World!")
    runtime.register_resource("new_resource", resource)
```

## Prompt Operations

```python
async with RuntimeConfig.open("config.yml") as runtime:
    # Format a prompt
    messages = await runtime.render_prompt(
        "my_prompt",
        arguments={"name": "World"}
    )

    # List available prompts
    prompts = runtime.list_prompt_names()

    # Get prompt by name
    prompt = runtime.get_prompt("my_prompt")

    # Get all prompts
    all_prompts = runtime.get_prompts()
```

## Tool Operations

```python
async with RuntimeConfig.open("config.yml") as runtime:
    # List available tools
    tools = runtime.list_tool_names()

    # Get tool by name
    tool = runtime.get_tool("my_tool")

    # Get all tools
    all_tools = runtime.get_tools()

    # Execute a tool
    result = await runtime.execute_tool(
        "my_tool",
        arg1="value1",
        arg2="value2"
    )

```

## Registry Events

All registries are `BaseRegistry` subclasses backed by a psygnal `EventedDict`.
Their `.events` signal group emits `adding`, `added`, `removing`, `removed`,
`changing` and `changed`:

```python
from llmling.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.events.added.connect(lambda key, tool: print(f"Tool added: {key}"))
registry.register("len", len)  # prints "Tool added: len"
```

## Agent Integration

LLMling was the backend of [llmling-agent](https://github.com/phil65/llmling-agent),
which has since been succeeded by [agentpool](https://github.com/phil65/agentpool).
Neither currently consumes a `RuntimeConfig` directly; use the runtime API above
to feed resources, prompts and tools into your agent framework of choice.
