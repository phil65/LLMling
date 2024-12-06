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

## Event Handling

```python
from llmling.core.events import Event, EventHandler

class MyEventHandler(EventHandler):
    async def handle_event(self, event: Event) -> None:
        match event.type:
            case "RESOURCE_MODIFIED":
                print(f"Resource changed: {event.name}")
            case "TOOL_ADDED":
                print(f"New tool: {event.name}")

async with RuntimeConfig.open("config.yml") as runtime:
    # Add event handler
    runtime.add_event_handler(MyEventHandler())
```

## Registry Observation

```python
from llmling.core.events import RegistryEvents

class ResourceObserver(RegistryEvents):
    def on_item_added(self, key: str, item: Any) -> None:
        print(f"Resource added: {key}")

    def on_item_modified(self, key: str, item: Any) -> None:
        print(f"Resource modified: {key}")

async with RuntimeConfig.open("config.yml") as runtime:
    # Add observers
    runtime.add_observer(ResourceObserver(), registry_type="resource")
    runtime.add_observer(PromptObserver(), registry_type="prompt")
    runtime.add_observer(ToolObserver(), registry_type="tool")
```

## Example: Agent Integration

Here's an example of using LLMling with an agent:

```python
from llmling import RuntimeConfig
from llmling_agent import LLMlingAgent
from pydantic import BaseModel

# Define structured output
class Analysis(BaseModel):
    summary: str
    complexity: int
    suggestions: list[str]

# Create agent with runtime
async with RuntimeConfig.open("config.yml") as runtime:
    # Create agent with structured output
    agent = LLMlingAgent[Analysis](
        runtime,
        result_type=Analysis,
        model="openai:gpt-4",
        system_prompt=[
            "You are a code analysis assistant.",
            "Provide structured analysis results.",
        ],
    )

    # Use the agent
    result = await agent.run(
        "Analyze the Python code in resources/main.py"
    )

    # Access structured results
    print(f"Summary: {result.data.summary}")
    print(f"Complexity: {result.data.complexity}")
    for suggestion in result.data.suggestions:
        print(f"- {suggestion}")
```
