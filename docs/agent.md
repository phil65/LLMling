# Agent

LLMling provides an agent implementation that integrates with its resource and tool system. The agent can be used standalone or as part of a larger application.

## Basic Usage

Create and use an agent:

```python
from llmling import RuntimeConfig
from llmling.agents import LLMlingAgent
from pydantic import BaseModel

# Optional: Define structured output
class Analysis(BaseModel):
    summary: str
    suggestions: list[str]

async with RuntimeConfig.from_file("config.yml") as runtime:
    # Create agent with string output. It will have all resources and tools available from the config.
    basic_agent = LLMlingAgent(
        runtime,
        model="openai:gpt-4",
        system_prompt="You are a helpful assistant."
    )
    result = await basic_agent.run("Analyze this code.")
    print(result.data)  # string response

    # Create agent with structured output
    typed_agent = LLMlingAgent[Analysis](
        runtime,
        result_type=Analysis,
        model="openai:gpt-4",
        system_prompt=[
            "You are a code analysis assistant.",
            "Always provide structured results.",
        ]
    )
    result = await typed_agent.run("Analyze this code.")
    print(result.data.summary)         # Typed access
    print(result.data.suggestions)     # Type-checked
```

## Agent Configuration

The agent can be configured with various options:

```python
agent = LLMlingAgent(
    runtime,
    # Model settings
    model="openai:gpt-4",            # Model to use
    result_type=Analysis,            # Optional result type

    # Prompt configuration
    system_prompt=[                  # Static system prompts
        "You are an assistant.",
        "Be concise and clear.",
    ],
    name="code-assistant",          # Agent name

    # Execution settings
    retries=3,                      # Max retries
    result_tool_name="output",      # Tool name for final result
    result_tool_description="Final analysis output",
    defer_model_check=False,        # Check model on init
)
```

## Running the Agent

Different ways to run the agent:

```python
# Basic run
result = await agent.run("Analyze this code.")

# With message history
from pydantic_ai import messages

history = [
    messages.Message(role="user", content="Previous message"),
    messages.Message(role="assistant", content="Previous response")
]
result = await agent.run("Follow up question", message_history=history)

# Stream responses
async with agent.run_stream("Analyze this.") as stream:
    async for message in stream:
        print(message.content)

# Synchronous operation (convenience wrapper)
result = agent.run_sync("Quick question")
```

## Customizing Agent Behavior

Add custom tools and system prompts:

```python
class CodeAgent(LLMlingAgent[Analysis]):
    def __init__(self, runtime: RuntimeConfig) -> None:
        super().__init__(
            runtime,
            result_type=Analysis,
            model="openai:gpt-4"
        )
        self._setup_tools()

    def _setup_tools(self) -> None:
        @self.tool
        async def analyze_code(
            ctx: RunContext[RuntimeConfig],
            code: str
        ) -> dict[str, Any]:
            """Analyze Python code."""
            return await ctx.deps.execute_tool("analyze", code=code)

        @self.system_prompt
        async def get_language_prompt(
            ctx: RunContext[RuntimeConfig]
        ) -> str:
            """Dynamic system prompt."""
            langs = await ctx.deps.execute_tool("list_languages")
            return f"Supported languages: {', '.join(langs)}"
```

## Tools and System Prompts

Register tools and prompts with decorators:

```python
# Register a tool
@agent.tool
async def my_tool(
    ctx: RunContext[RuntimeConfig],
    arg: str
) -> str:
    """Tool description."""
    return f"Processed: {arg}"

# Register a plain tool (no context)
@agent.tool_plain
def plain_tool(text: str) -> str:
    """Plain tool without context."""
    return text.upper()

# Register dynamic system prompt
@agent.system_prompt
async def get_prompt(ctx: RunContext[RuntimeConfig]) -> str:
    """Dynamic system prompt."""
    resources = await ctx.deps.list_resources()
    return f"Available resources: {', '.join(resources)}"

# Register result validator
@agent.result_validator
async def validate(
    ctx: RunContext[RuntimeConfig],
    result: str
) -> str:
    """Validate result before returning."""
    if len(result) < 10:
        raise ModelRetry("Response too short")
    return result
```

## Event Handling

The agent can handle runtime events:

```python
class MyAgent(LLMlingAgent[str]):
    async def handle_event(self, event: Event) -> None:
        """Handle runtime events."""
        match event.type:
            case "RESOURCE_MODIFIED":
                print(f"Resource changed: {event.name}")
            case "TOOL_ADDED":
                print(f"New tool available: {event.name}")
```

## Best Practices

### Type Safety
- Use typed results when possible
- Validate tool inputs/outputs
- Handle model errors appropriately

### Resource Management
```python
# Proper cleanup with context manager
async with runtime as r:
    agent = LLMlingAgent(r)
    try:
        result = await agent.run("Query")
    except Exception as e:
        print(f"Agent error: {e}")
```

### Tool Design
- Keep tools focused
- Provide clear descriptions
- Use type hints
- Handle errors gracefully
- Report progress for long operations

### System Prompts
- Keep them clear and focused
- Use dynamic prompts when needed
- Don't leak implementation details
- Consider using templates

### Error Handling
```python
from llmling.core.exceptions import LLMError

try:
    result = await agent.run("Query")
except LLMError as e:
    print(f"Model error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

## Performance Tips

- Use streaming for long responses
- Cache expensive tool results
- Use appropriate timeouts
- Consider batching requests
- Profile tool performance

## Next Steps

For more information about agents and integration examples, see:
- [Agent Examples](https://llmling.readthedocs.io/en/latest/examples/agents/)
- [Tool Development](https://llmling.readthedocs.io/en/latest/tools/)
- [Type System](https://llmling.readthedocs.io/en/latest/types/)
