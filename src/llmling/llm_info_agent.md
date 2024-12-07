# Understanding LLMling Usage in Agent Package

This guide helps you understand how our LLMling library is used to implement AI agents in the package "llmling_agent". It demonstrates how to wrap LLMling's functionality for LLM interactions.

## Core Integration: RuntimeConfig Usage

The Agent uses LLMling's RuntimeConfig as its backend, making resources, tools, and processors available to LLMs through function calling:

```python
class LLMLingAgent[TResult]:
    """LLM agent powered by LLMling's RuntimeConfig."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        result_type: type[TResult] | None = None,
        *,
        model: str = "gpt-4",
        system_prompt: str | Sequence[str] = (),
    ) -> None:
        """Store runtime and set up LLM interaction."""
        self._runtime = runtime
        self.pydantic_agent = PydanticAgent(
            result_type=result_type or str,
            model=model,
            system_prompt=system_prompt,
        )
        # Make LLMling functionality available to LLM
        self._setup_default_tools()

    async def run(
        self,
        prompt: str,
        *,
        message_history: list[Message] | None = None,
    ) -> RunResult[TResult]:
        """Run LLM interaction with structured output."""
```

## LLMling Tools as LLM Functions

The agent exposes LLMling's functionality through function calling:

```python
def _setup_default_tools(self) -> None:
    """Register LLMling operations as LLM-callable tools."""

    @self.tool
    async def load_resource(
        ctx: RunContext[RuntimeConfig],
        uri: str,
    ) -> LoadedResource:
        """Let LLM load resources by URI or name."""
        return await ctx.deps.load_resource_by_uri(uri)

    @self.tool
    async def process_content(
        ctx: RunContext[RuntimeConfig],
        content: str,
        processor_name: str,
        **kwargs: Any,
    ) -> str:
        """Let LLM use content processors."""
        result = await ctx.deps.process_content(
            content, processor_name, **kwargs
        )
        return result.content

    @self.tool
    async def list_resource_names(
        ctx: RunContext[RuntimeConfig]
    ) -> Sequence[str]:
        """Let LLM discover available resources."""
        return ctx.deps.list_resource_names()
```

## Usage Pattern

The agent shows how to use RuntimeConfig for LLM interactions:

```python
# Define structured output
class Analysis(BaseModel):
    summary: str
    complexity: int

# Use LLMling with LLM
async with RuntimeConfig.open("config.yml") as runtime:
    # Create agent with structured output
    agent = LLMLingAgent[Analysis](runtime)

    # Add custom tool using LLMling
    @agent.tool
    async def analyze_code(
        ctx: RunContext[RuntimeConfig],
        file: str,
    ) -> str:
        # Load and process through RuntimeConfig
        resource = await ctx.deps.load_resource(file)
        result = await ctx.deps.process_content(
            resource.content,
            processor_name="code_analyzer",
        )
        return result.content

    # Run LLM interaction
    result = await agent.run(
        "Analyze the code in main.py"
    )
    print(f"Summary: {result.data.summary}")
```

## Dynamic System Prompts

The agent can use RuntimeConfig state for dynamic prompts:

```python
@agent.system_prompt
async def get_context(ctx: RunContext[RuntimeConfig]) -> str:
    """Create system prompt from available resources."""
    resources = await ctx.deps.list_resource_names()
    return f"You can access these resources: {', '.join(resources)}"
```

## Key Integration Points

The agent package shows several important LLMling usage patterns:

1. RuntimeConfig as LLM Backend
   - Resources/tools available through function calling
   - Clean separation between LLM and functionality
   - Type-safe interactions via Pydantic models

2. Resource Access
   - Loading through tools
   - Content processing
   - Resource discovery

3. Tool Integration
   - Automatic registration of RuntimeConfig tools
   - Type-safe parameter passing
   - Result validation

4. System Prompts
   - Dynamic generation from RuntimeConfig state
   - Resource-aware context
   - Runtime updates

## Error Handling

The agent shows how to handle LLMling errors in LLM context:

```python
@self.tool
async def safe_load_resource(
    ctx: RunContext[RuntimeConfig],
    uri: str,
) -> str:
    try:
        resource = await ctx.deps.load_resource(uri)
        return resource.content
    except ResourceError as exc:
        # Convert to LLM-friendly message
        return f"Error loading resource: {exc}"
```

The llmling_agent package demonstrates how LLMling can be used as a backend for LLM interactions, showing:
1. How to wrap RuntimeConfig for LLMs
2. How to provide structured interactions
3. How to expose functionality through tools
4. How to handle async operations
5. How to manage LLM context

When analyzing this integration, note how it maintains a clean separation between LLM interaction (PydanticAI) and core functionality (LLMling).
