# Understanding LLMling Usage in MCP Server

This guide helps you understand how our LLMling library is used to implement a MCP (Model Control Protocol) server in the package "mcp_server_llmling". It serves as a real-world example of LLMling integration.

## Core Integration: RuntimeConfig Usage

The MCP server uses LLMling's RuntimeConfig as its backend, translating network protocol calls into LLMling operations:

```python
class LLMLingServer:
    """MCP server backed by LLMling's RuntimeConfig."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        transport: Literal["stdio", "sse"] = "stdio",
    ) -> None:
        self.runtime = runtime
        self._subscriptions: defaultdict[str, set[Session]] = defaultdict(set)

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start serving MCP protocol requests."""

    # How MCP protocol maps to RuntimeConfig
    async def handle_list_resources(self) -> list[Resource]:
        """When client requests resource list."""
        names = self.runtime.list_resource_names()
        return [
            Resource(
                uri=self.runtime.get_resource_uri(name),
                name=name,
                description=self.runtime.get_resource(name).description,
            )
            for name in names
        ]

    async def handle_read_resource(self, uri: str) -> str:
        """When client requests resource content."""
        resource = await self.runtime.load_resource_by_uri(uri)
        return resource.content

    async def handle_call_tool(self, name: str, args: dict) -> Any:
        """When client wants to execute a tool."""
        return await self.runtime.execute_tool(name, **args)

    async def handle_get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """When client needs a formatted prompt."""
        return await self.runtime.render_prompt(name, arguments)
```

## Resource Change Notifications

The server listens to LLMling's resource events to notify clients of changes:

```python
class ResourceObserver:
    """Bridges LLMling events to MCP notifications."""

    def __init__(self, server: LLMLingServer) -> None:
        self.server = server
        # Register for LLMling events
        server.runtime.add_observer(self.events, "resource")

    def _handle_resource_modified(self, key: str, resource: Resource) -> None:
        """When LLMling reports resource change."""
        uri = self.server.runtime.get_resource_uri(key)
        # Notify subscribed MCP clients
        for session in self.server._subscriptions[uri]:
            session.send_resource_updated(uri)
```

## Usage Pattern

The server demonstrates how to use RuntimeConfig as a backend service:

```python
# Server startup using LLMling
async def main() -> None:
    # 1. Create RuntimeConfig from configuration
    async with RuntimeConfig.open("config.yml") as runtime:
        # 2. Create MCP server using RuntimeConfig
        server = LLMLingServer(runtime)

        # 3. Start serving - now RuntimeConfig handles all requests
        await server.start()
```

## Key Integration Points

The MCP server shows several important LLMling usage patterns:

1. RuntimeConfig as Service Backend
   - All functionality comes from RuntimeConfig
   - Clean separation between protocol and functionality
   - Async-first interaction

2. Resource Management
   - Loading via name or URI
   - Content access and modification
   - Change notification system

3. Tool Execution
   - Direct forwarding of tool calls
   - Parameter passing
   - Result handling

4. Prompt Handling
   - Prompt rendering with arguments
   - Access to prompt registry
   - Completion support

## Error Handling

The server shows how to handle LLMling errors in integrations:

```python
async def handle_read_resource(self, uri: str) -> str:
    try:
        return await self.runtime.load_resource_by_uri(uri)
    except ResourceError as exc:
        # Convert LLMling errors to protocol errors
        raise McpError(
            error=ErrorData(code=INVALID_PARAMS, message=str(exc))
        ) from exc
```

The MCP server package demonstrates how LLMling can be used as a backend for network protocols, showing:
1. How to wrap RuntimeConfig in a service
2. How to translate between systems
3. How to handle events and notifications
4. How to manage async operations
5. How to handle errors

When analyzing this integration, note how it maintains a clean separation between protocol handling (MCP) and core functionality (LLMling).
