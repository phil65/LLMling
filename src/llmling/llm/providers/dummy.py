# llmling/llm/providers/dummy.py
from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.capabilities import Capabilities
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
    ToolCall,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class DummyProvider(LLMProvider):
    """Provider for testing that returns configured responses with optional tool calls."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with configuration.

        Config metadata format:
        metadata:
          delay: float = 0.1  # Delay between responses/chunks
          error_rate: float = 0.0  # Probability of raising an error
          chunk_size: int = 10  # Characters per chunk when streaming
          responses:  # Map of triggers to responses
            "What's 2+2?":  # Exact message match
              content: "Let me calculate that"
              tool_calls:  # Optional tool calls
                - name: "calculator"
                  parameters: {"operation": "add", "numbers": [2, 2]}
            "tool_result:calculator:4":  # Response after tool execution
              content: "The result is 4"
        """
        super().__init__(config)
        self.responses = self._parse_responses()
        self._capabilities = self._setup_capabilities()

    def _parse_responses(self) -> dict[str, CompletionResult]:
        """Parse response configurations into CompletionResults."""
        responses: dict[str, CompletionResult] = {}

        for trigger, value in self.config.metadata.get("responses", {}).items():
            if isinstance(value, str):
                # Simple string response
                responses[trigger] = CompletionResult(
                    content=value,
                    model=self.config.model,
                )
            else:
                # Response with optional tool calls
                tool_calls = None
                if tool_configs := value.get("tool_calls"):
                    tool_calls = [
                        ToolCall(
                            id=str(i),
                            name=tc["name"],
                            parameters=tc["parameters"],
                        )
                        for i, tc in enumerate(tool_configs)
                    ]

                responses[trigger] = CompletionResult(
                    content=value["content"],
                    model=self.config.model,
                    tool_calls=tool_calls,
                    metadata=value.get("metadata", {}),
                )
        return responses

    def _setup_capabilities(self) -> Capabilities:
        """Set up simulated provider capabilities."""
        return Capabilities(
            key=self.config.model,
            mode="chat",
            supports_system_messages=True,
            supports_function_calling=True,
            supports_vision=self.config.metadata.get("supports_vision", False),
            max_tokens=self.config.metadata.get("max_tokens", 4096),
        )

    def _get_response(self, messages: list[Message]) -> CompletionResult:
        """Get appropriate response based on message history."""
        if not messages:
            return self.responses.get(
                "default",
                CompletionResult(
                    content="No input provided",
                    model=self.config.model,
                ),
            )

        last_msg = messages[-1]

        # Check for tool result message
        if last_msg.role == "tool":
            trigger = f"tool_result:{last_msg.name}:{last_msg.content}"
            if response := self.responses.get(trigger):
                return response
            # Fallback to default tool response
            return CompletionResult(
                content=f"I got the result: {last_msg.content}",
                model=self.config.model,
            )

        # Normal message - exact match or default
        return self.responses.get(
            last_msg.content,
            self.responses.get(
                "default",
                CompletionResult(
                    content="I'm a dummy provider! Configure some responses for me.",
                    model=self.config.model,
                ),
            ),
        )

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Return configured response for the input messages."""
        # Maybe raise error based on error_rate
        if random.random() < self.config.metadata.get("error_rate", 0.0):
            msg = "Simulated error from dummy provider"
            raise exceptions.LLMError(msg)

        # Simulate processing delay
        await asyncio.sleep(self.config.metadata.get("delay", 0.1))

        return self._get_response(messages)

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Stream response in chunks."""
        response = self._get_response(messages)
        delay = self.config.metadata.get("delay", 0.1)
        chunk_size = self.config.metadata.get("chunk_size", 10)

        # Stream content in chunks
        for i in range(0, len(response.content), chunk_size):
            chunk = response.content[i : i + chunk_size]
            await asyncio.sleep(delay)
            yield CompletionResult(
                content=chunk,
                model=self.config.model,
                metadata={"chunk": True},
            )


if __name__ == "__main__":
    import asyncio

    import devtools

    async def demo():
        # Example configuration with tool usage
        config = LLMConfig(
            model="dummy/test",
            provider_name="dummy",
            metadata={
                "delay": 0.2,  # 200ms delay
                "responses": {
                    "default": {
                        "content": "I am a dummy provider!",
                    },
                    "What's 2+2?": {
                        "content": "Let me calculate that for you.",
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "parameters": {"operation": "add", "numbers": [2, 2]},
                            }
                        ],
                    },
                    "tool_result:calculator:4": {
                        "content": "The result is 4!",
                    },
                },
            },
        )

        provider = DummyProvider(config)

        # Test basic response
        print("\nBasic response:")
        result = await provider.complete([
            Message(role="user", content="Hello!"),
        ])
        print(devtools.debug(result))

        # Test tool usage flow
        print("\nTool usage flow:")
        result = await provider.complete([
            Message(role="user", content="What's 2+2?"),
        ])
        print(devtools.debug(result))

        # Tool result handling
        result = await provider.complete([
            Message(role="user", content="What's 2+2?"),
            Message(role="tool", content="4", name="calculator"),
        ])
        print(devtools.debug(result))

        # Test streaming
        print("\nStreaming response:")
        async for chunk in provider.complete_stream([
            Message(role="user", content="Hello!"),
        ]):
            print(chunk.content, end="", flush=True)
        print()

    asyncio.run(demo())
