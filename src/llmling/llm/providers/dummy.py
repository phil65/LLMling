"""Enhanced dummy provider for development and testing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling.core import exceptions
from llmling.core.capabilities import Capabilities
from llmling.core.log import get_logger
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
    MessageContent,
    ToolCall,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = get_logger(__name__)


class DummyProvider(LLMProvider):
    """Configurable dummy provider for testing and user guidance."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with configuration.

        Config metadata format:
        metadata:
          delay: float = 0.1  # Delay between responses/chunks
          error_rate: float = 0.0  # Probability of raising an error
          chunk_size: int = 10  # Characters per chunk when streaming
          capabilities:  # Optional capabilities override
            supports_vision: bool = False
            supports_system_messages: bool = True
            max_tokens: int = 4096
          responses:  # Map of triggers to responses
            "exact match":  # String for exact message match
              content: "Simple response"
            {"has_image": true}:  # Condition dict for special cases
              content: "I see an image"
              metadata: {"type": "vision"}
            {"role": "tool"}:  # Match message role
              content: "Tool result received"
            {"has_tools": true}:  # Check for tool calls
              content: "Using tools"
              tool_calls:  # Optional tool calls
                - name: "calculator"
                  parameters: {"operation": "add", "numbers": [2, 2]}
        """
        super().__init__(config)
        self.responses = self._parse_responses()
        self._capabilities = self._setup_capabilities()
        self.pricing = self.config.metadata.get(
            "pricing",
            {
                "input_cost_per_token": 0.001,  # $0.001 per 1k input tokens
                "output_cost_per_token": 0.002,  # $0.002 per 1k output tokens
                "tokens_per_char": 0.25,  # Rough estimate of tokens per character
            },
        )

    def _calculate_usage(
        self,
        messages: list[Message],
        response: CompletionResult,
    ) -> CompletionResult:
        """Add token counting and cost tracking to response."""
        tokens_per_char = self.pricing["tokens_per_char"]

        # Calculate prompt tokens
        prompt_tokens = sum(int(len(m.content) * tokens_per_char) for m in messages)

        # Calculate completion tokens
        completion_tokens = int(len(response.content) * tokens_per_char)

        # Calculate costs
        input_cost = prompt_tokens * self.pricing["input_cost_per_token"]
        output_cost = completion_tokens * self.pricing["output_cost_per_token"]

        # Update response with usage information
        return CompletionResult(
            content=response.content,
            model=response.model,
            tool_calls=response.tool_calls,
            metadata={
                **response.metadata,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": input_cost + output_cost,
                },
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=input_cost + output_cost,
        )

    def _setup_capabilities(self) -> Capabilities:
        """Set up provider capabilities."""
        caps = self.config.metadata.get("capabilities", {})
        return Capabilities(
            key=self.config.model,
            mode="chat",
            supports_system_messages=caps.get("supports_system_messages", True),
            supports_function_calling=True,
            supports_vision=caps.get("supports_vision", False),
            max_tokens=caps.get("max_tokens", 4096),
            supported_openai_params=[
                "temperature",
                "max_tokens",
                "tools",
                "tool_choice",
            ],
        )

    def _parse_responses(self) -> dict[str, CompletionResult]:
        """Parse response configurations into CompletionResults."""
        responses: dict[str, CompletionResult] = {}

        for trigger, value in self.config.metadata.get("responses", {}).items():
            # Convert dict conditions to strings
            if isinstance(trigger, dict):
                # Convert condition dict to string key
                # e.g. {"has_image": true} -> "condition:has_image:true"
                trigger = self._condition_to_key(trigger)

            if isinstance(value, str):
                # Simple string response
                responses[trigger] = CompletionResult(
                    content=value,
                    model=self.config.model,
                )
                continue

            # Complex response with potential tool calls
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

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Return configured response with usage information."""
        if (
            error_rate := self.config.metadata.get("error_rate", 0.0)
        ) and random.random() < error_rate:
            msg = "Simulated error from dummy provider"
            raise exceptions.LLMError(msg)

        # Simulate processing delay
        if delay := self.config.metadata.get("delay", 0.1):
            await asyncio.sleep(delay)

        response = self._get_response(messages)
        return self._calculate_usage(messages, response)

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Stream response in chunks with usage information."""
        response = self._get_response(messages)
        response_with_usage = self._calculate_usage(messages, response)
        delay = self.config.metadata.get("delay", 0.1)
        chunk_size = self.config.metadata.get("chunk_size", 10)

        # Stream content in chunks
        for i in range(0, len(response.content), chunk_size):
            chunk = response.content[i : i + chunk_size]
            await asyncio.sleep(delay)
            # For streaming, we add usage info only to the last chunk
            is_last = i + chunk_size >= len(response.content)
            yield CompletionResult(
                content=chunk,
                model=self.config.model,
                metadata={
                    "chunk": True,
                    **(response_with_usage.metadata if is_last else {}),
                },
                prompt_tokens=response_with_usage.prompt_tokens if is_last else None,
                completion_tokens=response_with_usage.completion_tokens
                if is_last
                else None,
                total_cost=response_with_usage.total_cost if is_last else None,
            )

    def _condition_to_key(self, condition: dict[str, Any]) -> str:
        """Convert a condition dictionary to a string key."""
        # Sort keys for consistent ordering
        items = sorted(condition.items())
        return "condition:" + ":".join(f"{k}={v}" for k, v in items)

    def _get_response(self, messages: list[Message]) -> CompletionResult:
        """Get appropriate response based on triggers and conditions."""
        if not messages:
            return self._get_default_response()

        last_msg = messages[-1]

        # First check exact matches
        if last_msg.content in self.responses:
            return self.responses[last_msg.content]

        # Then check conditions
        for condition_key in self.responses:
            if condition_key.startswith("condition:") and self._matches_condition(
                condition_key, last_msg, messages
            ):
                return self.responses[condition_key]

        return self._get_default_response()

    def _matches_condition(
        self,
        condition_key: str,
        message: Message,
        history: list[Message],
    ) -> bool:
        """Check if a message matches a condition."""
        if not condition_key.startswith("condition:"):
            return False

        # Parse condition from key
        # condition:has_image=true -> {"has_image": true}
        parts = condition_key.split(":", 1)[1].split("=")
        if len(parts) != 2:  # noqa: PLR2004
            return False

        key, value = parts
        match (key, value):
            case ("has_image", "true"):
                return any(
                    item.type in ("image_url", "image_base64")
                    for item in message.content_items
                )
            case ("has_tools", "true"):
                return bool(message.tool_calls)
            case ("role", role):
                return message.role == role
        return False

    def _get_default_response(self) -> CompletionResult:
        """Get default response from config."""
        return self.responses.get(
            "default",
            CompletionResult(
                content="I am a dummy provider! Configure some responses for me.",
                model=self.config.model,
            ),
        )


if __name__ == "__main__":
    import random

    import devtools

    async def demo():
        # Example configuration
        config = LLMConfig(
            model="dummy/test",
            provider_name="dummy",
            metadata={
                "delay": 0.2,  # 200ms delay
                "error_rate": 0.1,  # 10% chance of error
                "capabilities": {
                    "supports_vision": True,
                },
                "responses": {
                    "Hello!": "Hi there!",  # Simple response
                    "condition:has_image=true": {  # Vision response
                        "content": "I see an image!",
                        "metadata": {"type": "vision"},
                    },
                    "calculate": {  # Tool usage
                        "content": "Let me calculate that.",
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "parameters": {"operation": "add", "numbers": [2, 2]},
                            }
                        ],
                    },
                    "condition:role=tool": {  # Tool result handling
                        "content": "The result is: {content}",
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

        # Test vision
        print("\nVision response:")
        result = await provider.complete([
            Message(
                role="user",
                content="What's in this image?",
                content_items=[
                    MessageContent(
                        type="image_url", content="http://example.com/img.jpg"
                    ),
                ],
            ),
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
