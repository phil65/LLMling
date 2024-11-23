"""LLM library provider implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Unpack
import uuid

from llmling.core import capabilities, exceptions
from llmling.core.log import get_logger
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
    MessageContent,
    ToolCall,
)
from llmling.llm.clients import libllmclient


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.llm.clients.protocol import LiteLLMCompletionParams


logger = get_logger(__name__)


class LLMLibProvider(LLMProvider):
    """Provider implementation using the llm library."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with configuration."""
        super().__init__(config)
        self._capabilities = self._get_capabilities()
        logger.debug(
            "Initialized LLMLib provider for model %s with capabilities: %s",
            config.model,
            self._capabilities,
        )

    def _get_capabilities(self) -> capabilities.Capabilities:
        """Get model capabilities."""
        return libllmclient.get_model_info(self.config.model)

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to llm library format."""
        prepared = []
        for msg in messages:
            # Basic message data
            message_data: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }

            # Add content items if present (for multi-modal)
            if msg.content_items:
                message_data["content_items"] = [
                    {
                        "type": item.type,
                        "content": item.content,
                        "alt_text": item.alt_text,
                    }
                    for item in msg.content_items
                ]

            prepared.append(message_data)
        return prepared

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
    ) -> list[ToolCall] | None:
        """Extract tool calls from response."""
        try:
            if "function_call" not in response:
                return None

            function_call = response["function_call"]
            return [
                ToolCall(
                    id=str(uuid.uuid4()),
                    name=function_call["name"],
                    parameters=json.loads(function_call["arguments"]),
                )
            ]
        except Exception:  # noqa: BLE001
            logger.warning("Failed to process tool calls", exc_info=True)
            return None

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Unpack[LiteLLMCompletionParams],
    ) -> CompletionResult:
        """Generate a completion for the messages."""
        try:
            messages_dict = self._prepare_messages(messages)
            response = await libllmclient.complete(
                self.config.model,
                messages_dict,
                **kwargs,
            )

            # Extract usage information
            usage = response.get("usage", {})

            return CompletionResult(
                content=response["choices"][0]["message"]["content"],
                model=response["model"],
                tool_calls=self._extract_tool_calls(response),
                metadata={
                    "provider": "llm",
                    "usage": usage,
                },
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_cost=0.0,  # llm library doesn't provide cost info
            )

        except Exception as exc:
            msg = f"LLMLib completion failed: {exc}"
            raise exceptions.LLMError(msg) from exc

    async def complete_stream(
        self,
        messages: list[Message],
        *,
        chunk_size: int | None = None,
        **kwargs: Unpack[LiteLLMCompletionParams],
    ) -> AsyncIterator[CompletionResult]:
        """Generate a streaming completion for the messages."""
        try:
            messages_dict = self._prepare_messages(messages)

            async for chunk in libllmclient.stream(
                self.config.model,
                messages_dict,
                **kwargs,
            ):
                if content := chunk["choices"][0]["delta"].get("content"):
                    yield CompletionResult(
                        content=content,
                        model=chunk["model"],
                        metadata={"chunk": True},
                    )

        except Exception as exc:
            msg = f"LLMLib streaming failed: {exc}"
            raise exceptions.LLMError(msg) from exc


if __name__ == "__main__":
    import asyncio

    import devtools

    async def test_provider():
        """Test the LLMLib provider."""
        # Create test config
        config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
        )

        # Initialize provider
        provider = LLMLibProvider(config)
        print("\nProvider initialized with capabilities:")
        print(devtools.debug(provider._capabilities))

        # Test basic completion
        print("\nTesting basic completion:")
        result = await provider.complete([
            Message(
                role="system",
                content="You are a helpful assistant.",
            ),
            Message(
                role="user",
                content="Write a haiku about coding.",
            ),
        ])
        print(devtools.debug(result))

        # Test streaming
        print("\nTesting streaming:")
        messages = [
            Message(
                role="user",
                content="Count from 1 to 5 slowly.",
            ),
        ]
        async for chunk in provider.complete_stream(messages):
            print(chunk.content, end="", flush=True)
        print()

        # Test vision if supported
        if provider._capabilities.supports_vision:
            print("\nTesting vision capabilities:")
            vision_result = await provider.complete([
                Message(
                    role="user",
                    content="What's in this image?",
                    content_items=[
                        MessageContent(
                            type="image_url",
                            content="https://example.com/image.jpg",
                            alt_text="An example image",
                        ),
                    ],
                ),
            ])
            print(devtools.debug(vision_result))

    asyncio.run(test_provider())
