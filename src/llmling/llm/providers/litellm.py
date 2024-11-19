"""LiteLLM provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm

from llmling.core import exceptions
from llmling.llm.base import CompletionResult, Message, RetryableProvider, ToolCall


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class LiteLLMProvider(RetryableProvider):
    """Provider implementation using LiteLLM."""

    async def _complete_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement completion using LiteLLM."""
        try:
            # Convert messages to dict format
            messages_dict = []
            for msg in messages:
                msg_dict: dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content,
                }
                if msg.name:
                    msg_dict["name"] = msg.name
                if msg.tool_calls:
                    msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
                messages_dict.append(msg_dict)

            # Tools are now already in the correct format from TaskExecutor
            response = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
                **kwargs,
            )

            # Handle tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls"):
                tc = response.choices[0].message.tool_calls
                if tc:
                    tool_calls = [
                        ToolCall(
                            id=call.id,
                            name=call.function.name,
                            parameters=call.function.arguments,
                        )
                        for call in tc
                    ]

            return CompletionResult(
                content=response.choices[0].message.content or "",
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                tool_calls=tool_calls,
                metadata={
                    "provider": "litellm",
                    "usage": response.usage.model_dump(),
                },
            )

        except Exception as exc:
            error_msg = f"LiteLLM completion failed: {exc}"
            raise exceptions.LLMError(error_msg) from exc

    def _is_local_provider(self) -> bool:
        """Check if the current model is a local provider (like Ollama)."""
        return self.config.model.startswith(("ollama/", "local/"))

    async def _complete_stream_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Implement streaming completion using LiteLLM."""
        try:
            # Convert messages to dict format, same as above
            messages_dict = []
            for msg in messages:
                msg_dict: dict[str, Any] = {
                    "role": msg.role,
                    "content": msg.content,
                }
                if msg.name:
                    msg_dict["name"] = msg.name
                if msg.tool_calls:
                    msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
                messages_dict.append(msg_dict)

            # Add tool configuration if present and provider supports it
            if self.config.tools and not self._is_local_provider():
                kwargs["tools"] = self.config.tools
                if self.config.tool_choice is not None:
                    kwargs["tool_choice"] = self.config.tool_choice

            response_stream = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
                stream=True,
                **kwargs,
            )

            async for chunk in response_stream:
                if not chunk.choices[0].delta.content:
                    continue

                # Tool calls aren't supported in streaming mode yet
                yield CompletionResult(
                    content=chunk.choices[0].delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason,
                    metadata={
                        "provider": "litellm",
                        "chunk": True,
                    },
                )

        except Exception as e:
            error_msg = f"LiteLLM streaming failed: {e}"
            raise exceptions.LLMError(error_msg) from e
