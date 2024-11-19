"""LiteLLM provider implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import litellm
from pydantic import BaseModel, ConfigDict, model_validator

from llmling.core import exceptions
from llmling.llm.base import CompletionResult, Message, RetryableProvider, ToolCall


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class LiteLLMFunction(BaseModel):
    """Function definition for LiteLLM tool calls."""

    name: str
    description: str
    parameters: dict[str, Any]

    model_config = ConfigDict(frozen=True)


class LiteLLMTool(BaseModel):
    """Tool definition for LiteLLM."""

    type: Literal["function"] = "function"
    function: LiteLLMFunction

    model_config = ConfigDict(frozen=True)


class LiteLLMMessage(BaseModel):
    """Message format for LiteLLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(frozen=True)


class LiteLLMRequest(BaseModel):
    """Complete request format for LiteLLM."""

    model: str
    messages: list[LiteLLMMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: int = 30
    stream: bool = False
    tools: list[LiteLLMTool] | None = None
    tool_choice: Literal["none", "auto"] | str | None = None  # noqa: PYI051

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_tools_and_choice(self) -> LiteLLMRequest:
        """Validate tool configuration."""
        if self.tool_choice and not self.tools:
            msg = "tool_choice provided but no tools defined"
            raise ValueError(msg)
        return self


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

            # Log the full configuration before the call
            logger.debug("=== LiteLLM Request Configuration ===")
            logger.debug("Model: %s", self.config.model)
            logger.debug("Messages: %s", messages_dict)
            logger.debug("Temperature: %s", self.config.temperature)
            logger.debug("Max tokens: %s", self.config.max_tokens)
            logger.debug("Raw kwargs: %s", kwargs)
            response = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
                **kwargs,
            )
            logger.debug("=== LiteLLM Response ===")
            logger.debug("Response object: %s", response)
            # Handle tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls"):
                tc = response.choices[0].message.tool_calls
                logger.debug("Received tool calls from LLM: %s", tc)
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
            error_msg = f"Failed to process LiteLLM response: {exc}"
            raise ValueError(error_msg) from exc

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
