"""LiteLLM provider implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import litellm
from pydantic import BaseModel, ConfigDict

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
    ToolCall,
)


logger = get_logger(__name__)

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

    type: str = "function"
    function: LiteLLMFunction

    model_config = ConfigDict(frozen=True)


class LiteLLMProvider(LLMProvider):
    """Provider implementation using LiteLLM."""

    # Default capabilities by provider
    DEFAULT_CAPABILITIES: ClassVar = {
        "ollama": {
            "supports_function_calling": False,
            "supported_openai_params": ["temperature", "max_tokens", "top_p", "stream"],
            "supports_system_messages": True,
        },
        "openai": {
            "supports_function_calling": True,
            "supported_openai_params": [
                "tools",
                "tool_choice",
                "temperature",
                "max_tokens",
                "top_p",
                "stream",
            ],
            "supports_system_messages": True,
        },
    }

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with capability checking."""
        super().__init__(config)
        self.model_info = self._get_model_capabilities()
        # Preserve important settings that should always be passed
        self._base_settings = {
            k: v
            for k, v in config.model_dump(exclude_unset=True, exclude_none=True).items()
            if k in {"api_base", "api_key"} and v is not None
        }

    def _get_model_name_without_provider(self) -> str:
        """Extract model name without provider prefix."""
        try:
            return self.config.model.split("/")[1]
        except IndexError:
            return self.config.model

    def _get_provider_from_model(self) -> str:
        """Extract provider name from model string."""
        try:
            return self.config.model.split("/")[0]
        except Exception:  # noqa: BLE001
            return "unknown"

    def _get_model_capabilities(self) -> dict[str, Any]:
        """Get model capabilities with fallback to defaults."""
        provider = self._get_provider_from_model()
        model_name = self._get_model_name_without_provider()

        try:
            return litellm.get_model_info(model_name)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Using default capabilities for %s model %s", provider, self.config.model
            )
            return self.DEFAULT_CAPABILITIES.get(
                provider,
                {
                    "supports_function_calling": False,
                    "supported_openai_params": ["temperature", "max_tokens", "stream"],
                    "supports_system_messages": True,
                },
            )

    def _prepare_request_kwargs(self, **additional_kwargs: Any) -> dict[str, Any]:
        """Prepare request kwargs from config and additional kwargs."""
        # Start with essential settings preserved from initialization
        kwargs = self._base_settings.copy()

        # Get all fields that were explicitly set
        config_dict = self.config.model_dump(exclude_unset=True)

        # Fields we don't want to pass to litellm
        exclude_fields = {
            "provider_name",
            "display_name",
            "streaming",
            "model",  # Exclude model as it's passed explicitly
        }

        # Add all config values except excluded fields and None values
        kwargs.update({
            k: v
            for k, v in config_dict.items()
            if k not in exclude_fields and v is not None
        })
        # Add additional kwargs (highest priority)
        # Filter out empty tools array
        filtered_kwargs = {
            k: v
            for k, v in additional_kwargs.items()
            if v is not None and not (k == "tools" and not v)
        }
        kwargs.update(filtered_kwargs)
        return kwargs

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement completion using LiteLLM."""
        try:
            # Convert messages to dict format
            messages_dict = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                }
                for msg in messages
            ]

            # Clean up kwargs
            # Remove empty tools array and related settings
            if "tools" in kwargs and not kwargs["tools"]:
                kwargs.pop("tools")
                kwargs.pop("tool_choice", None)

            # Prepare request kwargs
            request_kwargs = self._prepare_request_kwargs(**kwargs)

            # Execute completion
            response = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                **request_kwargs,
            )

            return self._process_response(response)

        except Exception as exc:
            msg = f"LiteLLM completion failed: {exc}"
            raise exceptions.LLMError(msg) from exc

    async def complete_stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Implement streaming completion using LiteLLM."""
        try:
            # Convert messages to dict format
            messages_dict = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                }
                for msg in messages
            ]

            # Clean up kwargs
            # Remove empty tools array and related settings
            if "tools" in kwargs and not kwargs["tools"]:
                kwargs.pop("tools")
                kwargs.pop("tool_choice", None)

            # Remove tool-related kwargs if model doesn't support them
            if not self.model_info.get("supports_function_calling", False):
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)

            # Prepare kwargs with streaming enabled
            request_kwargs = self._prepare_request_kwargs(stream=True, **kwargs)

            # Execute streaming completion
            stream = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                **request_kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
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

    def _process_response(self, response: Any) -> CompletionResult:
        """Process LiteLLM response into CompletionResult."""
        try:
            # Handle tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls"):
                tc = response.choices[0].message.tool_calls
                logger.debug("Received tool calls from LLM: %s", tc)
                if tc:
                    tool_calls = []
                    for call in tc:
                        try:
                            parameters = (
                                json.loads(call.function.arguments)
                                if isinstance(call.function.arguments, str)
                                else call.function.arguments
                            )
                        except json.JSONDecodeError:
                            logger.exception(
                                "Failed to parse tool parameters: %s",
                                call.function.arguments,
                            )
                            parameters = {}

                        tool_calls.append(
                            ToolCall(
                                id=call.id,
                                name=call.function.name,
                                parameters=parameters,
                            )
                        )

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
            msg = f"Failed to process LiteLLM response: {exc}"
            raise exceptions.LLMError(msg) from exc
