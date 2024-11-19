"""LiteLLM provider implementation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import litellm
from pydantic import BaseModel, ConfigDict, model_validator

from llmling.core import exceptions
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    Message,
    RetryableProvider,
    ToolCall,
)


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
        if (
            self.tool_choice
            and self.tool_choice not in ("none", "auto")
            and not self.tools
        ):
            msg = "tool_choice provided but no tools defined"
            raise ValueError(msg)
        return self


class LiteLLMProvider(RetryableProvider):
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
        # Get model capabilities on initialization
        self.model_info = self._get_model_capabilities()
        logger.debug(
            "Model %s capabilities: supports_tools=%s, supported_params=%s",
            self.config.model,
            self.model_info.get("supports_function_calling"),
            self.model_info.get("supported_openai_params"),
        )

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
            # Try getting official info first
            return litellm.get_model_info(model_name)
        except Exception:  # noqa: BLE001
            # Don't log a warning - this is expected for new/custom models
            logger.debug(
                "Using default capabilities for %s model %s", provider, self.config.model
            )
            return self.DEFAULT_CAPABILITIES.get(
                provider,
                {
                    # Conservative defaults if provider unknown
                    "supports_function_calling": False,
                    "supported_openai_params": ["temperature", "max_tokens", "stream"],
                    "supports_system_messages": True,
                },
            )

    def _prepare_request(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LiteLLMRequest:
        """Prepare request based on model capabilities."""
        try:
            messages_litellm = [
                LiteLLMMessage(
                    role=msg.role,
                    content=msg.content,
                    name=msg.name,
                ).model_dump()
                for msg in messages
            ]

            # Filter kwargs based on supported parameters
            supported_params = self.model_info.get("supported_openai_params", [])
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

            # Handle tools specifically
            supports_tools = (
                self.model_info.get("supports_function_calling", False)
                or "tools" in supported_params
            )

            tools_litellm = None
            if supports_tools and "tools" in kwargs:
                tools_litellm = [
                    LiteLLMTool(
                        type="function",
                        function=LiteLLMFunction(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=tool["parameters"],
                        ),
                    ).model_dump()
                    for tool in kwargs["tools"]
                ]

            # Create request with supported parameters
            request = LiteLLMRequest(
                model=self.config.model,
                messages=messages_litellm,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
                tools=tools_litellm if supports_tools else None,
                tool_choice=filtered_kwargs.get("tool_choice")
                if supports_tools
                else None,
            )
        except Exception as exc:
            error_msg = f"Failed to prepare LiteLLM request: {exc}"
            raise ValueError(error_msg) from exc
        else:
            return request

    async def _complete_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement completion using LiteLLM."""
        try:
            # Format request
            request = self._prepare_request(messages, **kwargs)

            try:
                # Only retry the actual API call
                response = await litellm.acompletion(
                    **request.model_dump(exclude_none=True)
                )
            except (
                litellm.APIError,
                litellm.APIConnectionError,
            ) as exc:
                # These are retryable errors
                msg = f"LiteLLM API error: {exc}"
                raise exceptions.LLMError(msg) from exc
            except Exception as exc:
                # Other errors should not be retried
                msg = f"LiteLLM completion failed: {exc}"
                raise exceptions.TaskError(msg) from exc

            return self._process_response(response)

        except exceptions.LLMError:
            # Re-raise LLM errors for retry
            raise
        except Exception as exc:
            # Convert other errors to TaskError
            msg = f"LiteLLM completion failed: {exc}"
            raise exceptions.TaskError(msg) from exc

    def _process_response(self, response: Any) -> CompletionResult:
        """Process LiteLLM response into CompletionResult.

        Args:
            response: Raw response from LiteLLM

        Returns:
            Processed completion result

        Raises:
            ValueError: If response processing fails
        """
        try:
            # Handle tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls"):
                tc = response.choices[0].message.tool_calls
                logger.debug("Received tool calls from LLM: %s", tc)
                if tc:
                    tool_calls = []
                    for call in tc:
                        # Parse the arguments string into a dictionary
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
