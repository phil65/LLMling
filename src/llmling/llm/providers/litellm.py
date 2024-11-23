"""LiteLLM provider implementation."""

from __future__ import annotations

from datetime import timedelta
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from diskcache import Cache
import litellm
import logfire

from llmling.core import capabilities, exceptions
from llmling.core.log import get_logger
from llmling.llm.base import CompletionResult, LLMConfig, LLMProvider, Message, ToolCall


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import py2openai

logger = get_logger(__name__)

# Initialize capability cache with 1 day TTL
_cache = Cache(".model_cache")
_CACHE_TTL = timedelta(days=1).total_seconds()


class LiteLLMProvider(LLMProvider):
    """Provider implementation using LiteLLM."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with capability checking."""
        super().__init__(config)
        self.model_info = get_model_capabilities(self.config.model)
        # Preserve important settings that should always be passed
        self._base_settings = {
            k: v
            for k, v in config.model_dump(exclude_unset=True, exclude_none=True).items()
            if k in {"api_base", "api_key"} and v is not None
        }

    def _prepare_messages(self, msg: Message) -> str | list[dict[str, Any]]:
        """Prepare message content for LiteLLM."""
        if not msg.content_items:
            return msg.content

        content: list[dict[str, Any]] = []
        for item in msg.content_items:
            match item.type:
                case "text":
                    content.append({"type": "text", "text": item.content})
                case "image_url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": item.content},
                    })
                case "image_base64":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{item.content}"},
                    })

        # For better compatibility, if only text and single item, return just the text
        if len(content) == 1 and content[0]["type"] == "text":
            return content[0]["text"]

        return content

    def _check_vision_support(self, messages: list[Message]) -> None:
        """Check if model supports vision when image content is present."""
        types = ("image_url", "image_base64")
        has_images = any(i.type in types for msg in messages for i in msg.content_items)
        if has_images and not self.model_info.supports_vision:
            msg = f"Model {self.config.model} does not support vision inputs"
            raise exceptions.LLMError(msg)

    def _prepare_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare request kwargs from config and runtime parameters."""
        final_kwargs = self._base_settings.copy()

        # Add config values, excluding model and other special fields
        exclude_fields = {
            "model",  # Will be passed directly
            "provider_name",
            "display_name",
            "streaming",
        }

        config_dict = self.config.model_dump(
            exclude=exclude_fields,
            exclude_none=True,
            exclude_unset=True,  # Exclude fields not explicitly set
        )
        final_kwargs.update(config_dict)

        # Add runtime parameters
        final_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Final request kwargs: %s", json.dumps(final_kwargs, indent=2))

        return final_kwargs

    @logfire.instrument("LiteLLM completion")
    async def complete(
        self,
        messages: list[Message],
        *,  # Force keyword arguments
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: int | None = None,
        tools: list[py2openai.OpenAIFunctionTool] | None = None,
        tool_choice: Literal["none", "auto"] | str | None = None,  # noqa: PYI051
        max_image_size: int | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        num_retries: int | None = None,
        request_timeout: float | None = None,
        cache: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Implement completion using LiteLLM."""
        try:
            self._check_vision_support(messages)
            messages_list = [
                {
                    "role": msg.role,
                    **({"name": msg.name} if msg.name else {}),
                    "content": self._prepare_messages(msg),
                }
                for msg in messages
            ]

            # Convert parameters to a dict, filtering out None values
            request_kwargs = self._prepare_kwargs(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                max_image_size=max_image_size,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                num_retries=num_retries,
                request_timeout=request_timeout,
                cache=cache,
                metadata=metadata,
            )
            response = await litellm.acompletion(
                model=self.config.model,
                messages=messages_list,
                **request_kwargs,
            )

            return self._process_response(response)

        except Exception as exc:
            msg = f"LiteLLM completion failed: {exc}"
            raise exceptions.LLMError(msg) from exc

    async def complete_stream(
        self,
        messages: list[Message],
        *,  # Force keyword arguments
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: int | None = None,
        tools: list[py2openai.OpenAIFunctionTool] | None = None,
        tool_choice: Literal["none", "auto"] | str | None = None,  # noqa: PYI051
        max_image_size: int | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        num_retries: int | None = None,
        request_timeout: float | None = None,
        cache: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[CompletionResult]:
        """Stream response chunks."""
        try:
            self._check_vision_support(messages)
            messages_list = [
                {
                    "role": msg.role,
                    **({"name": msg.name} if msg.name else {}),
                    "content": self._prepare_messages(msg),
                }
                for msg in messages
            ]

            # Explicitly collect non-None parameters
            request_kwargs = self._prepare_kwargs(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                max_image_size=max_image_size,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                num_retries=num_retries,
                request_timeout=request_timeout,
                cache=cache,
                metadata=metadata,
                stream=True,  # Always True for streaming
            )
            stream = await litellm.acompletion(
                model=self.config.model,
                messages=messages_list,
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
        tool_calls = None
        try:
            # Handle tool calls if present
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


def get_model_capabilities(
    model: str,
    provider: str | None = None,
) -> capabilities.Capabilities:
    """Get model capabilities from LiteLLM (caches because no idea if IO is involved)."""
    # Construct cache key
    cache_key = f"{provider}/{model}" if provider else model

    # Try to get from cache
    try:
        if cached := _cache.get(cache_key):
            return capabilities.Capabilities.model_validate(cached)
    except Exception:  # noqa: BLE001
        # Handle potential cache corruption
        _cache.delete(cache_key)

    # Not in cache or cache error, fetch fresh
    try:
        model_name = f"{provider}/{model}" if provider else model
        info = litellm.get_model_info(model_name)
        caps = capabilities.Capabilities(**info)

        # Cache the dict representation
        _cache.set(cache_key, caps.model_dump(), expire=_CACHE_TTL)
    except Exception:  # noqa: BLE001
        # If we can't get info, return minimal capabilities
        logger.warning("Could not fetch info for model %s", model)
        return capabilities.Capabilities(
            key=model,
            litellm_provider=provider,
        )
    else:
        return caps


def clear_capabilities_cache() -> None:
    """Clear the disk cache of model capabilities."""
    _cache.clear()


if __name__ == "__main__":
    import devtools

    info = get_model_capabilities("openai/gpt-3.5-turbo")
    devtools.debug(info)
