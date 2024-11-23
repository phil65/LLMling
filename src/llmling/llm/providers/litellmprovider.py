"""LiteLLM provider implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Unpack

from pydantic import BaseModel, ConfigDict

from llmling.core import capabilities, exceptions
from llmling.core.log import get_logger
from llmling.llm.base import (
    CompletionResult,
    LLMConfig,
    LLMProvider,
    Message,
    MessageContent,
)
from llmling.llm.clients import litellmclient


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.llm.clients.protocol import LiteLLMCompletionParams


logger = get_logger(__name__)


class LiteLLMContent(BaseModel):
    """Content item in LiteLLM message format."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, str] | None = None

    model_config = ConfigDict(frozen=True)


class LiteLLMMessage(BaseModel):
    """Message in LiteLLM format."""

    role: str
    content: str | list[LiteLLMContent]
    name: str | None = None

    model_config = ConfigDict(frozen=True)


class LiteLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with configuration."""
        super().__init__(config)
        self._capabilities = self._get_capabilities()

    @property
    def model_info(self) -> capabilities.Capabilities:
        """Get model capabilities."""
        return self._capabilities

    def _get_capabilities(self) -> capabilities.Capabilities:
        """Get model capabilities."""
        try:
            info = litellmclient.get_model_info(self.config.model)
            return capabilities.Capabilities(**info)
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch info for model %s", self.config.model)
            return capabilities.Capabilities(
                key=self.config.model,
                supports_vision=False,
            )

    def _prepare_content(self, item: MessageContent) -> LiteLLMContent:
        """Convert a single content item to LiteLLM format."""
        match item.type:
            case "text":
                return LiteLLMContent(type="text", text=item.content)
            case "image_url":
                return LiteLLMContent(
                    type="image_url",
                    image_url={"url": item.content},
                )
            case "image_base64":
                url = f"data:image/jpeg;base64,{item.content}"
                return LiteLLMContent(type="image_url", image_url={"url": url})

    def _to_litellm_dict(self, msg: Message) -> dict[str, Any]:
        """Convert a message to LiteLLM dict format."""
        if not msg.content_items:
            return LiteLLMMessage(
                role=msg.role,
                content=msg.content,
                name=msg.name,
            ).model_dump()

        contents = [self._prepare_content(item) for item in msg.content_items]
        content = (
            contents[0].text or ""
            if len(contents) == 1 and contents[0].type == "text"
            else contents
        )
        return LiteLLMMessage(
            role=msg.role,
            content=content,
            name=msg.name,
        ).model_dump()

    async def complete(
        self,
        messages: list[Message],
        **kwargs: Unpack[LiteLLMCompletionParams],
    ) -> CompletionResult:
        """Generate a completion for the messages."""
        try:
            messages_dict = [self._to_litellm_dict(m) for m in messages]
            response = await litellmclient.complete(
                self.config.model,
                messages_dict,
                **kwargs,
            )

            usage = litellmclient.get_completion_cost(response)
            return CompletionResult(
                content=response.choices[0].message.content or "",
                model=response.model,
                tool_calls=self._process_tool_calls(response),
                metadata={
                    "provider": "litellm",
                    "usage": usage,
                },
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_cost=usage.get("total_cost"),
            )
        except Exception as exc:
            msg = f"LiteLLM completion failed: {exc}"
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
            messages_dict = [self._to_litellm_dict(m) for m in messages]
            stream_kwargs: dict[str, Any] = {**kwargs, "stream": True}
            if chunk_size is not None:
                stream_kwargs["chunk_size"] = chunk_size

            async for chunk in litellmclient.stream(
                self.config.model,
                messages_dict,
                **stream_kwargs,
            ):
                if content := chunk.choices[0].delta.content:
                    yield CompletionResult(
                        content=content,
                        model=chunk.model,
                        metadata={"chunk": True},
                    )
        except Exception as exc:
            msg = f"LiteLLM streaming failed: {exc}"
            raise exceptions.LLMError(msg) from exc

    def _process_tool_calls(self, response: Any) -> list[dict[str, Any]] | None:
        """Process tool calls from response."""
        if not hasattr(response.choices[0].message, "tool_calls"):
            return None

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return None

        processed = []
        for call in tool_calls:
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

            processed.append({
                "id": call.id,
                "name": call.function.name,
                "parameters": parameters,
            })
        return processed
