"""LiteLLM provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm

from llmling.core import exceptions
from llmling.llm.base import CompletionResult, RetryableProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.llm.base import Message


class LiteLLMProvider(RetryableProvider):
    """Provider implementation using LiteLLM."""

    async def _complete_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> CompletionResult:
        """Implement completion using LiteLLM."""
        try:
            response = await litellm.acompletion(
                model=self.config.model,
                messages=[msg.model_dump() for msg in messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout,
                **kwargs,
            )

            return CompletionResult(
                content=response.choices[0].message.content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "provider": "litellm",
                    "usage": response.usage.model_dump(),
                },
            )

        except Exception as exc:
            msg = f"LiteLLM completion failed: {exc}"
            raise exceptions.LLMError(msg) from exc

    async def _complete_stream_impl(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResult]:
        """Implement streaming completion using LiteLLM."""
        try:
            response_stream = await litellm.acompletion(
                model=self.config.model,
                messages=[msg.model_dump() for msg in messages],
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

                yield CompletionResult(
                    content=chunk.choices[0].delta.content,
                    model=chunk.model,
                    finish_reason=chunk.choices[0].finish_reason,
                    metadata={
                        "provider": "litellm",
                        "chunk": True,
                    },
                )

        except Exception as exc:
            msg = f"LiteLLM streaming failed: {exc}"
            raise exceptions.LLMError(msg) from exc
