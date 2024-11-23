"""Simple LiteLLM client."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from diskcache import Cache

from llmling.core import exceptions


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Unpack

    from llmling.llm.clients.protocol import LiteLLMCompletionParams


# Initialize cache with 1 day TTL for API responses
_cache = Cache(".model_cache")
_CACHE_TTL = timedelta(days=1).total_seconds()


def _ensure_vision_support(model: str, messages: list[dict[str, Any]]) -> None:
    """Check if model supports vision when image content present."""
    import litellm

    # Check if any message has image content
    has_images = any(
        isinstance(msg.get("content"), list)
        and any(item.get("type") == "image_url" for item in msg["content"])
        for msg in messages
    )
    if not has_images:
        return

    # Get model info and check vision support
    info = litellm.get_model_info(model)
    if not info.get("supports_vision"):
        msg = f"Model {model} does not support vision inputs"
        raise exceptions.LLMError(msg)


def get_model_info(model: str) -> dict[str, Any]:
    """Get model capabilities (cached)."""
    try:
        if cached := _cache.get(f"info_{model}"):
            return cached
    except Exception:  # noqa: BLE001
        _cache.delete(f"info_{model}")

    import litellm

    info = litellm.get_model_info(model)
    _cache.set(f"info_{model}", info, expire=_CACHE_TTL)
    return dict(info)


async def complete(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Unpack[LiteLLMCompletionParams],
) -> Any:
    """Execute completion."""
    import litellm

    _ensure_vision_support(model, messages)
    return await litellm.acompletion(model=model, messages=messages, **kwargs)


async def stream(
    model: str,
    messages: list[dict[str, Any]],
    chunk_size: int | None = None,
    **kwargs: Unpack[LiteLLMCompletionParams],
) -> AsyncIterator[Any]:
    """Stream completions."""
    import litellm

    _ensure_vision_support(model, messages)
    request_kwargs: dict[str, Any] = {**kwargs, "stream": True}
    if chunk_size is not None:
        request_kwargs["chunk_size"] = chunk_size

    async for chunk in await litellm.acompletion(
        model=model,
        messages=messages,
        **request_kwargs,
    ):
        yield chunk
