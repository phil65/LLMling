"""Client for the llm library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling.core import capabilities, exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import llm
    from llm.models import AsyncModel


logger = get_logger(__name__)

# Cache for model instances and capabilities
_model_cache: dict[str, AsyncModel] = {}
_capabilities_cache: dict[str, capabilities.Capabilities] = {}


def _get_cached_model(model_id: str) -> AsyncModel:
    """Get or create cached async model instance."""
    import llm

    if model_id not in _model_cache:
        try:
            _model_cache[model_id] = llm.get_async_model(model_id)
        except llm.UnknownModelError as exc:
            msg = f"Model {model_id} not found"
            raise exceptions.LLMError(msg) from exc
    return _model_cache[model_id]


def _detect_model_capabilities(model: AsyncModel) -> capabilities.Capabilities:
    """Detect model capabilities from instance."""
    try:
        # Get supported options
        valid_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        }
        supported_params = [attr for attr in dir(model) if attr in valid_params]

        # Detect vision support from attachments
        supports_vision = hasattr(model, "handle_attachments")

        return capabilities.Capabilities(
            key=model.model_id,
            litellm_provider="llm",
            mode="chat",
            supports_system_messages=True,
            supports_vision=supports_vision,
            supported_openai_params=supported_params,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to detect capabilities: %s", exc)
        return capabilities.Capabilities(
            key=model.model_id,
            litellm_provider="llm",
            mode="chat",
        )


def get_model_info(model_id: str) -> capabilities.Capabilities:
    """Get model capabilities (cached)."""
    if model_id in _capabilities_cache:
        return _capabilities_cache[model_id]

    model = _get_cached_model(model_id)
    caps = _detect_model_capabilities(model)
    _capabilities_cache[model_id] = caps
    return caps


def _prepare_attachments(
    messages: list[dict[str, Any]],
) -> tuple[list[llm.Attachment], list[dict[str, Any]]]:
    """Extract attachments from messages."""
    import llm

    attachments = []
    clean_messages = []

    for msg in messages:
        if "content_items" in msg:
            msg_attachments = []
            text_content = []

            for item in msg["content_items"]:
                if item["type"] == "text":
                    text_content.append(item["content"])
                elif item["type"] == "image_url":
                    msg_attachments.append(llm.Attachment(url=item["content"]))
                elif item["type"] == "image_base64":
                    msg_attachments.append(
                        llm.Attachment(content=item["content"].encode())
                    )

            attachments.extend(msg_attachments)
            clean_messages.append({
                **msg,
                "content": "\n".join(text_content),
            })
        else:
            clean_messages.append(msg)

    return attachments, clean_messages


async def complete(
    model_id: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    """Execute completion."""
    try:
        model = _get_cached_model(model_id)

        # Extract system message if present
        system_prompt = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)

        # Handle attachments
        attachments, clean_messages = _prepare_attachments(user_messages)

        # Build prompt from remaining messages
        prompt = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in clean_messages
        )

        # Execute completion
        response = await model.prompt(
            prompt,
            system=system_prompt,
            attachments=attachments or None,
            **kwargs,
        )

        return {
            "choices": [{"message": {"content": await response.text()}}],
            "model": model_id,
            "usage": {
                "prompt_tokens": getattr(response, "prompt_tokens", 0),
                "completion_tokens": getattr(response, "completion_tokens", 0),
                "total_tokens": getattr(response, "total_tokens", 0),
            },
        }

    except Exception as exc:
        exc_msg = f"Completion failed: {exc}"
        raise exceptions.LLMError(exc_msg) from exc


async def stream(
    model_id: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Stream completions."""
    try:
        model = _get_cached_model(model_id)

        # Extract system message if present
        system_prompt = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)

        # Handle attachments
        attachments, clean_messages = _prepare_attachments(user_messages)

        # Build prompt
        prompt = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in clean_messages
        )

        # Stream response
        response = await model.prompt(
            prompt,
            system=system_prompt,
            attachments=attachments or None,
            **kwargs,
        )

        async for chunk in response:
            yield {
                "choices": [{"delta": {"content": chunk}}],
                "model": model_id,
            }

    except Exception as exc:
        error_msg = f"Streaming failed: {exc}"
        raise exceptions.LLMError(error_msg) from exc


if __name__ == "__main__":
    import asyncio

    import devtools
    import llm

    async def test_client():
        # List available models
        print("\nAvailable models:")
        for model in llm.get_async_models():
            print(f"- {model.model_id}")

        # Test with Claude
        model_id = "claude-3"
        print(f"\nTesting with {model_id}:")

        # Get capabilities
        info = get_model_info(model_id)
        print("\nCapabilities:")
        print(devtools.debug(info))

        # Test completion
        print("\nTesting completion:")
        response = await complete(
            model_id,
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Write a haiku about Python.",
                },
            ],
            temperature=0.7,
        )
        print(devtools.debug(response))

        # Test streaming
        print("\nTesting streaming:")
        async for chunk in stream(
            model_id,
            [{"role": "user", "content": "Count from 1 to 5 slowly."}],
            temperature=0.7,
        ):
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
        print()

        # Test vision (if supported)
        if info.supports_vision:
            print("\nTesting vision:")
            response = await complete(
                model_id,
                [
                    {
                        "role": "user",
                        "content": "What's in this image?",
                        "content_items": [
                            {
                                "type": "image_url",
                                "content": "https://example.com/image.jpg",
                            }
                        ],
                    }
                ],
            )
            print(devtools.debug(response))

    asyncio.run(test_client())
