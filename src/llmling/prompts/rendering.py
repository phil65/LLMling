from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from logfire import instrument

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.prompts.models import PromptMessage, PromptResult, ResolvedContent


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.prompts.models import Prompt
    from llmling.resources import ResourceLoaderRegistry

logger = get_logger(__name__)


@instrument("Resolving resources")
async def resolve_resources(
    message: PromptMessage,
    resource_registry: ResourceLoaderRegistry,
    processor_registry: ProcessorRegistry,
) -> PromptMessage:
    """Resolve resources in a message.

    Args:
        message: Message to resolve
        resource_registry: Registry for loading resources
        processor_registry: Registry for content processors

    Returns:
        Message with resolved resources

    Raises:
        ResourceResolutionError: If resource resolution fails
        ProcessorError: If content processing fails
    """
    if not message.needs_resolution():
        return message

    now = datetime.now()
    resolved: list[ResolvedContent] = []

    for content in message.get_content_items():
        if content.type != "resource":
            resolved.append(ResolvedContent(original=content))
            continue

        try:
            # Find appropriate loader
            loader = resource_registry.find_loader_for_uri(content.content)
            from llmling.resources.base import ResourceLoader

            if not isinstance(loader, ResourceLoader):
                msg = f"Invalid loader type for {content.content}"
                raise exceptions.ResourceResolutionError(msg)  # noqa: TRY301

            # Load resource
            try:
                resource = await loader.load(
                    loader.context,
                    processor_registry,
                )
            except Exception as exc:
                msg = f"Failed to load resource {content.content}: {exc}"
                raise exceptions.ResourceResolutionError(msg) from exc

            resolved.append(
                ResolvedContent(
                    original=content,
                    resolved=resource,
                    resolved_at=now,
                )
            )
        except exceptions.ResourceResolutionError:
            # Re-raise specific resource errors
            raise
        except Exception as exc:
            msg = f"Unexpected error resolving {content.content}: {exc}"
            raise exceptions.ResourceResolutionError(msg) from exc

    return PromptMessage(
        role=message.role,
        content=message.content,
        resolved_content=resolved,
    )


@instrument("Rendering prompt")
async def render_prompt(
    prompt: Prompt,
    arguments: dict[str, Any],
    *,
    resource_registry: ResourceLoaderRegistry | None = None,
    processor_registry: ProcessorRegistry | None = None,
) -> PromptResult:
    """Render a prompt template with arguments.

    Args:
        prompt: Prompt to render
        arguments: Arguments for template
        resource_registry: Optional registry for resolving resources
        processor_registry: Optional registry for content processors

    Returns:
        Rendered prompt result

    Raises:
        ProcessorError: If rendering fails
    """
    try:
        # Validate arguments first
        prompt.validate_arguments(arguments)

        rendered_messages: list[PromptMessage] = []
        now = datetime.now()

        for message in prompt.messages:
            # Handle string content
            if isinstance(message.content, str):
                msg = PromptMessage(
                    role=message.role, content=message.content.format(**arguments)
                )
                rendered_messages.append(msg)
                continue

            # Handle complex content
            contents = message.get_content_items()
            rendered_contents: list[MessageContent] = []

            for content in contents:
                if content.type == "text":
                    # Format text content
                    rendered_contents.append(
                        MessageContent.text(content.content.format(**arguments))
                    )
                else:
                    # Keep other content types as-is
                    rendered_contents.append(content)

            rendered_message = PromptMessage(
                role=message.role,
                content=rendered_contents,
            )

            # Resolve resources if registries provided
            if (
                resource_registry
                and processor_registry
                and rendered_message.needs_resolution()
            ):
                rendered_message = await resolve_resources(
                    rendered_message,
                    resource_registry,
                    processor_registry,
                )

            rendered_messages.append(rendered_message)

        return PromptResult(
            messages=rendered_messages,
            metadata={
                "prompt_name": prompt.name,
                "arguments": arguments,
            },
            resolved_at=now if resource_registry else None,
        )

    except KeyError as exc:
        error_msg = f"Missing argument in template: {exc}"
        raise exceptions.ProcessorError(error_msg) from exc
    except ValueError as exc:
        error_msg = f"Invalid arguments: {exc}"
        raise exceptions.ProcessorError(error_msg) from exc
    except Exception as exc:
        error_msg = f"Failed to render prompt: {exc}"
        raise exceptions.ProcessorError(error_msg) from exc
