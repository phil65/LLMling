"""Text context loader implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling.context_loaders.base import ContextLoader, LoadedContext
from llmling.processors import ProcessorError, ProcessorRegistry, process_context


if TYPE_CHECKING:
    from llmling.config import Context, TextContext


class TextLoadError(Exception):
    """Raised when text content loading or processing fails."""


class TextLoader(ContextLoader):
    """Loader for text contexts with processor support."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load and process content from configured text.

        Args:
            context: Configuration for text-based context
            processor_registry: Registry containing available processors

        Returns:
            LoadedContext: Processed content with metadata

        Raises:
            TextLoadError: If processing fails or context type is invalid
        """
        # Runtime type check for text context
        if context.type != "text":
            msg = f"Expected text context, got {context.type}"
            raise TextLoadError(msg)

        # Now we know it's a TextContext
        text_context: TextContext = context  # type: ignore

        content = text_context.content

        if text_context.processors:
            try:
                content = await process_context(
                    content=content,
                    steps=text_context.processors,
                    registry=processor_registry,
                    parallel=False,  # Sequential processing for text is safer
                )
            except ProcessorError as exc:
                msg = "Failed to process text content"
                raise TextLoadError(msg) from exc

        return LoadedContext(
            content=content,
            metadata={"type": "raw_text"},
        )


async def _main() -> None:
    """Example usage of TextLoader."""
    from llmling.config import TextContext
    from llmling.processors import ProcessingStep, ProcessorConfig

    # Create sample text context with processors
    context = TextContext(
        type="text",
        content="Hello, this is a sample text!",
        description="Sample text for testing",
        processors=[
            ProcessingStep(name="uppercase"),
            ProcessingStep(name="wrap"),
        ],
    )

    # Initialize loader and processor registry
    loader = TextLoader()
    registry = ProcessorRegistry()

    # Register some processors
    registry.register(
        "uppercase",
        ProcessorConfig(type="function", import_path="reprlib.repr"),
    )
    registry.register(
        "wrap",
        ProcessorConfig(type="template", template="[{{ content }}]"),
    )

    try:
        # Load and process the context
        result = await loader.load(context, registry)
        print(f"Loaded content: {result.content}")
        print(f"Metadata: {result.metadata}")
    except TextLoadError as exc:
        print(f"Failed to load text: {exc}")


if __name__ == "__main__":
    asyncio.run(_main())
