"""Context loader registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.context_loaders.base import ContextLoader, LoadedContext, LoaderError
from llmling.context_loaders.callable import CallableLoader
from llmling.context_loaders.cli import CLILoader
from llmling.context_loaders.path import PathLoader
from llmling.context_loaders.source import SourceLoader
from llmling.context_loaders.text import TextLoader


if TYPE_CHECKING:
    from llmling.config import Config, Context
    from llmling.processors import ProcessorRegistry


class LoaderRegistry:
    """Registry for context loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, ContextLoader] = {
            "callable": CallableLoader(),
            "cli": CLILoader(),
            "path": PathLoader(),
            "source": SourceLoader(),
            "text": TextLoader(),
        }

    def get_loader(self, context_type: str) -> ContextLoader:
        """Get appropriate loader for context type."""
        loader = self._loaders.get(context_type)
        if not loader:
            msg = f"No loader found for context type: {context_type}"
            raise LoaderError(msg)
        return loader

    async def load_context(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load context using appropriate loader."""
        loader = self.get_loader(context.type)
        return await loader.load(context, processor_registry)


# Convenience function
async def load_all_contexts(
    config: Config,
    processor_registry: ProcessorRegistry,
) -> dict[str, LoadedContext]:
    """Load all contexts from configuration."""
    loader_registry = LoaderRegistry()
    loaded_contexts = {}

    for name, context in config.contexts.items():
        loaded_contexts[name] = await loader_registry.load_context(
            context,
            processor_registry,
        )

    return loaded_contexts


if __name__ == "__main__":
    import asyncio

    from llmling.config import TextContext
    from llmling.processors import ProcessorRegistry

    async def main() -> None:
        # Create a sample text context
        context = TextContext(type="text", content="Hello!", description="Sample")
        # Initialize registry and load context
        registry = LoaderRegistry()
        processor_registry = ProcessorRegistry()

        try:
            result = await registry.load_context(context, processor_registry)
            print(f"Loaded content: {result.content}")
            print(f"Metadata: {result.metadata}")
        except LoaderError as e:
            print(f"Error loading context: {e}")

    asyncio.run(main())
