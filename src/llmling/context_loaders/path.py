"""Path context loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from upath import UPath

from llmling.config import PathContext
from llmling.context_loaders.base import ContextLoader, LoadedContext, LoaderError
from llmling.processors import ProcessorRegistry, process_context


if TYPE_CHECKING:
    from llmling.config import Context


class PathLoadError(LoaderError):
    """Raised when path content loading or processing fails."""


class PathLoader(ContextLoader):
    """Loader for path contexts."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load context from file or URL.

        Args:
            context: Configuration for path-based context
            processor_registry: Registry containing available processors

        Returns:
            LoadedContext: Processed content with metadata

        Raises:
            PathLoadError: If loading fails or context type is invalid
        """
        if context.type != "path":
            msg = f"Expected path context, got {context.type}"
            raise PathLoadError(msg)

        # Now we know it's a PathContext
        path_context: PathContext = context  # type: ignore

        path = UPath(path_context.path)
        if not path.is_file():
            msg = f"Path {path_context.path} is not a file"
            raise PathLoadError(msg)
        try:
            content = path.read_text()
            source_type = "file"
            # Process content through processors
            if path_context.processors:
                content = await process_context(
                    content,
                    path_context.processors,
                    processor_registry,
                )

            return LoadedContext(
                content=content,
                metadata={
                    "source_type": source_type,
                    "path": str(path_context.path),
                },
            )

        except Exception as exc:
            msg = f"Failed to load content from {path_context.path}"
            raise PathLoadError(msg) from exc


if __name__ == "__main__":
    import asyncio
    from typing import NoReturn

    async def demo() -> NoReturn:
        # Create a sample processor registry
        processor_registry = ProcessorRegistry()
        # Example with a local file
        file_context = PathContext(
            type="path",
            path="README.md",
            description="Docs",
        )
        # Example with a URL
        url = (
            "https://raw.githubusercontent.com/python/peps/"
            "refs/heads/main/peps/pep-0008.rst"
        )
        url_context = PathContext(
            type="path",
            path=url,
            description="PEP 8 style guide",
        )
        # Initialize the loader
        loader = PathLoader()

        try:
            # Load and display file content
            print("Loading file using UPath...")
            file_result = await loader.load(file_context, processor_registry)
            print(f"File metadata: {file_result.metadata}")
            print(f"First 100 chars: {file_result.content[:100]}\n")

            # Load and display URL content
            print("Loading URL content...")
            url_result = await loader.load(url_context, processor_registry)
            print(f"URL metadata: {url_result.metadata}")
            print(f"First 100 chars: {url_result.content[:100]}")

        except Exception as exc:
            print(f"Error during demonstration: {exc}")
            raise

    # Run the async demo
    asyncio.run(demo())
