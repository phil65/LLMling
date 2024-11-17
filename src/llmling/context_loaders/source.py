"""Source code context loader."""

from __future__ import annotations

from importlib import import_module
import inspect
import pkgutil
from typing import TYPE_CHECKING

from llmling.context_loaders.base import ContextLoader, LoadedContext, LoaderError
from llmling.processors import ProcessorRegistry, process_context


if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from llmling.config import Context, SourceContext


class SourceLoadError(LoaderError):
    """Raised when source code loading fails."""


class SourceLoader(ContextLoader):
    """Loader for source code contexts."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load source code from module.

        Args:
            context: Configuration for source code context
            processor_registry: Registry containing available processors

        Returns:
            LoadedContext: Processed source code with metadata

        Raises:
            SourceLoadError: If loading or processing fails or context type is invalid
        """
        if context.type != "source":
            msg = f"Expected source context, got {context.type}"
            raise SourceLoadError(msg)

        # Now we know it's a SourceContext
        source_context: SourceContext = context  # type: ignore

        try:
            module = import_module(source_context.import_path)
            sources = list(
                self._get_sources(
                    module,
                    recursive=source_context.recursive,
                    include_tests=source_context.include_tests,
                )
            )
            content = "\n\n# " + "-" * 40 + "\n\n".join(sources)
            # Process content through processors
            if source_context.processors:
                content = await process_context(
                    content,
                    source_context.processors,
                    processor_registry,
                )

            return LoadedContext(
                content=content,
                metadata={
                    "module": source_context.import_path,
                    "recursive": source_context.recursive,
                    "include_tests": source_context.include_tests,
                },
            )

        except ImportError as exc:
            msg = f"Could not import module: {source_context.import_path}"
            raise SourceLoadError(msg) from exc

    def _get_sources(
        self,
        module: ModuleType,
        recursive: bool,
        include_tests: bool,
    ) -> Generator[str, None, None]:
        """Generate source code for a module and optionally its submodules."""
        if hasattr(module, "__file__") and module.__file__:
            path = module.__file__
            if self._should_include_file(path, include_tests):
                yield f"# File: {path}\n{inspect.getsource(module)}"

        if recursive and hasattr(module, "__path__"):
            for _, name, _ in pkgutil.iter_modules(module.__path__):
                submodule_path = f"{module.__name__}.{name}"
                try:
                    submodule = import_module(submodule_path)
                    yield from self._get_sources(
                        submodule,
                        recursive,
                        include_tests,
                    )
                except ImportError:
                    continue

    def _should_include_file(self, path: str, include_tests: bool) -> bool:
        """Check if a file should be included in the source."""
        if not include_tests:
            parts = path.split("/")
            if any(p.startswith("test") for p in parts):
                return False
        return path.endswith(".py")


if __name__ == "__main__":
    import asyncio

    from llmling.config import SourceContext
    from llmling.processors import ProcessorRegistry

    async def main() -> None:
        # Create a sample source context
        context = SourceContext(
            type="source",
            import_path="llmling.context_loaders.source",
            description="Source code loader example",
            recursive=True,
            include_tests=False,
        )

        # Initialize the loader and processor registry
        loader = SourceLoader()
        registry = ProcessorRegistry()
        # Load and display the source code
        result = await loader.load(context, registry)
        print(f"Loaded source from: {result.metadata['module']}")
        print(f"Recursive: {result.metadata['recursive']}")
        print(f"Include tests: {result.metadata['include_tests']}")
        print("\nSource code:")
        print("=" * 80)
        print(result.content)

    # Run the async example
    asyncio.run(main())
