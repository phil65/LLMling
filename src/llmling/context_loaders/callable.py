"""Callable context loader."""

from __future__ import annotations

import asyncio
from importlib import import_module
import inspect
from typing import TYPE_CHECKING

from llmling.context_loaders.base import ContextLoader, LoadedContext, LoaderError
from llmling.processors import ProcessorRegistry, process_context


if TYPE_CHECKING:
    from llmling.config import CallableContext, Context


class CallableLoadError(LoaderError):
    """Raised when callable loading or execution fails."""


class CallableLoader(ContextLoader):
    """Loader for callable contexts."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load context by calling the specified function.

        Args:
            context: Configuration for callable-based context
            processor_registry: Registry containing available processors

        Returns:
            LoadedContext: Function output with metadata

        Raises:
            CallableLoadError: If loading or execution fails or context type is invalid
        """
        if context.type != "callable":
            msg = f"Expected callable context, got {context.type}"
            raise CallableLoadError(msg)

        # Now we know it's a CallableContext
        callable_context: CallableContext = context  # type: ignore

        try:
            # Import the callable
            module_path, func_name = callable_context.import_path.rsplit(".", 1)
            try:
                module = import_module(module_path)
            except ImportError as exc:
                msg = f"Failed to import module: {module_path}"
                raise CallableLoadError(msg) from exc

            try:
                func = getattr(module, func_name)
            except AttributeError as exc:
                msg = f"Function {func_name} not found in module {module_path}"
                raise CallableLoadError(msg) from exc

            # Call the function with kwargs
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(**callable_context.keyword_args)
                else:
                    result = func(**callable_context.keyword_args)
            except Exception as exc:
                msg = (
                    f"Error executing {callable_context.import_path} "
                    f"with args {callable_context.keyword_args}"
                )
                raise CallableLoadError(msg) from exc

            # Convert result to string if needed
            if not isinstance(result, str):
                result = str(result)

            # Process content through processors
            if callable_context.processors:
                result = await process_context(
                    result,
                    callable_context.processors,
                    processor_registry,
                )

            return LoadedContext(
                content=result,
                metadata={
                    "source": callable_context.import_path,
                    "kwargs": callable_context.keyword_args,
                    "is_coroutine": inspect.iscoroutinefunction(func),
                },
            )

        except Exception as exc:
            if not isinstance(exc, CallableLoadError):
                msg = f"Unexpected error with {callable_context.import_path}"
                raise CallableLoadError(msg) from exc
            raise


if __name__ == "__main__":
    from llmling.config import CallableContext

    async def example_function(name: str) -> str:
        return f"Hello, {name}!"

    async def main() -> None:
        # Create a proper CallableContext instance
        context = CallableContext(
            type="callable",
            import_path=f"{__name__}.example_function",
            description="Example callable",
            keyword_args={"name": "World"},
        )

        loader = CallableLoader()
        registry = ProcessorRegistry()

        try:
            result = await loader.load(context, registry)
            print(f"Content: {result.content}")
            print(f"Metadata: {result.metadata}")
        except CallableLoadError as exc:
            print(f"Error: {exc}")

    asyncio.run(main())
