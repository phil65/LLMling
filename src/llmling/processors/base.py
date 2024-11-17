"""Base classes for content processors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core import exceptions
from llmling.core.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.context.models import ProcessingContext


logger = get_logger(__name__)


class ProcessorConfig(BaseModel):
    """Configuration for text processors."""

    type: Literal["function", "template"]
    name: str | None = None  # Make it optional initially
    description: str | None = None

    # Function processor fields
    import_path: str | None = None
    async_execution: bool = False

    # Template processor fields
    template: str | None = None
    template_engine: Literal["jinja2"] = "jinja2"

    # Validation settings
    validate_output: bool = False
    validate_schema: dict[str, Any] | None = None

    # Additional settings
    timeout: float | None = None
    cache_results: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> ProcessorConfig:
        """Validate processor configuration."""
        match self.type:
            case "function":
                if not self.import_path:
                    msg = "import_path is required for function processors"
                    raise ValueError(msg)
                self.name = self.name or self.import_path.split(".")[-1]
            case "template":
                if not self.template:
                    msg = "template is required for template processors"
                    raise ValueError(msg)
                self.name = self.name or "template_processor"
            case _:
                msg = f"Invalid processor type: {self.type}"
                raise ValueError(msg)
        return self


class ProcessorResult(BaseModel):
    """Result of processing content."""

    content: str
    original_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class BaseProcessor:
    """Base class for all processors."""

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize processor with configuration."""
        self.config = config
        self._initialized = False

    async def startup(self) -> None:
        """Perform any necessary initialization."""
        self._initialized = True

    @abstractmethod
    async def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process content with given context.

        Args:
            context: Processing context

        Returns:
            Processing result

        Raises:
            ProcessorError: If processing fails
        """

    async def shutdown(self) -> None:
        """Perform any necessary cleanup."""

    async def validate_result(self, result: ProcessorResult) -> None:
        """Validate processing result."""
        if not self.config.validate_output:
            return

        if not result.content:
            msg = "Processor returned empty content"
            raise exceptions.ProcessorError(msg)

        if self.config.validate_schema:
            try:
                # Schema validation would go here
                if not isinstance(result.content, str):
                    msg = f"Expected string output, got {type(result.content)}"
                    raise exceptions.ProcessorError(msg)
            except Exception as exc:
                msg = "Output validation failed"
                raise exceptions.ProcessorError(msg) from exc


class AsyncProcessor(BaseProcessor):
    """Base class for asynchronous processors."""

    async def process_stream(
        self,
        context: ProcessingContext,
    ) -> AsyncIterator[ProcessorResult]:
        """Process content in streaming mode."""
        result = await self.process(context)
        yield result


class ChainableProcessor(AsyncProcessor):
    """Processor that can be chained with others."""

    async def pre_process(self, context: ProcessingContext) -> ProcessingContext:
        """Prepare context for processing."""
        return context

    async def post_process(
        self,
        context: ProcessingContext,
        result: ProcessorResult,
    ) -> ProcessorResult:
        """Modify result after processing."""
        return result

    async def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process content with pre and post processing."""
        try:
            prepared_context = await self.pre_process(context)
            result = await self._process_impl(prepared_context)
            final_result = await self.post_process(prepared_context, result)
            await self.validate_result(final_result)
            return final_result
        except Exception as exc:
            msg = f"Processing failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc

    @abstractmethod
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Implement actual processing logic."""
