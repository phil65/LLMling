"""Base classes for content processors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmling.core import exceptions
from llmling.core.log import get_logger


PROCESSOR_TYPES = Literal["function", "template", "image"]


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling.resources.models import ProcessingContext


logger = get_logger(__name__)


class FunctionProcessorConfig(TypedDict, total=False):
    """Configuration specific to function processors."""

    import_path: str
    async_execution: bool


class TemplateProcessorConfig(TypedDict, total=False):
    """Configuration specific to template processors."""

    template: str
    template_engine: Literal["jinja2"]


class ProcessorConfig(BaseModel):
    """Configuration for text processors."""

    type: PROCESSOR_TYPES
    name: str | None = None
    description: str | None = None

    # Function processor fields
    import_path: str = ""  # Required for function type
    async_execution: bool = False

    # Template processor fields
    template: str = ""  # Required for template type
    template_engine: Literal["jinja2"] = "jinja2"

    # Validation settings
    validate_output: bool = False
    validate_schema: dict[str, Any] | None = None

    # Additional settings
    timeout: float | None = None
    cache_results: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    # MCP-specific fields
    supported_mime_types: list[str] = Field(default_factory=lambda: ["text/plain"])
    max_input_size: int | None = None
    streaming: bool = False

    @model_validator(mode="after")
    def validate_basic_structure(self) -> ProcessorConfig:
        """Validate basic structure only."""
        # Only validate structure, not content
        if self.type not in ("function", "template"):
            msg = f"Invalid processor type: {self.type}"
            raise ValueError(msg)
        return self

    def get_function_config(self) -> FunctionProcessorConfig:
        """Get function processor specific configuration."""
        if self.type != "function":
            msg = "Not a function processor configuration"
            raise ValueError(msg)
        return {
            "import_path": self.import_path,
            "async_execution": self.async_execution,
        }

    def get_template_config(self) -> TemplateProcessorConfig:
        """Get template processor specific configuration."""
        if self.type != "template":
            msg = "Not a template processor configuration"
            raise ValueError(msg)
        return {
            "template": self.template,
            "template_engine": self.template_engine,
        }


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

        if self.config.validate_schema:  # noqa: SIM102
            # Schema validation would go here
            if not isinstance(result.content, str):
                msg = f"Expected string output, got {type(result.content)}"
                raise exceptions.ProcessorError(msg)


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
        except Exception as exc:
            msg = f"Processing failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
        else:
            return final_result

    @abstractmethod
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Implement actual processing logic."""
