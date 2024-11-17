"""Template-based processor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jinja2
import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import ChainableProcessor, ProcessorResult


if TYPE_CHECKING:
    from llmling.processors.base import ProcessingContext


logger = get_logger(__name__)


class TemplateProcessor(ChainableProcessor):
    """Processor that applies a Jinja2 template."""

    def __init__(self, config: Any) -> None:
        """Initialize the processor.

        Args:
            config: Processor configuration including the template
        """
        super().__init__(config)
        self.template: jinja2.Template | None = None

    async def startup(self) -> None:
        """Compile template during startup."""
        try:
            env = jinja2.Environment(
                loader=jinja2.BaseLoader(),
                autoescape=True,
                enable_async=True,
            )
            self.template = env.from_string(self.config.template)
        except Exception as exc:
            msg = f"Failed to compile template: {exc}"
            raise exceptions.ProcessorError(msg) from exc

    @logfire.instrument("Rendering template with {len(context.kwargs)} variables")
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Apply template to content."""
        if not self.template:
            msg = "Processor not initialized"
            raise exceptions.ProcessorError(msg)

        try:
            render_context = {
                "content": context.current_content,
                **context.kwargs,
            }

            result = await self.template.render_async(**render_context)
            print(f"Template rendered: {result}")  # Debug print

            return ProcessorResult(
                content=result,
                original_content=context.original_content,
                metadata={
                    "template_vars": list(render_context.keys()),
                    "template": self.config.template,  # Add this for debugging
                },
            )
        except Exception as exc:
            msg = f"Template rendering failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
