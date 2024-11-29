"""Template-based processor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jinja2
import logfire

from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import ChainableProcessor, ProcessorConfig, ProcessorResult


if TYPE_CHECKING:
    from llmling.resources.models import ProcessingContext


logger = get_logger(__name__)


class TemplateProcessor(ChainableProcessor):
    """Processor that applies a Jinja2 template."""

    def __init__(self, config: ProcessorConfig):
        """Initialize the processor."""
        super().__init__(config)
        self.template_config = config.get_template_config()
        self.template: jinja2.Template | None = None

    async def startup(self) -> None:
        """Compile template during startup."""
        try:
            loader = jinja2.BaseLoader()
            env = jinja2.Environment(loader=loader, autoescape=True, enable_async=True)
            self.template = env.from_string(self.config.template or "")
        except Exception as exc:
            msg = f"Failed to compile template: {exc}"
            raise exceptions.ProcessorError(msg) from exc

    @logfire.instrument("Rendering template")
    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        """Apply template to content."""
        if not self.template:
            msg = "Processor not initialized"
            raise exceptions.ProcessorError(msg)

        try:
            render_ctx = {"content": context.current_content, **context.kwargs}
            result = await self.template.render_async(**render_ctx)
            logger.debug("Template rendered: %s", result)
            keys = list(render_ctx.keys())
            meta = {"template_vars": keys, "template": self.config.template}
            orig = context.original_content
            return ProcessorResult(content=result, original_content=orig, metadata=meta)
        except Exception as exc:
            msg = f"Template rendering failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
