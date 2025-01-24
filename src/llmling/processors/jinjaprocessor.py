"""Builtin Jinja2 processor.."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.models import Jinja2Config
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.processors.base import BaseProcessor, ProcessorResult


if TYPE_CHECKING:
    from llmling.resources.models import ProcessingContext


logger = get_logger(__name__)


class Jinja2Processor(BaseProcessor):
    """Processor that renders content using Jinja2."""

    def __init__(self, jinja_settings: Jinja2Config | None = None) -> None:
        import jinja2

        super().__init__()
        settings = jinja_settings or Jinja2Config()
        jinja_kwargs = settings.create_environment_kwargs()
        jinja_filters = jinja_kwargs.pop("filters", {})
        jinja_globals = jinja_kwargs.pop("globals", {})
        jinja_tests = jinja_kwargs.pop("tests", {})
        self._env = jinja2.Environment(**jinja_kwargs, enable_async=True)
        self._env.filters = jinja_filters
        self._env.globals = jinja_globals
        self._env.tests = jinja_tests
        self._initialized = True

    async def process(self, context: ProcessingContext) -> ProcessorResult:
        try:
            template = self._env.from_string(context.current_content)
            content = await template.render_async(**context.kwargs)
            return ProcessorResult(
                content=content,
                original_content=context.original_content,
                metadata={"template_vars": list(context.kwargs.keys())},
            )
        except Exception as exc:
            msg = f"Template rendering failed: {exc}"
            raise exceptions.ProcessorError(msg) from exc
