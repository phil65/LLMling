"""Image context loader implementation."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import upath

from llmling.config.models import ImageContext
from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry

logger = get_logger(__name__)


class ImageContextLoader(ContextLoader[ImageContext]):
    """Loads image content from files or URLs."""

    context_class = ImageContext

    async def load(
        self,
        context: ImageContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load and process image content.

        Args:
            context: Image context configuration
            processor_registry: Registry of available processors

        Returns:
            Loaded and processed context

        Raises:
            LoaderError: If loading fails or context type is invalid
        """
        try:
            # Use UPath to handle the path
            path_obj = upath.UPath(context.path)
            is_url = path_obj.as_uri().startswith(("http://", "https://"))

            content_item = MessageContent(
                type="image_url" if is_url else "image_base64",
                content=await self._load_content(path_obj, is_url),
                alt_text=context.alt_text,
            )

            return LoadedContext(
                content="",  # Keep empty for backward compatibility
                content_items=[content_item],
                source_type="image",
                metadata={
                    "path": context.path,
                    "type": "url" if is_url else "local",
                    "alt_text": context.alt_text,
                },
            )

        except Exception as exc:
            msg = f"Failed to load image from {context.path}"
            raise exceptions.LoaderError(msg) from exc

    async def _load_content(self, path_obj: upath.UPath, is_url: bool) -> str:
        """Load content from path.

        Args:
            path_obj: UPath object representing the path
            is_url: Whether the path is a URL

        Returns:
            URL or base64-encoded content

        Raises:
            LoaderError: If loading fails
        """
        if is_url:
            return path_obj.as_uri()

        try:
            if not path_obj.exists():
                msg = f"Image file not found: {path_obj}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            with path_obj.open("rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
            msg = f"Failed to read image file: {path_obj}"
            raise exceptions.LoaderError(msg) from exc
