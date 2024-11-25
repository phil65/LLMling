"""Image context loader implementation."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar

import upath

from llmling.config.models import ImageContext
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource

logger = get_logger(__name__)


class ImageResourceLoader(ResourceLoader[ImageContext]):
    """Loads image content from files or URLs."""

    context_class = ImageContext
    uri_scheme = "image"
    supported_mime_types: ClassVar[list[str]] = ["image/jpeg", "image/png", "image/gif"]

    @classmethod
    def get_uri_template(cls) -> str:
        """Image URIs follow the same pattern as file URIs."""
        return "image:///{name}"

    @classmethod
    def create_uri(cls, *, name: str) -> str:
        """Handle image paths properly."""
        normalized = name.replace("\\", "/").lstrip("/")
        return cls.get_uri_template().format(name=normalized)

    async def load(
        self,
        context: ImageContext,
        processor_registry: ProcessorRegistry,
    ) -> LoadedResource:
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
            path_obj = upath.UPath(context.path)
            is_url = path_obj.as_uri().startswith(("http://", "https://"))

            # Get image content and type
            if is_url:
                image_content = str(path_obj.as_uri())
                content_type = "image_url"
            else:
                if not path_obj.exists():
                    msg = f"Image file not found: {path_obj}"
                    raise exceptions.LoaderError(msg)  # noqa: TRY301

                with path_obj.open("rb") as f:
                    image_content = base64.b64encode(f.read()).decode()
                    content_type = "image_base64"

            # Create placeholder text for backwards compatibility
            placeholder_text = f"Image: {context.path}"
            if context.alt_text:
                placeholder_text = f"{placeholder_text} - {context.alt_text}"

            return create_loaded_resource(
                content=placeholder_text,
                source_type="image",
                uri=str(path_obj.as_uri()),
                mime_type=self._detect_mime_type(path_obj),
                name=path_obj.name,
                description=context.alt_text,
                additional_metadata={
                    "path": str(context.path),
                    "type": "url" if is_url else "local",
                    "alt_text": context.alt_text,
                },
                content_items=[
                    MessageContent(
                        type=content_type,
                        content=image_content,
                        alt_text=context.alt_text,
                    )
                ],
            )

        except Exception as exc:
            msg = f"Failed to load image from {context.path}"
            raise exceptions.LoaderError(msg) from exc

    def _detect_mime_type(self, path: upath.UPath) -> str:
        """Detect MIME type from file extension."""
        ext = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }.get(ext, "application/octet-stream")

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
