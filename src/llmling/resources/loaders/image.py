"""Image context loader implementation."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar

import upath

from llmling.config.models import ImageResource
from llmling.core import exceptions
from llmling.core.log import get_logger
from llmling.core.typedefs import MessageContent
from llmling.resources.base import ResourceLoader, create_loaded_resource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    from llmling.processors.registry import ProcessorRegistry
    from llmling.resources.models import LoadedResource

logger = get_logger(__name__)


class ImageResourceLoader(ResourceLoader[ImageResource]):
    """Loads image content from files or URLs."""

    context_class = ImageResource
    uri_scheme = "image"
    supported_mime_types: ClassVar[list[str]] = ["image/jpeg", "image/png", "image/gif"]

    @classmethod
    def get_uri_template(cls) -> str:
        """Image URIs follow the same pattern as file URIs."""
        return "image:///{name}"

    def create_uri(self, *, name: str) -> str:
        """Handle image paths properly."""
        normalized = name.replace("\\", "/").lstrip("/")
        return self.get_uri_template().format(name=normalized)

    @classmethod
    def get_name_from_uri(cls, uri: str) -> str:
        """Handle image URIs which might be URLs."""
        try:
            if not cls.supports_uri(uri):
                msg = f"Unsupported URI scheme: {uri}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            if uri.startswith(("http://", "https://")):
                # For URLs, use the full URL as name
                return uri
            # For local files, use normalized path
            return super().get_name_from_uri(uri)

        except Exception as exc:
            msg = f"Invalid image URI: {uri}"
            raise exceptions.LoaderError(msg) from exc

    async def _load_impl(
        self,
        resource: ImageResource,
        name: str,
        processor_registry: ProcessorRegistry | None,
    ) -> AsyncIterator[LoadedResource]:
        """Load and process image content."""
        try:
            path_obj = upath.UPath(resource.path)
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
            placeholder_text = f"Image: {resource.path}"
            if resource.alt_text:
                placeholder_text = f"{placeholder_text} - {resource.alt_text}"

            yield create_loaded_resource(
                content=placeholder_text,
                source_type="image",
                uri=self.create_uri(name=name),
                mime_type=self._detect_mime_type(path_obj),
                name=path_obj.name,
                description=resource.alt_text,
                additional_metadata={
                    "path": str(resource.path),
                    "type": "url" if is_url else "local",
                    "alt_text": resource.alt_text,
                },
                content_items=[
                    MessageContent(
                        type=content_type,
                        content=image_content,
                        alt_text=resource.alt_text,
                    )
                ],
            )
        except Exception as exc:
            msg = f"Failed to load image from {resource.path}"
            raise exceptions.LoaderError(msg) from exc

    def _detect_mime_type(self, path: str | os.PathLike[str]) -> str:
        """Detect MIME type from file extension."""
        ext = upath.UPath(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }.get(ext, "application/octet-stream")

    async def _load_content(self, path_obj: str | os.PathLike[str], is_url: bool) -> str:
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
            return upath.UPath(path_obj).as_uri()

        try:
            if not upath.UPath(path_obj).exists():
                msg = f"Image file not found: {path_obj}"
                raise exceptions.LoaderError(msg)  # noqa: TRY301

            with upath.UPath(path_obj).open("rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as exc:
            if isinstance(exc, exceptions.LoaderError):
                raise
            msg = f"Failed to read image file: {path_obj}"
            raise exceptions.LoaderError(msg) from exc
