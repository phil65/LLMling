from __future__ import annotations

import io

from PIL import Image

from llmling.context.models import ProcessingContext
from llmling.core.typedefs import Content, ContentType
from llmling.processors.base import ChainableProcessor, ProcessorResult


class ImageProcessor(ChainableProcessor):
    """Base class for image processors."""

    def _load_image(self, image_data: bytes) -> Image.Image:
        """Load bytes into PIL Image."""
        return Image.open(io.BytesIO(image_data))

    def _save_image(self, image: Image.Image, fmt: str = "PNG") -> bytes:
        """Save PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()


class ImageResizeProcessor(ImageProcessor):
    """Resize images to specified dimensions."""

    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        if context.current_content.type != ContentType.IMAGE:
            return ProcessorResult(
                content=context.current_content,
                original_content=context.original_content,
            )

        width = context.kwargs.get("width")
        height = context.kwargs.get("height")

        if not width and not height:
            return ProcessorResult(
                content=context.current_content,
                original_content=context.original_content,
            )

        img = self._load_image(context.current_content.data)
        resized = img.resize((width, height))
        meta = {**context.current_content.metadata, "size": resized.size}
        data = self._save_image(resized)
        new = Content(type=ContentType.IMAGE, data=data, metadata=meta)

        return ProcessorResult(content=new, original_content=context.original_content)


class ImageFormatProcessor(ImageProcessor):
    """Convert images between formats."""

    async def _process_impl(self, context: ProcessingContext) -> ProcessorResult:
        if context.current_content.type != ContentType.IMAGE:
            return ProcessorResult(
                content=context.current_content,
                original_content=context.original_content,
            )

        target_format = context.kwargs.get("format", "PNG")
        img = self._load_image(context.current_content.data)
        data = self._save_image(img, format=target_format)
        meta = {**context.current_content.metadata, "format": target_format}
        new = Content(type=ContentType.IMAGE, data=data, metadata=meta)
        return ProcessorResult(content=new, original_content=context.original_content)
