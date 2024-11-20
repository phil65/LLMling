from pathlib import Path
from typing import Any

from PIL import Image
from upath import UPath

from llmling.context.base import ContextLoader
from llmling.context.models import LoadedContext
from llmling.core.exceptions import LoaderError
from llmling.core.typedefs import Content, ContentType


class VisionPathLoader(ContextLoader):
    """Loads images from files or URLs."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".avif"}

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load image content from path."""
        if not isinstance(context, PathContext):
            msg = f"Expected PathContext, got {type(context).__name__}"
            raise LoaderError(msg)

        try:
            path = UPath(context.path)
            if not self._is_supported_format(path):
                msg = f"Unsupported image format: {path.suffix}"
                raise LoaderError(msg)

            # Load and validate image
            image_data = await self._load_image(path)
            image_meta = await self._get_image_metadata(image_data)

            content = Content(
                type=ContentType.IMAGE, data=image_data, metadata=image_meta
            )

            # Process if needed
            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="vision_path",
                metadata={"path": str(path), "format": path.suffix.lower(), **image_meta},
            )

        except Exception as exc:
            msg = f"Failed to load image from {context.path}: {exc}"
            raise LoaderError(msg) from exc

    def _is_supported_format(self, path: Path | UPath) -> bool:
        """Check if file format is supported."""
        return path.suffix.lower() in self.SUPPORTED_FORMATS

    async def _load_image(self, path: Path | UPath) -> bytes:
        """Load image data from path or URL."""
        if isinstance(path, UPath) and path.protocol != "file":
            # Handle remote URLs
            async with aiohttp.ClientSession() as session:
                async with session.get(str(path)) as response:
                    response.raise_for_status()
                    return await response.read()

        # Local file
        return path.read_bytes()

    async def _get_image_metadata(self, image_data: bytes) -> dict[str, Any]:
        """Extract image metadata."""
        with Image.open(io.BytesIO(image_data)) as img:
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "has_exif": hasattr(img, "_getexif") and img._getexif() is not None,
            }


class VisionCallableLoader(ContextLoader):
    """Loads images from callable execution."""

    async def load(
        self,
        context: Context,
        processor_registry: ProcessorRegistry,
    ) -> LoadedContext:
        """Load image from callable execution."""
        if not isinstance(context, CallableContext):
            msg = f"Expected CallableContext, got {type(context).__name__}"
            raise LoaderError(msg)

        try:
            # Execute callable
            result = await calling.execute_callable(
                context.import_path, **context.keyword_args
            )

            # Validate and process result
            if isinstance(result, (str, Path)):
                # Load from path
                path = UPath(str(result))
                loader = VisionPathLoader()
                return await loader.load(PathContext(path=str(path)), processor_registry)
            if isinstance(result, bytes):
                # Direct bytes
                image_meta = await self._get_image_metadata(result)
                content = Content(
                    type=ContentType.IMAGE, data=result, metadata=image_meta
                )
            elif isinstance(result, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                result.save(buffer, format=result.format or "PNG")
                content = Content(
                    type=ContentType.IMAGE,
                    data=buffer.getvalue(),
                    metadata={
                        "size": result.size,
                        "mode": result.mode,
                        "format": result.format,
                    },
                )
            else:
                msg = f"Unsupported image result type: {type(result)}"
                raise LoaderError(msg)

            # Process if needed
            if procs := context.processors:
                processed = await processor_registry.process(content, procs)
                content = processed.content

            return LoadedContext(
                content=content,
                source_type="vision_callable",
                metadata={"import_path": context.import_path, **content.metadata},
            )

        except Exception as exc:
            msg = f"Failed to load image from callable {context.import_path}: {exc}"
            raise LoaderError(msg) from exc
