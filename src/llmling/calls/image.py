"""Image processing API call implementation."""

import litellm
from pydantic import BaseModel, Field, HttpUrl

from jinjarope.llm.calls.base import APICall

class ImageParameters(BaseModel):
    resolution: str | None = None
    """Resolution of the image"""

    format: str | None = None
    """Format of the image (e.g., JPEG, PNG)"""

    quality: str | None = Field(default=None)
    """Quality of the image (standard/hd)"""

    color_mode: str | None = None
    """Color mode of the image (e.g., RGB, CMYK)"""

class ImageGenerationCall(APICall):
    image_url: HttpUrl
    """URL of the image to process"""

    parameters: ImageParameters
    """Image-specific parameters for this call"""

    def execute(self) -> dict:
        """Execute the image API call using litellm."""
        response = litellm.image_generation(
            prompt=str(self.image_url),
            size=self.parameters.resolution or "1024x1024",
            quality=self.parameters.quality or "standard",
            n=1,
        )
        return {
            "call_id": str(self.call_id),
            "name": self.name,
            "description": self.description,
            "image_urls": [img.url for img in response.data],
            "parameters": self.parameters.model_dump(),
            "context_sources": [source.model_dump() for source in self.context_sources],
        }

if __name__ == "__main__":
    # Create sample image parameters
    params = ImageParameters(
        resolution="1024x1024",
        format="PNG",
        quality="standard",
        color_mode="RGB"
    )

    # Create and execute image call
    image_call = ImageGenerationCall(
        name="Test Image Generation",
        description="Testing image generation capabilities",
        image_url="https://example.com/sample.jpg",
        parameters=params,
        context_sources=[]
    )

    # Execute and print results
    try:
        result = image_call.execute()
        print("Generated images:", result["image_urls"])
    except Exception as e:
        print(f"Error: {str(e)}")
