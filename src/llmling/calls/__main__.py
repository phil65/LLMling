from typing import List
from .image import ImageGenerationCall, ImageParameters
from pydantic import HttpUrl

def main() -> List[str]:
    """Example of using ImageGenerationCall for image processing."""
    try:
        # Create image parameters
        params = ImageParameters(
            resolution="1024x1024",
            format="PNG",
            quality="standard",
            color_mode="RGB"
        )

        # Create image call instance
        image_call = ImageGenerationCall(
            name="Sample Image Processing",
            description="Demonstrating image processing capabilities",
            image_url=HttpUrl("https://example.com/sample-image.jpg"),
            parameters=params,
            context_sources=[]
        )

        # Execute the call
        result = image_call.execute()
        return result["image_urls"]

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []

if __name__ == "__main__":
    image_urls = main()
    print("Generated image URLs:", image_urls)
