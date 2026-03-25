"""
OpenAI Image Generation for COG-JEPA

Uses DALL-E to generate images from video scenes.
"""

import logging
import os

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images using DALL-E API."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Set API key from environment or use provided key
        api_key = os.environ.get(
            "OPENAI_API_KEY", "sk-poe-aaL_6wZBF3IoyUsuCEuZzjPjREwk6TYZ6GvdTaoj2PY"
        )

        try:
            self.client = OpenAI(api_key=api_key)
            self.api_key = api_key

            # Test connection
            self.client.models.list()
            self.connected = True
            logger.info("OpenAI API connected for image generation")

        except Exception as e:
            logger.warning(f"OpenAI API not connected: {e}")
            self.connected = False

    def generate_image(self, prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image from text prompt.

        Args:
            prompt: Description of image to generate
            size: Image size (1024x1024, 1024x1792, or 1792x1024)

        Returns:
            URL of generated image or error message
        """
        if not self.connected:
            return "❌ OpenAI API not connected. Please check your API key."

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            return image_url

        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return f"❌ Error generating image: {str(e)}"

    def generate_from_video_context(self, events: list, frame_idx: int = None) -> str:
        """Generate image based on video events."""
        if not events:
            return "❌ No video events to generate image from."

        # Get a representative event
        if frame_idx is not None:
            event = next(
                (e for e in events if e.get("frame_index") == frame_idx), events[-1]
            )
        else:
            # Get highest surprise event
            event = max(events, key=lambda x: x.get("surprise_score", 0))

        desc = event.get("description", "action scene")

        # Build detailed prompt
        prompt = f"""A cinematic movie scene: {desc}. 
Professional cinematography, dramatic lighting, high quality film still, 4k resolution, photorealistic."""

        return self.generate_image(prompt)


def get_image_generator() -> ImageGenerator:
    """Get or create image generator."""
    return ImageGenerator()
