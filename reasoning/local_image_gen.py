"""
Local Image Generation using Stable Diffusion

Generates images locally on Mac M4 using MPS acceleration.
"""

import logging
import torch
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Try to import diffusers
try:
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not available")


class LocalImageGenerator:
    """Generate images locally using Stable Diffusion."""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()

    def _load_model(self):
        """Load Stable Diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            self._use_fallback = True
            logger.info("Using fallback image generation")
            return

        try:
            # Use a smaller, faster model for Mac M4
            model_id = "stabilityai/sdxl-turbo"

            logger.info(f"Loading Stable Diffusion model: {model_id}")

            # Determine device
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Metal) acceleration")
            else:
                device = "cpu"

            # Load pipeline with minimal memory usage
            self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                variant="fp16" if device == "mps" else None,
            )

            self._pipeline = self._pipeline.to(device)

            # Optimize for M1/M4
            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()

            self._device = device
            self._use_fallback = False
            logger.info(f"Stable Diffusion loaded on {device}")

        except Exception as e:
            logger.warning(f"Failed to load SDXL: {e}")
            # Try smaller model
            try:
                model_id = "runwayml/stable-diffusion-v1-5"
                logger.info(f"Trying smaller model: {model_id}")

                self._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                self._pipeline = self._pipeline.to("mps")
                self._device = "mps"
                self._use_fallback = False
                logger.info("Stable Diffusion v1.5 loaded on MPS")

            except Exception as e2:
                logger.warning(f"Failed to load SD v1.5: {e2}")
                self._use_fallback = True

    def generate_image(self, prompt: str, size: str = "512x512") -> str:
        """
        Generate an image from text prompt.

        Args:
            prompt: Description of image
            size: Image dimensions (512x512 or 1024x1024)

        Returns:
            Path to saved image or error message
        """
        if self._use_fallback:
            return self._fallback_generate(prompt)

        try:
            # Parse size
            if "1024" in size:
                width, height = 1024, 1024
                num_inference_steps = 30
            else:
                width, height = 512, 512
                num_inference_steps = 20

            # Generate
            image = self._pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,
            ).images[0]

            # Save
            import uuid

            filename = f"data/generated_{uuid.uuid4().hex[:8]}.png"

            # Ensure data directory exists
            import os

            os.makedirs("data", exist_ok=True)

            image.save(filename)

            # Clean up MPS memory
            if self._device == "mps":
                torch.mps.empty_cache()

            return filename

        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when SD is not available."""
        return f"Image generation requires Stable Diffusion model. For now, use the video query feature!"

    def generate_from_video_context(self, events: list, frame_idx: int = None) -> str:
        """Generate image based on video context."""
        if not events:
            return "No video events available."

        # Get representative event
        if frame_idx:
            event = next(
                (e for e in events if e.get("frame_index") == frame_idx), events[-1]
            )
        else:
            event = max(events, key=lambda x: x.get("surprise_score", 0))

        desc = event.get("description", "")

        # Build cinematic prompt
        prompt = f"""Cinematic movie still: {desc}, 
professional cinematography, dramatic lighting, 4k, photorealistic, film grain"""

        return self.generate_image(prompt)


# Singleton accessor
def get_image_generator() -> LocalImageGenerator:
    """Get or create image generator."""
    return LocalImageGenerator()
