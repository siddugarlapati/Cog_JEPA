"""
Local Image Generation using SDXL-Turbo (MPS-accelerated on Mac).
Applies the upstream MPS float16 VAE black-image fix.
"""

import logging
import os
import uuid
import warnings

logger = logging.getLogger(__name__)


class LocalImageGenerator:
    """Generate images using local SDXL-Turbo on MPS."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load SDXL-Turbo with MPS VAE upcast fix."""
        if self._pipeline is not None:
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import torch
                from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                logger.info(f"Loading SDXL-Turbo on {device}...")

                pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype=torch.float16,
                )
                pipe = pipe.to(device)

                # ── MPS BLACK IMAGE FIX ───────────────────────────────────
                # The VAE decoder produces black images with float16 on MPS.
                # Upcasting only the VAE to float32 fixes it while keeping
                # the rest of the pipeline fast in float16.
                pipe.vae = pipe.vae.to(dtype=torch.float32)
                # ─────────────────────────────────────────────────────────

                pipe.enable_attention_slicing()
                self._pipeline = pipe
                self._device = device
                logger.info(f"✅ SDXL-Turbo ready on {device} (VAE in float32)")
                return True
            except Exception as e:
                logger.error(f"Failed to load SDXL-Turbo: {e}")
                self._pipeline = None
                return False

    def generate_image(self, prompt: str, size: str = "512x512", **kwargs):
        """Generate a real image. Returns PIL Image or None."""
        from PIL import Image
        import numpy as np

        if not self._load_pipeline():
            logger.error("Pipeline not available.")
            return None

        try:
            import torch
            # Parse size
            parts = size.lower().replace(" ", "").split("x")
            w, h = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (512, 512)
            # SDXL-Turbo requires multiples of 8
            w = (w // 8) * 8
            h = (h // 8) * 8

            logger.info(f"Generating: '{prompt[:60]}' @ {w}x{h}")

            with torch.no_grad():
                result = self._pipeline(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=4,
                    guidance_scale=0.0,  # SDXL-Turbo requires 0.0
                )

            img = result.images[0]

            # Sanity check: reject black images and retry with higher steps
            arr = np.array(img)
            if arr.mean() < 2.0:
                logger.warning("Image looks black, retrying with guidance_scale=2.0...")
                with torch.no_grad():
                    result = self._pipeline(
                        prompt=prompt,
                        width=w, height=h,
                        num_inference_steps=8,
                        guidance_scale=2.0,
                    )
                img = result.images[0]

            # Save
            os.makedirs("data", exist_ok=True)
            path = f"data/generated_{uuid.uuid4().hex[:8]}.png"
            img.save(path)
            logger.info(f"Saved to {path}")

            if hasattr(self, "_device") and self._device == "mps":
                torch.mps.empty_cache()

            return img

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return None


# Singleton
_instance = None

def get_image_generator() -> LocalImageGenerator:
    global _instance
    if _instance is None:
        _instance = LocalImageGenerator()
    return _instance
