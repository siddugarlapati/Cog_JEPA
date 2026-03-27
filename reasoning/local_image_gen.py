"""
Local Image Generation using Stable Diffusion (MPS-accelerated on Mac).
Falls back to FREE Pollinations.ai API when SD is not available.
Pollinations.ai provides real AI-generated images using FLUX model — no auth required.
"""

import logging
import os
import time
import uuid
import requests
from io import BytesIO
import urllib.parse

from PIL import Image, ImageDraw
import textwrap

logger = logging.getLogger(__name__)

# Diffusers availability checked lazily inside _load_model to avoid 5-min startup
DIFFUSERS_AVAILABLE = None  # None = not checked yet


class LocalImageGenerator:
    """Generate images locally using Stable Diffusion on MPS, or via Pollinations.ai API."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
            cls._instance._device = "cpu"
            cls._instance._use_fallback = True
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self):
        global DIFFUSERS_AVAILABLE
        # Lazy import — only triggered when LocalImageGenerator is first created
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import torch as _torch
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
                DIFFUSERS_AVAILABLE = True
            except ImportError:
                DIFFUSERS_AVAILABLE = False

        if not DIFFUSERS_AVAILABLE:
            logger.info("diffusers not available – using Pollinations.ai API for real image generation")
            self._use_fallback = True
            return

        import torch as _torch
        device = "mps" if _torch.backends.mps.is_available() else "cpu"
        dtype = _torch.float16 if device == "mps" else _torch.float32

        # Try SDXL-Turbo first (fast, good quality)
        for model_id, cls_ in [
            ("stabilityai/sdxl-turbo", StableDiffusionXLPipeline),
            ("runwayml/stable-diffusion-v1-5", StableDiffusionPipeline),
        ]:
            try:
                logger.info(f"Loading {model_id} on {device}...")
                pipe = cls_.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    variant="fp16" if device == "mps" else None,
                )
                pipe = pipe.to(device)
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                self._pipeline = pipe
                self._device = device
                self._use_fallback = False
                logger.info(f"✅ {model_id} loaded on {device}")
                return
            except Exception as e:
                logger.warning(f"Could not load {model_id}: {e}")

        logger.warning("All SD models failed – using Pollinations.ai API for real image generation")
        self._use_fallback = True

    # ── Public API ───────────────────────────────────────────────────────────

    def generate_image(self, prompt: str, size: str = "512x512", model: str = "flux") -> Image.Image:
        """
        Generate an image from a text prompt.

        Returns a PIL Image (always – never None).
        Uses local SD if available, otherwise Pollinations.ai API (FREE, real AI images).

        Args:
            prompt: Text description of image to generate
            size: Resolution string like "512x512" or "1024x1024"
            model: Pollinations model: "flux", "flux-realism", "flux-anime", "flux-3d", "turbo"
        """
        if self._use_fallback or self._pipeline is None:
            return self._generate_via_api(prompt, size, model=model)

        try:
            w, h = self._parse_size(size)
            steps = 4 if "turbo" in str(type(self._pipeline)).lower() else 20

            result = self._pipeline(
                prompt=prompt,
                width=w,
                height=h,
                num_inference_steps=steps,
                guidance_scale=0.0 if steps <= 4 else 7.5,
            )
            pil_img = result.images[0]

            os.makedirs("data", exist_ok=True)
            path = f"data/generated_{uuid.uuid4().hex[:8]}.png"
            pil_img.save(path)
            logger.info(f"Image saved to {path}")

            if self._device == "mps":
                import torch as _torch
                _torch.mps.empty_cache()

            return pil_img

        except Exception as e:
            logger.error(f"Local generation error: {e}, falling back to API")
            return self._generate_via_api(prompt, size, model=model)

    def generate_from_video_context(self, events: list) -> Image.Image:
        """Generate image from the most dramatic video event."""
        if not events:
            return self._placeholder("No video events available")

        top = max(events, key=lambda e: e.get("surprise_score", 0))
        desc = top.get("description", "dramatic scene")
        prompt = (
            f"Cinematic movie still: {desc}, "
            "dramatic lighting, professional cinematography, 4k, photorealistic"
        )
        return self.generate_image(prompt, model="flux-realism")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _parse_size(self, size: str):
        """Parse size string to (width, height) tuple."""
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 512, 512

    # ── API Generation (FREE - Pollinations.ai) ─────────────────────────────

    def _generate_via_api(self, prompt: str, size: str = "512x512", model: str = "flux") -> Image.Image:
        """
        Generate image using FREE Pollinations.ai API.
        No API key required. Uses FLUX model for high-quality images.
        
        API: https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&model={model}
        """
        w, h = self._parse_size(size)
        
        # Encode prompt safely for URL
        encoded_prompt = urllib.parse.quote(prompt[:500], safe="")
        
        # Add unique seed for variety
        seed = abs(hash(prompt + str(time.time()))) % 1000000
        
        # Build URL with all params
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={w}&height={h}&model={model}&seed={seed}&nologo=true&enhance=false"
        )
        
        logger.info(f"🎨 Generating via Pollinations.ai ({model}, {w}x{h})...")
        logger.info(f"   URL: {url[:100]}...")
        
        # Retry up to 3 times
        for attempt in range(1, 4):
            try:
                logger.info(f"   Attempt {attempt}/3...")
                response = requests.get(
                    url,
                    timeout=120,
                    stream=True,
                    headers={
                        "User-Agent": "COG-JEPA/1.0",
                        "Accept": "image/*,*/*",
                    }
                )
                
                if response.status_code == 200:
                    # Check content type - should be an image
                    content_type = response.headers.get("content-type", "")
                    
                    # Read all image data
                    image_data = BytesIO()
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            image_data.write(chunk)
                            total_size += len(chunk)
                    
                    if total_size < 1000:
                        logger.warning(f"   Response too small ({total_size} bytes), retrying...")
                        time.sleep(2)
                        continue
                    
                    image_data.seek(0)
                    
                    try:
                        pil_img = Image.open(image_data).convert("RGB")
                        img_w, img_h = pil_img.size
                        
                        # Validate: reject tiny images (likely error responses)
                        if img_w < 64 or img_h < 64:
                            logger.warning(f"   Image too small ({img_w}x{img_h}), retrying...")
                            time.sleep(2)
                            continue
                        
                        # Save to disk
                        os.makedirs("data", exist_ok=True)
                        path = f"data/generated_{uuid.uuid4().hex[:8]}.png"
                        pil_img.save(path)
                        logger.info(f"✅ Image generated ({img_w}x{img_h}, {total_size} bytes) → {path}")
                        
                        return pil_img
                        
                    except Exception as img_err:
                        logger.error(f"   Failed to decode image: {img_err}")
                        time.sleep(2)
                        continue
                        
                elif response.status_code == 429:
                    logger.warning(f"   Rate limited (429), waiting 5s...")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"   API returned status {response.status_code}")
                    time.sleep(2)
                    continue
                    
            except requests.Timeout:
                logger.error(f"   Attempt {attempt} timed out after 120s")
                time.sleep(2)
                continue
            except requests.ConnectionError as e:
                logger.error(f"   Connection error: {e}")
                time.sleep(3)
                continue
            except Exception as e:
                logger.error(f"   Attempt {attempt} failed: {e}")
                time.sleep(2)
                continue
        
        logger.error("All Pollinations.ai attempts failed – returning placeholder")
        return self._placeholder(prompt)

    # ── Placeholder ──────────────────────────────────────────────────────────

    def _placeholder(self, prompt: str) -> Image.Image:
        """Return a styled placeholder image only when API also fails."""
        img = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(img)

        # Gradient background
        for y in range(512):
            t = y / 512
            r = int(20 + t * 40)
            g = int(20 + t * 15)
            b = int(60 + t * 80)
            draw.line([(0, y), (512, y)], fill=(r, g, b))

        # Border
        draw.rectangle([10, 10, 501, 501], outline=(120, 100, 220), width=2)

        # Title
        draw.text((256, 60), "🎨 Image Generation", fill=(200, 180, 255), anchor="mm")
        draw.text((256, 95), "API Temporarily Unavailable", fill=(160, 140, 200), anchor="mm")
        draw.line([(40, 115), (472, 115)], fill=(80, 70, 140), width=1)

        # Prompt preview
        draw.text((256, 140), "Prompt:", fill=(180, 180, 220), anchor="mm")
        lines = textwrap.wrap(prompt[:180], width=38)
        y = 170
        for line in lines[:6]:
            draw.text((256, y), line, fill=(220, 215, 240), anchor="mm")
            y += 28

        # Footer
        draw.text((256, 440), "Check internet connection", fill=(100, 90, 160), anchor="mm")
        draw.text((256, 465), "or try again in a moment", fill=(100, 90, 160), anchor="mm")

        return img


# Singleton
_instance = None

def get_image_generator() -> LocalImageGenerator:
    global _instance
    if _instance is None:
        _instance = LocalImageGenerator()
    return _instance
