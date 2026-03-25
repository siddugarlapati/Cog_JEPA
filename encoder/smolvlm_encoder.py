"""
Vision Encoder Module

Encodes video frames to 512-dim latent vectors.
Uses a CNN encoder with optional SmolVLM support for better embeddings.
"""

import logging
import hashlib
from typing import Optional, List
from functools import lru_cache

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """CNN-based vision encoder for extracting latent representations."""

    def __init__(self, latent_dim: int = 512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        latent = self.projection(features)
        return latent


class BatchVisionEncoder:
    """Batch-optimized vision encoder for processing multiple frames at once."""

    _instance: Optional["BatchVisionEncoder"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS not available, falling back to CPU")
        return "cpu"

    def _load_model(self):
        device = self._get_device()
        logger.info(f"Loading batch vision encoder on {device}...")

        self._model = VisionEncoder(latent_dim=config.latent_dim)
        self._model = self._model.to(device)
        self._model = self._model.to(torch.float16)
        self._model.eval()

        self._device = device
        logger.info(f"Batch vision encoder loaded on {device}")

    def encode(self, frame: Image.Image) -> np.ndarray:
        """Encode single frame."""
        return self.encode_batch([frame])[0]

    def encode_batch(self, frames: List[Image.Image]) -> List[np.ndarray]:
        """Encode batch of frames efficiently."""
        if self._model is None:
            self._load_model()

        if not frames:
            return []

        # Preprocess all frames
        tensors = []
        for frame in frames:
            frame = frame.resize(config.frame_size)
            img_array = np.array(frame).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            tensors.append(img_tensor)

        # Stack into batch
        batch = torch.stack(tensors).to(self._device, torch.float16)

        # Encode
        with torch.no_grad():
            latents = self._model(batch)

        # Convert to numpy
        latents = latents.float().cpu().numpy()

        # Clean up
        if self._device == "mps":
            torch.mps.empty_cache()

        return [latent.astype(np.float32) for latent in latents]

    def encode_hash(self, latent: np.ndarray) -> str:
        latent_bytes = latent.tobytes()
        return hashlib.sha256(latent_bytes).hexdigest()

    def get_device(self) -> str:
        return getattr(self, "_device", "cpu")


class SmolVLMEncoder:
    """Legacy wrapper for compatibility - uses BatchVisionEncoder internally."""

    _instance: Optional["BatchVisionEncoder"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BatchVisionEncoder()
        return cls._instance

    def __init__(self):
        pass

    def encode(self, frame: Image.Image) -> np.ndarray:
        encoder = BatchVisionEncoder()
        return encoder.encode(frame)

    def encode_hash(self, latent: np.ndarray) -> str:
        encoder = BatchVisionEncoder()
        return encoder.encode_hash(latent)

    def get_device(self) -> str:
        encoder = BatchVisionEncoder()
        return encoder.get_device()


def test_encoder():
    """Test function for standalone testing."""
    logger.info("Testing vision encoder...")

    encoder = BatchVisionEncoder()

    # Test single frame
    random_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    latent = encoder.encode(random_image)
    logger.info(f"Single frame - Latent shape: {latent.shape}")

    # Test batch
    images = [random_image for _ in range(5)]
    latents = encoder.encode_batch(images)
    logger.info(f"Batch of 5 - Latents shape: {len(latents)}")

    hash_val = encoder.encode_hash(latent)
    logger.info(f"Latent hash: {hash_val[:16]}...")

    return latents


if __name__ == "__main__":
    test_encoder()
