"""
Surprise Module

Computes prediction error between predicted and actual latent vectors.
"""

import logging
from typing import Tuple

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurpriseComputer:
    """
    Computes surprise score (prediction error) between predicted and actual latents.

    Uses cosine distance as primary metric and L2 norm as secondary.
    Maintains EMA baseline for adaptive thresholding.
    """

    def __init__(self, ema_alpha: float = 0.1):
        """
        Initialize surprise computer.

        Args:
            ema_alpha: EMA smoothing factor for baseline (default: 0.1)
        """
        self.ema_alpha = ema_alpha
        self.ema_baseline: float = 0.0
        self.surprise_history: list = []

        logger.info(f"SurpriseComputer initialized with alpha={ema_alpha}")

    def compute_surprise(
        self, z_predicted: np.ndarray, z_actual: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute surprise score between predicted and actual latent vectors.

        Args:
            z_predicted: Predicted latent vector (512-dim)
            z_actual: Actual latent vector (512-dim)

        Returns:
            Tuple of (cosine_distance, l2_distance)
        """
        # Ensure numpy arrays
        if isinstance(z_predicted, torch.Tensor):
            z_predicted = z_predicted.cpu().numpy()
        if isinstance(z_actual, torch.Tensor):
            z_actual = z_actual.cpu().numpy()

        # Flatten if needed
        z_predicted = z_predicted.flatten()
        z_actual = z_actual.flatten()

        # Cosine distance: 1 - cosine_similarity
        cos_sim = cosine_similarity(
            z_predicted.reshape(1, -1), z_actual.reshape(1, -1)
        )[0, 0]
        cosine_distance = 1.0 - cos_sim

        # L2 distance
        l2_distance = np.linalg.norm(z_predicted - z_actual)

        # Update EMA baseline
        self.update_ema(cosine_distance)

        # Store in history
        self.surprise_history.append(cosine_distance)

        return float(cosine_distance), float(l2_distance)

    def update_ema(self, surprise_score: float):
        """Update EMA baseline with new surprise score."""
        if self.ema_baseline == 0.0:
            self.ema_baseline = surprise_score
        else:
            self.ema_baseline = (
                self.ema_alpha * surprise_score
                + (1 - self.ema_alpha) * self.ema_baseline
            )

    def get_adaptive_threshold(self, multiplier: float = 1.5) -> float:
        """
        Get adaptive threshold based on EMA baseline.

        Args:
            multiplier: Multiplier for EMA baseline (default: 1.5)

        Returns:
            Adaptive threshold value
        """
        return self.ema_baseline * multiplier

    def get_statistics(self) -> dict:
        """Get statistics about surprise scores."""
        if not self.surprise_history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "ema_baseline": 0.0,
            }

        return {
            "mean": float(np.mean(self.surprise_history)),
            "std": float(np.std(self.surprise_history)),
            "min": float(np.min(self.surprise_history)),
            "max": float(np.max(self.surprise_history)),
            "ema_baseline": float(self.ema_baseline),
            "count": len(self.surprise_history),
        }

    def reset(self):
        """Reset history and EMA baseline."""
        self.surprise_history.clear()
        self.ema_baseline = 0.0


def test_surprise():
    """Test function for standalone testing."""
    logger.info("Testing SurpriseComputer...")

    computer = SurpriseComputer(ema_alpha=0.1)

    # Test with similar vectors (low surprise)
    z1 = np.random.randn(config.latent_dim).astype(np.float32)
    z2 = z1 + np.random.randn(config.latent_dim).astype(np.float32) * 0.1

    surprise, l2 = computer.compute_surprise(z1, z2)
    logger.info(f"Similar vectors - Surprise: {surprise:.4f}, L2: {l2:.4f}")

    # Test with different vectors (high surprise)
    z3 = np.random.randn(config.latent_dim).astype(np.float32)
    surprise2, l2_2 = computer.compute_surprise(z1, z3)
    logger.info(f"Different vectors - Surprise: {surprise2:.4f}, L2: {l2_2:.4f}")

    # Test threshold
    threshold = computer.get_adaptive_threshold()
    logger.info(f"Adaptive threshold: {threshold:.4f}")

    # Test statistics
    stats = computer.get_statistics()
    logger.info(f"Statistics: {stats}")

    # Test surprise score range
    assert 0 <= surprise <= 2, f"Surprise score {surprise} out of range [0, 2]"
    assert 0 <= surprise2 <= 2, f"Surprise score {surprise2} out of range [0, 2]"

    return computer


if __name__ == "__main__":
    test_surprise()
