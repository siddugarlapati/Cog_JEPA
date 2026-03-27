"""
Memory Gate Module

Threshold gate that decides whether to store events in memory based on surprise score.
Uses adaptive EMA thresholding for intelligent filtering.
"""

import logging
from typing import Dict

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryGate:
    """
    Memory gate that decides what to store based on surprise score.

    Uses adaptive threshold based on EMA of surprise scores.
    """

    def __init__(self, base_threshold: float = 0.3, ema_alpha: float = 0.1):
        """
        Initialize memory gate.

        Args:
            base_threshold: Initial threshold for surprise score (default: 0.3)
            ema_alpha: EMA smoothing factor (default: 0.1)
        """
        self.base_threshold = base_threshold
        self.ema_alpha = ema_alpha

        # Statistics tracking
        self.total_frames: int = 0
        self.stored_frames: int = 0
        self.discarded_frames: int = 0
        self.surprise_scores: list = []

        # EMA baseline for adaptive thresholding
        self.ema_baseline: float = 0.0
        self.threshold_multiplier: float = 0.5

        logger.info(
            f"MemoryGate initialized: base_threshold={base_threshold}, ema_alpha={ema_alpha}"
        )

    def should_store(self, surprise_score: float) -> bool:
        """
        Decide whether to store this frame in memory.

        Args:
            surprise_score: Computed surprise score for the frame

        Returns:
            True if frame should be stored, False otherwise
        """
        self.total_frames += 1
        self.surprise_scores.append(surprise_score)

        # Update EMA baseline
        if self.ema_baseline == 0.0:
            self.ema_baseline = surprise_score
        else:
            self.ema_baseline = (
                self.ema_alpha * surprise_score
                + (1 - self.ema_alpha) * self.ema_baseline
            )

        # Calculate adaptive threshold
        adaptive_threshold = self.get_adaptive_threshold()

        # Decision
        should_store = surprise_score >= adaptive_threshold

        if should_store:
            self.stored_frames += 1
        else:
            self.discarded_frames += 1

        return should_store

    def get_adaptive_threshold(self) -> float:
        """
        Get adaptive threshold based on EMA baseline.

        Returns:
            Adaptive threshold value
        """
        if self.ema_baseline > 0:
            # Adaptive: base_threshold scaled by EMA baseline
            return max(self.base_threshold, self.ema_baseline * (1.0 + self.threshold_multiplier))
        return self.base_threshold

    def get_compression_ratio(self) -> float:
        """
        Get memory compression ratio (stored / total).

        Returns:
            Compression ratio as float between 0 and 1
        """
        if self.total_frames == 0:
            return 0.0
        return self.stored_frames / self.total_frames

    def get_statistics(self) -> Dict:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_frames": self.total_frames,
            "stored_frames": self.stored_frames,
            "discarded_frames": self.discarded_frames,
            "compression_ratio": self.get_compression_ratio(),
            "current_threshold": self.get_adaptive_threshold(),
            "ema_baseline": self.ema_baseline,
            "avg_surprise": (
                sum(self.surprise_scores) / len(self.surprise_scores)
                if self.surprise_scores
                else 0.0
            ),
        }

    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_frames = 0
        self.stored_frames = 0
        self.discarded_frames = 0
        self.surprise_scores.clear()
        self.ema_baseline = 0.0


def test_memory_gate():
    """Test function for standalone testing."""
    logger.info("Testing MemoryGate...")

    gate = MemoryGate(base_threshold=0.3, ema_alpha=0.1)

    # Test with low surprise (should discard)
    for _ in range(5):
        result = gate.should_store(0.1)
        logger.info(f"Low surprise (0.1) - stored: {result}")

    # Test with high surprise (should store)
    for _ in range(3):
        result = gate.should_store(0.8)
        logger.info(f"High surprise (0.8) - stored: {result}")

    # Test statistics
    stats = gate.get_statistics()
    logger.info(f"Statistics: {stats}")

    # Test compression ratio
    compression = gate.get_compression_ratio()
    logger.info(f"Compression ratio: {compression:.2%}")

    # Test adaptive threshold
    threshold = gate.get_adaptive_threshold()
    logger.info(f"Adaptive threshold: {threshold:.4f}")

    return gate


if __name__ == "__main__":
    test_memory_gate()
