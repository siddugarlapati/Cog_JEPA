"""
Video Pipeline Module

Orchestrates the full COG-JEPA inference loop with batch processing support.
"""

import logging
import time
import asyncio
from datetime import datetime
from typing import Optional, List, Deque
from collections import deque

import numpy as np
import torch
from PIL import Image

from config import config
from encoder.smolvlm_encoder import BatchVisionEncoder
from jepa.context_encoder import ContextEncoder
from reasoning.captioner import get_captioner
from jepa.predictor import Predictor
from jepa.surprise import SurpriseComputer
from memory.memory_gate import MemoryGate
from memory.cognee_store import CogneeMemoryStore
from reasoning.llm_reasoner import LLMReasoner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Main video processing pipeline for COG-JEPA.

    Orchestrates encoding, prediction, surprise computation, and memory storage.
    """

    def __init__(
        self,
        video_source: int | str = 0,
        use_llm: bool = True,
        threshold_override: Optional[float] = None,
    ):
        """
        Initialize video pipeline.

        Args:
            video_source: Video source - int (webcam index), str (file path), or "test"
            use_llm: Whether to use LLM for descriptions
            threshold_override: Override surprise threshold
        """
        self.video_source = video_source
        self.use_llm = use_llm
        self.running = False

        # Get device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Pipeline device: {self.device}")

        # Initialize batch-optimized encoder
        logger.info("Initializing batch vision encoder...")
        self.encoder = BatchVisionEncoder()

        logger.info("Initializing context encoder...")
        self.context_encoder = ContextEncoder()
        self.context_encoder = self.context_encoder.to(self.device)
        self.context_encoder.to(torch.float16)
        self.context_encoder.eval()

        logger.info("Initializing predictor...")
        self.predictor = Predictor()
        self.predictor = self.predictor.to(self.device)
        self.predictor.to(torch.float16)
        self.predictor.eval()

        # Initialize optimizer for online learning
        self.optimizer = torch.optim.Adam(
            list(self.predictor.parameters()), lr=config.online_learning_lr
        )

        logger.info("Initializing surprise computer...")
        self.surprise_computer = SurpriseComputer(ema_alpha=config.ema_alpha)

        # Override threshold if specified
        base_threshold = threshold_override or config.base_surprise_threshold
        logger.info("Initializing memory gate...")
        self.memory_gate = MemoryGate(
            base_threshold=base_threshold, ema_alpha=config.ema_alpha
        )

        logger.info("Initializing memory store...")
        self.memory_store = CogneeMemoryStore()

        logger.info("Initializing LLM reasoner...")
        self.llm_reasoner = LLMReasoner(use_fallback=not use_llm)

        # Context window - rolling buffer
        self.context_window: Deque[np.ndarray] = deque(
            maxlen=config.context_window_size
        )

        # Statistics
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self.surprise_history: List[float] = []

        # Video capture (will be initialized in run)
        self.cap = None

        logger.info("Pipeline initialized successfully")

    def _init_video_capture(self):
        """Initialize video capture based on source."""
        if self.video_source == "test":
            logger.info("Using synthetic test frames")
            self.cap = None
        elif isinstance(self.video_source, int):
            import cv2

            logger.info(f"Opening webcam {self.video_source}...")
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open webcam {self.video_source}")
        elif isinstance(self.video_source, str):
            import cv2

            logger.info(f"Opening video file {self.video_source}...")
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video {self.video_source}")
        else:
            raise ValueError(f"Invalid video source: {self.video_source}")

    def _get_test_frame(self, frame_idx: int) -> Image.Image:
        """Generate synthetic test frame with occasional anomalies."""
        # Base pattern: stable gradient
        x = np.linspace(0, 255, 224, dtype=np.float32)
        img = np.outer(x, x)

        # Normalize to 0-255
        img = (img / img.max() * 255).astype(np.uint8)

        # Add some noise
        noise_level = 10
        noise = np.random.randint(-noise_level, noise_level, (224, 224), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add color channel
        img = np.stack([img, img, img], axis=-1)

        # Occasionally add burst (high surprise)
        if frame_idx % 25 == 10 or frame_idx % 25 == 15:
            # Add bright burst in center
            center = 112
            radius = 30
            y, x = np.ogrid[:224, :224]
            mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
            img[mask] = 255

        return Image.fromarray(img)

    def _get_frame_from_cap(self) -> Optional[Image.Image]:
        """Get next frame from video capture."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame = frame[:, :, ::-1]
        return Image.fromarray(frame)

    def _update_context_window(self, latent: np.ndarray):
        """Add latent to rolling context window."""
        self.context_window.append(latent)

        # Pad with zeros if not full yet
        while len(self.context_window) < config.context_window_size:
            self.context_window.appendleft(
                np.zeros(config.latent_dim, dtype=np.float32)
            )

    def _predict_next(self) -> np.ndarray:
        """Predict next latent from context."""
        if len(self.context_window) < config.context_window_size:
            # Not enough context, return random prediction
            return np.random.randn(config.latent_dim).astype(np.float32)

        # Get context embeddings
        context_list = list(self.context_window)
        context_tensor = torch.stack([torch.from_numpy(z) for z in context_list]).to(
            self.device, torch.float16
        )

        # Encode context - no grad for inference, but store for update
        self.predictor.eval()
        with torch.no_grad():
            context_embedding = self.context_encoder(context_tensor)
            z_predicted = self.predictor(context_embedding)

        return z_predicted.detach().cpu().numpy().astype(np.float32)

    def _online_update(self, z_actual: torch.Tensor, z_predicted: torch.Tensor):
        """Update predictor weights online using prediction loss."""
        self.predictor.train()

        # Compute loss
        loss = torch.nn.functional.mse_loss(z_predicted, z_actual)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.predictor.parameters(), config.grad_clip_norm
        )

        self.optimizer.step()
        self.predictor.eval()

        return loss.item()

    def process_frame(self, frame: Image.Image) -> float:
        """
        Process a single frame through the pipeline.

        Args:
            frame: PIL Image frame

        Returns:
            Surprise score for this frame
        """
        # Step 1: Encode frame to latent
        z_actual = self.encoder.encode(frame)

        # Step 2: Predict next latent from context
        z_predicted = self._predict_next()

        # Step 3: Compute surprise
        surprise, l2_dist = self.surprise_computer.compute_surprise(
            z_predicted, z_actual
        )

        # Step 4: Update context window
        self._update_context_window(z_actual)

        # Step 5: Check memory gate
        should_store = self.memory_gate.should_store(surprise)

        if should_store:
            # Generate detailed caption using actual frame
            captioner = get_captioner()
            description = captioner.generate_caption(frame, surprise, self.frame_count)

            # Store in memory
            latent_hash = self.encoder.encode_hash(z_actual)
            context_scores = (
                self.surprise_history[-3:] if self.surprise_history else [0.0, 0.0, 0.0]
            )

            event = {
                "timestamp": datetime.now().isoformat(),
                "frame_index": self.frame_count,
                "surprise_score": surprise,
                "description": description,
                "latent_hash": latent_hash,
                "context_window": context_scores,
            }

            asyncio.run(self.memory_store.add_event(event))

            logger.info(
                f"[FRAME {self.frame_count}] Stored event - "
                f"surprise: {surprise:.3f}, description: {description[:60]}..."
            )

        # Step 6: Online update of predictor (optional - skip for now to ensure stability)
        # The predictor learns to minimize prediction error over time
        # For MVP, we skip this to ensure stability
        # In production, you would enable this for continuous learning

        # Clean up
        if self.device == "mps":
            torch.mps.empty_cache()

        self.frame_count += 1
        self.surprise_history.append(surprise)

        return surprise

    def run(self, max_frames: Optional[int] = None, display: bool = False):
        """
        Run the pipeline.

        Args:
            max_frames: Maximum frames to process (None = unlimited)
            display: Whether to display frames (not implemented for MVP)
        """
        self.running = True
        self.start_time = time.time()

        # Initialize video capture
        if self.video_source != "test":
            self._init_video_capture()

        logger.info(f"Starting pipeline (max_frames={max_frames})...")

        try:
            frame_idx = 0

            while self.running:
                # Check max frames
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break

                # Get frame
                if self.video_source == "test":
                    frame = self._get_test_frame(frame_idx)
                else:
                    frame = self._get_frame_from_cap()
                    if frame is None:
                        logger.info("End of video stream")
                        break

                # Process frame
                start_time = time.time()
                surprise = self.process_frame(frame)
                process_time = time.time() - start_time

                # Print stats every 10 frames
                if frame_idx % 10 == 0 and frame_idx > 0:
                    elapsed = time.time() - self.start_time
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    compression = self.memory_gate.get_compression_ratio()

                    stats = self.memory_gate.get_statistics()
                    logger.info(
                        f"[STATS] Frame {frame_idx} | "
                        f"FPS: {fps:.1f} | "
                        f"Surprise: {surprise:.3f} | "
                        f"Compression: {compression:.1%} | "
                        f"Stored: {stats['stored_frames']} | "
                        f"Process time: {process_time:.3f}s"
                    )

                frame_idx += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the pipeline and print summary."""
        self.running = False

        if self.cap is not None:
            self.cap.release()

        # Print session summary
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        stats = self.memory_gate.get_statistics()
        surprise_stats = self.surprise_computer.get_statistics()

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Frames processed: {self.frame_count}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Average FPS: {fps:.1f}")
        logger.info(
            f"Surprise scores - Mean: {surprise_stats.get('mean', 0):.3f}, "
            f"Max: {surprise_stats.get('max', 0):.3f}"
        )
        logger.info(f"Events stored: {stats['stored_frames']}")
        logger.info(f"Events discarded: {stats['discarded_frames']}")
        logger.info(f"Compression ratio: {stats['compression_ratio']:.1%}")
        logger.info("=" * 60)

    def get_current_state(self) -> dict:
        """Get current pipeline state for UI."""
        stats = self.memory_gate.get_statistics()
        surprise_stats = self.surprise_computer.get_statistics()

        return {
            "frame_count": self.frame_count,
            "surprise_score": self.surprise_history[-1] if self.surprise_history else 0,
            "surprise_history": self.surprise_history[-100:],
            "stored_events": stats["stored_frames"],
            "compression_ratio": stats["compression_ratio"],
            "current_threshold": stats["current_threshold"],
            "avg_surprise": surprise_stats.get("mean", 0),
        }


def test_pipeline():
    """Test function for synthetic pipeline."""
    logger.info("Testing VideoPipeline with synthetic frames...")

    pipeline = VideoPipeline(video_source="test", use_llm=False)
    pipeline.run(max_frames=50)

    return pipeline


if __name__ == "__main__":
    test_pipeline()
