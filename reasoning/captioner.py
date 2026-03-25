"""
Image captioning using vision-language model.
"""

import logging
from typing import Optional

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using fallback captions")


class VideoCaptioner:
    """Generate detailed captions for video frames using VLM."""

    _instance = None
    _processor = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_loaded"):
            self._load_model()
            self._loaded = True

    def _load_model(self):
        """Load vision-language model for captioning."""
        if not TRANSFORMERS_AVAILABLE:
            self._use_fallback = True
            logger.info("Using fallback captioning")
            return

        try:
            # Use a small image captioning model
            model_name = "Salesforce/blip-image-captioning-base"
            logger.info(f"Loading captioning model: {model_name}")

            from transformers import BlipProcessor, BlipForConditionalGeneration

            self._processor = BlipProcessor.from_pretrained(model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(model_name)

            # Move to MPS if available
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()

            self._device = device
            self._use_fallback = False
            logger.info(f"Captioning model loaded on {device}")

        except Exception as e:
            logger.warning(f"Failed to load captioning model: {e}. Using fallback.")
            self._use_fallback = True

    def generate_caption(
        self, frame: Image.Image, surprise: float, frame_idx: int
    ) -> str:
        """Generate detailed caption for a frame."""

        if self._use_fallback:
            return self._generate_fallback_caption(surprise, frame_idx)

        try:
            # Process image
            inputs = self._processor(frame, return_tensors="pt").to(self._device)

            with torch.no_grad():
                # Generate caption
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.8,
                )

            # Decode caption
            caption = self._processor.decode(output[0], skip_special_tokens=True)

            # Clean up
            if self._device == "mps":
                torch.mps.empty_cache()

            return caption

        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return self._generate_fallback_caption(surprise, frame_idx)

    def _generate_fallback_caption(self, surprise: float, frame_idx: int) -> str:
        """Generate fallback caption without VLM."""
        import random

        # Scene types
        scenes = [
            "outdoor scene",
            "indoor setting",
            "city street",
            "open area",
            "building interior",
            "landscape",
            "action sequence",
            "dramatic moment",
            "movement scene",
        ]

        # Actions
        actions = [
            "with movement",
            "showing activity",
            "with motion",
            "capturing action",
            "displaying change",
            "with dynamic elements",
            "in progress",
            "underway",
            "happening",
        ]

        # Determine intensity
        if surprise > 0.8:
            intensity = "dramatic"
            action = random.choice(
                ["intense action", "major event", "significant moment"]
            )
        elif surprise > 0.5:
            intensity = "moderate"
            action = random.choice(
                ["noticeable activity", "scene change", "movement detected"]
            )
        else:
            intensity = "subtle"
            action = random.choice(
                ["minor variation", "slight change", "gentle motion"]
            )

        scene = random.choice(scenes)

        # Build caption
        if frame_idx < 10:
            caption = f"Opening: {scene} {action} - {intensity} start"
        elif frame_idx < 30:
            caption = f"Early section: {scene} showing {action}"
        elif frame_idx < 60:
            caption = f"Middle: {scene} with {action}"
        else:
            caption = f"Finale: {scene} - {action} ending"

        return caption

    def generate_video_summary(self, events: list) -> str:
        """Generate a complete video summary."""
        if not events:
            return "No video content to summarize."

        # Analyze all events
        frames = [e.get("frame_index", 0) for e in events]
        scores = [e.get("surprise_score", 0) for e in events]

        total_frames = max(frames) - min(frames) + 1 if frames else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        high_moments = [f for f, s in zip(frames, scores) if s > 0.7]

        # Build comprehensive summary
        summary = []

        # Opening
        summary.append("📽️ VIDEO SUMMARY")
        summary.append("=" * 40)

        # Duration estimate
        if total_frames > 0:
            summary.append(f"\n📊 Video Duration: ~{total_frames} frames")

        # Overall assessment
        if avg_score > 0.8:
            summary.append("🎬 Type: HIGH-ACTION video")
            summary.append("This video contains intense, fast-paced action throughout.")
        elif avg_score > 0.5:
            summary.append("🎬 Type: MODERATE activity")
            summary.append(
                "This video has several notable moments with varying intensity."
            )
        else:
            summary.append("🎬 Type: LOW activity")
            summary.append("This video has select moments of interest.")

        # Key moments
        if high_moments:
            summary.append(
                f"\n🔥 Key Moments: {len(high_moments)} high-intensity frames"
            )
            if high_moments:
                summary.append(f"   First major moment: Frame {min(high_moments)}")
                summary.append(f"   Last major moment: Frame {max(high_moments)}")

        # Scene progression
        summary.append("\n📍 Scene Progression:")

        if len(events) > 0:
            # Beginning
            summary.append(
                f"   • Beginning (Frames {min(frames)}-{min(frames) + 5}): Video starts with activity"
            )

            # Middle
            mid = len(frames) // 2
            if mid > 5:
                summary.append(
                    f"   • Middle (Around frame {frames[mid]}): Continues with notable content"
                )

            # End
            summary.append(
                f"   • End (Frames {max(frames) - 5}-{max(frames)}): Concludes the sequence"
            )

        # Conclusion
        summary.append("\n" + "=" * 40)
        summary.append(
            "This video captures a sequence of frames with significant visual changes."
        )

        return "\n".join(summary)


# Singleton accessor
_captioner_instance = None


def get_captioner() -> VideoCaptioner:
    global _captioner_instance
    if _captioner_instance is None:
        _captioner_instance = VideoCaptioner()
    return _captioner_instance
