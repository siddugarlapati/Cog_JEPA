"""
LLM Reasoner Module

llama.cpp wrapper for querying memory and describing events.
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMReasoner:
    """
    LLM-based reasoner using llama.cpp with Metal support.

    Provides methods for describing events and querying memory.
    Falls back to rule-based descriptions if LLM fails to load.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        use_fallback: bool = False,
    ):
        """
        Initialize LLM reasoner.

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size
            use_fallback: Skip LLM loading, use rule-based only
        """
        self.model_path = model_path or config.llm_model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.llm = None
        self.use_fallback = use_fallback or not os.path.exists(self.model_path)

        if not self.use_fallback:
            self._load_llm()
        else:
            logger.warning("Using rule-based fallback describer (no LLM)")

    def _load_llm(self):
        """Load llama.cpp model with Metal support."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM from {self.model_path}...")

            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=8,
                verbose=False,
            )

            logger.info("LLM loaded successfully with Metal support")

        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}. Using fallback.")
            self.use_fallback = True
            self.llm = None

    def _get_score_label(self, score: float) -> str:
        """Convert surprise score to label."""
        if score < 0.2:
            return "routine and expected"
        elif score < 0.4:
            return "somewhat unusual"
        elif score < 0.6:
            return "notable and unexpected"
        elif score < 0.8:
            return "significant and surprising"
        else:
            return "highly unusual and noteworthy"

    def describe_event(
        self, latent_vec: np.ndarray, surprise_score: float, frame_idx: int = 0
    ) -> str:
        """
        Generate natural language description of an event.

        Args:
            latent_vec: Latent vector (512-dim)
            surprise_score: Computed surprise score
            frame_idx: Frame index for context

        Returns:
            Natural language description
        """
        score_label = self._get_score_label(surprise_score)

        if self.use_fallback or self.llm is None:
            # Improved rule-based fallback with more variety
            import random

            # Generate varied descriptions based on score
            if surprise_score < 0.3:
                templates = [
                    f"Frame {frame_idx}: Routine scene, minimal activity detected.",
                    f"Frame {frame_idx}: Stable environment, no significant changes.",
                    f"Frame {frame_idx}: Normal operation, consistent with previous frames.",
                ]
                description = random.choice(templates)
            elif surprise_score < 0.6:
                templates = [
                    f"Frame {frame_idx}: Moderate motion detected, slight scene variation.",
                    f"Frame {frame_idx}: Some activity change observed, worth noting.",
                    f"Frame {frame_idx}: Minor event occurred, subtle shift in frame.",
                ]
                description = random.choice(templates)
            elif surprise_score < 0.8:
                templates = [
                    f"Frame {frame_idx}: SIGNIFICANT change! Major motion or object detected.",
                    f"Frame {frame_idx}: Unexpected event - potential action/entry detected.",
                    f"Frame {frame_idx}: IMPORTANT - Notable scene transformation occurred.",
                ]
                description = random.choice(templates)
            else:
                templates = [
                    f"Frame {frame_idx}: CRITICAL ALERT! Major unexpected event detected.",
                    f"Frame {frame_idx}: HIGH PRIORITY - Significant anomaly captured.",
                    f"Frame {frame_idx}: NOTABLE EVENT - Dramatic scene change occurred.",
                ]
                description = random.choice(templates)

            return description

        # Use LLM
        try:
            prompt = f"""You are analyzing a video surveillance event. 
Surprise score: {surprise_score:.3f}. 
This event was {score_label}.
Describe what might have happened in 1-2 sentences.
Be specific about what might have changed in the scene."""

            response = self.llm(prompt, max_tokens=100, temperature=0.7, stop=["\n\n"])

            description = response["choices"][0]["text"].strip()

            return description

        except Exception as e:
            logger.warning(f"LLM inference failed: {e}. Using fallback.")
            return self._describe_event_fallback(surprise_score, score_label)

    def _describe_event_fallback(self, score: float, label: str) -> str:
        """Fallback description without LLM."""
        if score < 0.3:
            return (
                f"Routine scene captured. {label.capitalize()}. Surprise: {score:.3f}"
            )
        elif score < 0.6:
            return f"Activity detected in scene. {label.capitalize()}. Surprise: {score:.3f}"
        else:
            return f"Major change detected! {label.capitalize()}. Immediate attention recommended. Surprise: {score:.3f}"

    def query_memory(self, question: str, context: str) -> str:
        """
        Query memory with context.

        Args:
            question: User question
            context: Memory events context

        Returns:
            Answer string
        """
        if self.use_fallback or self.llm is None:
            # Rule-based fallback
            return self._query_memory_fallback(question, context)

        try:
            prompt = f"""Based on these memory events:
{context}

Answer the following question:
{question}

Provide a concise answer."""

            response = self.llm(prompt, max_tokens=200, temperature=0.7, stop=["\n\n"])

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.warning(f"LLM query failed: {e}. Using fallback.")
            return self._query_memory_fallback(question, context)

    def _query_memory_fallback(self, question: str, context: str) -> str:
        """Fallback query without LLM - narrative responses."""
        import re

        # Parse all events
        events = []
        for line in context.split("\n"):
            frame_match = re.search(r"Frame (\d+)", line)
            score_match = re.search(r"surprise=([\d.]+)", line)
            desc_match = re.search(r"description=(.+?)(?:\.|,|$)", line)
            if frame_match and score_match:
                events.append(
                    {
                        "frame": int(frame_match.group(1)),
                        "score": float(score_match.group(1)),
                        "desc": desc_match.group(1) if desc_match else "",
                    }
                )

        if not events:
            return "No events found. Please process a video first."

        # Sort by frame
        events = sorted(events, key=lambda x: x["frame"])

        question_lower = question.lower()

        # SUMMARY - Tell a story about the video
        if (
            "summary" in question_lower
            or "overall" in question_lower
            or "explain" in question_lower
            or "story" in question_lower
            or ("what" in question_lower and "happen" in question_lower)
        ):
            # Analyze the pattern
            first_frame = events[0]["frame"]
            last_frame = events[-1]["frame"]
            total_frames = last_frame - first_frame + 1
            event_count = len(events)

            # Determine video type based on pattern
            if event_count >= total_frames * 0.8:
                video_type = "action-packed"
                opening = "This video is FULL OF ACTION from start to finish!"
            elif event_count >= total_frames * 0.5:
                video_type = "eventful"
                opening = "This video has several action sequences throughout."
            else:
                video_type = "selective"
                opening = "This video has specific moments of interest."

            # Build narrative
            response = f"🎬 VIDEO SCENE ANALYSIS\n"
            response += f"{'=' * 40}\n\n"
            response += f"{opening}\n\n"

            # Break into segments
            segments = []
            if len(events) >= 3:
                # Find clusters
                cluster = [events[0]]
                for e in events[1:]:
                    if e["frame"] - cluster[-1]["frame"] <= 3:
                        cluster.append(e)
                    else:
                        if cluster:
                            segments.append(cluster)
                        cluster = [e]
                if cluster:
                    segments.append(cluster)

            # Describe segments
            if segments:
                response += f"📽️ SCENE BREAKDOWN:\n\n"
                for i, seg in enumerate(segments[:5], 1):
                    start_frame = seg[0]["frame"]
                    end_frame = seg[-1]["frame"]
                    avg_score = sum(e["score"] for e in seg) / len(seg)

                    # Generate scene description based on frame position
                    if i == 1:
                        scene_desc = (
                            "EXCITING OPENING - Major action starts immediately"
                        )
                    elif len(segments) == 1:
                        scene_desc = "MAIN ACTION - Key event throughout"
                    elif i == len(segments):
                        scene_desc = "FINALE - Climactic ending sequence"
                    else:
                        scene_desc = f"Action sequence {i}"

                    response += f"Scene {i} (Frames {start_frame}-{end_frame}):\n"
                    response += f"   {scene_desc}\n"
                    response += f"   Intensity: {'🔴 HIGH' if avg_score > 0.8 else '🟡 MEDIUM'}\n\n"

            # Conclusion
            response += f"{'=' * 40}\n"
            response += f"📊 SUMMARY:\n"
            response += f"• Total video coverage: {total_frames} frames\n"
            response += f"• Key moments captured: {event_count}\n"
            response += f"• Video type: {video_type.replace('-', ' ').title()}\n\n"

            if event_count > 30:
                response += (
                    "🎯 This is a HIGH-ACTION video with continuous significant events."
                )
            elif event_count > 10:
                response += "🎯 This video has moderate activity with several highlight moments."
            else:
                response += "🎯 This video has selective key moments of interest."

            return response

        # How many
        elif "how many" in question_lower:
            return f"Total important moments: {len(events)}\n\nThese are spread from Frame {events[0]['frame']} to Frame {events[-1]['frame']}"

        # Key moments
        elif (
            "key" in question_lower
            or "important" in question_lower
            or "major" in question_lower
        ):
            top = sorted(events, key=lambda x: x["score"], reverse=True)[:5]
            response = f"🔥 TOP {len(top)} MOST DRAMATIC MOMENTS:\n\n"
            for i, e in enumerate(top, 1):
                response += f"{i}. Frame {e['frame']} - {e['desc']}\n"
            return response

        # Default
        response = f"Found {len(events)} important moments:\n\n"
        for e in events[:10]:
            response += f"Frame {e['frame']}: {e['desc']}\n"
        return response

    def summarize_session(self, events: List[Dict]) -> str:
        """
        Summarize a session of events.

        Args:
            events: List of event dictionaries

        Returns:
            Session summary
        """
        if not events:
            return "No events to summarize."

        surprise_scores = [e.get("surprise_score", 0) for e in events]
        avg_surprise = np.mean(surprise_scores)
        max_surprise = np.max(surprise_scores)

        high_surprise_events = [e for e in events if e.get("surprise_score", 0) > 0.5]

        summary = f"Session Summary:\n"
        summary += f"- Total events: {len(events)}\n"
        summary += f"- Average surprise: {avg_surprise:.3f}\n"
        summary += f"- Max surprise: {max_surprise:.3f}\n"
        summary += f"- Notable events: {len(high_surprise_events)}\n"

        if high_surprise_events:
            summary += f"\nMost notable moments:\n"
            for e in high_surprise_events[:3]:
                summary += f"- Frame {e.get('frame_index', '?')}: {e.get('description', 'N/A')}\n"

        return summary


def test_llm_reasoner():
    """Test function for standalone testing."""
    logger.info("Testing LLMReasoner...")

    # Test with fallback (since model may not exist yet)
    reasoner = LLMReasoner(use_fallback=True)

    # Test describe
    latent = np.random.randn(512).astype(np.float32)
    description = reasoner.describe_event(latent, 0.85)
    logger.info(f"Description (high surprise): {description}")

    description2 = reasoner.describe_event(latent, 0.15)
    logger.info(f"Description (low surprise): {description2}")

    # Test query
    context = (
        "Frame 10: Moderate change detected\nFrame 25: Significant motion observed"
    )
    answer = reasoner.query_memory("What happened in the session?", context)
    logger.info(f"Query answer: {answer}")

    # Test summarize
    events = [
        {"frame_index": 0, "surprise_score": 0.2, "description": "Routine"},
        {"frame_index": 10, "surprise_score": 0.7, "description": "Major change"},
        {"frame_index": 25, "surprise_score": 0.85, "description": "Critical event"},
    ]
    summary = reasoner.summarize_session(events)
    logger.info(f"Summary: {summary}")

    return reasoner


if __name__ == "__main__":
    test_llm_reasoner()
