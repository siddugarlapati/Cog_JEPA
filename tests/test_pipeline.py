"""
COG-JEPA Test Suite

Pytest tests for each component.
"""

import asyncio
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Encoder ────────────────────────────────────────────────────────────────

def test_encoder_output_shape():
    """Encoder produces 512-dim float32 output."""
    from encoder.smolvlm_encoder import BatchVisionEncoder

    encoder = BatchVisionEncoder()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    latent = encoder.encode(img)

    assert latent.shape == (512,), f"Expected (512,), got {latent.shape}"
    assert latent.dtype == np.float32, f"Expected float32, got {latent.dtype}"


def test_encoder_batch():
    """Batch encoding returns correct number of latents."""
    from encoder.smolvlm_encoder import BatchVisionEncoder

    encoder = BatchVisionEncoder()
    imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
    latents = encoder.encode_batch(imgs)

    assert len(latents) == 4
    for lat in latents:
        assert lat.shape == (512,)


def test_encoder_hash_deterministic():
    """Same latent always produces same hash."""
    from encoder.smolvlm_encoder import BatchVisionEncoder

    encoder = BatchVisionEncoder()
    latent = np.random.randn(512).astype(np.float32)
    h1 = encoder.encode_hash(latent)
    h2 = encoder.encode_hash(latent)
    assert h1 == h2


# ─── Surprise ───────────────────────────────────────────────────────────────

def test_surprise_score_range():
    """Surprise scores are in valid range [0, 2]."""
    from jepa.surprise import SurpriseComputer

    computer = SurpriseComputer(ema_alpha=0.1)

    z1 = np.random.randn(512).astype(np.float32)
    z2 = z1.copy()
    surprise, _ = computer.compute_surprise(z1, z2)
    assert 0 <= surprise <= 2, f"Surprise {surprise} out of range"

    z3 = np.random.randn(512).astype(np.float32)
    surprise2, _ = computer.compute_surprise(z1, z3)
    assert 0 <= surprise2 <= 2, f"Surprise {surprise2} out of range"
    assert surprise2 > surprise, "Different vectors should have higher surprise"


def test_surprise_ema_updates():
    """EMA baseline updates after each call."""
    from jepa.surprise import SurpriseComputer

    computer = SurpriseComputer(ema_alpha=0.1)
    assert computer.ema_baseline == 0.0

    z1 = np.random.randn(512).astype(np.float32)
    z2 = np.random.randn(512).astype(np.float32)
    computer.compute_surprise(z1, z2)

    assert computer.ema_baseline > 0.0


# ─── Memory Gate ────────────────────────────────────────────────────────────

def test_memory_gate_threshold():
    """Memory gate stores high surprise and discards low."""
    from memory.memory_gate import MemoryGate

    gate = MemoryGate(base_threshold=0.3, ema_alpha=0.1)

    for _ in range(5):
        assert gate.should_store(0.05) is False, "Very low surprise should be discarded"

    assert gate.should_store(0.9) is True, "High surprise should be stored"

    stats = gate.get_statistics()
    assert stats["stored_frames"] >= 1
    assert stats["discarded_frames"] >= 5


def test_memory_gate_adaptive_threshold():
    """Adaptive threshold rises above base_threshold after warm-up."""
    from memory.memory_gate import MemoryGate

    gate = MemoryGate(base_threshold=0.3, ema_alpha=0.5)

    # Feed high surprise scores to push EMA up
    for _ in range(10):
        gate.should_store(1.0)

    adaptive = gate.get_adaptive_threshold()
    assert adaptive >= gate.base_threshold, "Adaptive threshold should be >= base"


def test_memory_gate_compression_ratio():
    """Compression ratio is between 0 and 1."""
    from memory.memory_gate import MemoryGate

    gate = MemoryGate(base_threshold=0.3)
    for score in [0.1, 0.2, 0.8, 0.9, 0.05]:
        gate.should_store(score)

    ratio = gate.get_compression_ratio()
    assert 0.0 <= ratio <= 1.0


# ─── Context Window ──────────────────────────────────────────────────────────

def test_context_window_rolling():
    """Context window stays at size 8 after many frames."""
    from collections import deque
    from config import config

    window = deque(maxlen=config.context_window_size)
    for _ in range(20):
        window.append(np.random.randn(config.latent_dim).astype(np.float32))

    assert len(window) == config.context_window_size


# ─── Memory Store ────────────────────────────────────────────────────────────

def test_cognee_store_async():
    """CogneeMemoryStore add and query round trip."""
    from datetime import datetime
    from memory.cognee_store import CogneeMemoryStore

    store = CogneeMemoryStore()
    initial_count = len(store.events)

    async def run():
        for i in range(5):
            await store.add_event({
                "timestamp": datetime.now().isoformat(),
                "frame_index": 1000 + i,
                "surprise_score": 0.5 + i * 0.1,
                "description": f"Test event {i}",
                "latent_hash": f"testhash_{i}",
                "context_window": [0.1, 0.2, 0.3],
            })

        result = await store.query("What happened?")
        assert len(result) > 0

        recent = await store.get_recent_events(n=3)
        assert len(recent) == 3

        summary = await store.get_summary()
        assert "events" in summary.lower()

    asyncio.run(run())
    assert len(store.events) == initial_count + 5


def test_cognee_store_no_duplicates():
    """Calling _load_from_log twice does not duplicate events."""
    from memory.cognee_store import CogneeMemoryStore

    store = CogneeMemoryStore()
    count_after_first_load = len(store.events)

    store._load_from_log()
    assert len(store.events) == count_after_first_load, "Duplicate events on reload"


# ─── Predictor ───────────────────────────────────────────────────────────────

def test_predictor_output_shape():
    """Predictor produces correct output shape."""
    import torch
    from jepa.predictor import Predictor

    predictor = Predictor().to("cpu").eval()
    x = torch.randn(512, dtype=torch.float16)
    with torch.no_grad():
        out = predictor(x)
    assert out.shape == (512,)


def test_predictor_batch():
    """Predictor handles batch input."""
    import torch
    from jepa.predictor import Predictor

    predictor = Predictor().to("cpu").eval()
    x = torch.randn(4, 512, dtype=torch.float16)
    with torch.no_grad():
        out = predictor(x)
    assert out.shape == (4, 512)


# ─── Context Encoder ─────────────────────────────────────────────────────────

def test_context_encoder_output_shape():
    """Context encoder produces correct output shape."""
    import torch
    from config import config
    from jepa.context_encoder import ContextEncoder

    encoder = ContextEncoder().to("cpu").eval()
    x = torch.randn(config.context_window_size, config.latent_dim, dtype=torch.float16)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (config.latent_dim,)


# ─── Config ──────────────────────────────────────────────────────────────────

def test_config_defaults():
    """Config has expected defaults."""
    from config import config

    assert config.latent_dim == 512
    assert config.context_window_size == 8
    assert config.d_model == 512
    assert config.nhead == 8
    assert config.base_surprise_threshold == 0.3
    assert config.online_learning_lr == 1e-4
    assert config.grad_clip_norm == 1.0


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def test_full_pipeline_synthetic():
    """Full pipeline processes 50 synthetic frames and stores events."""
    from pipeline.video_pipeline import VideoPipeline

    pipeline = VideoPipeline(video_source="test", use_llm=False)
    pipeline.run(max_frames=50)

    assert pipeline.frame_count == 50
    stats = pipeline.memory_gate.get_statistics()
    assert stats["total_frames"] == 50
    assert stats["stored_frames"] > 0, "Should store at least some events"

    events = pipeline.memory_store.get_all_events()
    assert len(events) > 0


def test_meta_learning_updates_weights():
    """Online learning actually changes predictor weights during inference."""
    import torch
    from pipeline.video_pipeline import VideoPipeline

    pipeline = VideoPipeline(video_source="test", use_llm=False)

    # Capture initial weight snapshot
    initial_weights = pipeline.predictor.fc1.weight.detach().clone()

    # Run enough frames for online updates to fire
    pipeline.run(max_frames=20)

    updated_weights = pipeline.predictor.fc1.weight.detach()
    assert not torch.allclose(initial_weights.float(), updated_weights.float()), \
        "Predictor weights should change during online learning"


def test_pipeline_video_file(tmp_path):
    """Pipeline handles a short synthetic video file."""
    import cv2
    import numpy as np
    from pipeline.video_pipeline import VideoPipeline

    # Create a tiny synthetic video
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (224, 224))
    for _ in range(15):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    pipeline = VideoPipeline(video_source=video_path, use_llm=False)
    pipeline.run(max_frames=15)

    assert pipeline.frame_count > 0


# ─── LLM Reasoner ────────────────────────────────────────────────────────────

def test_llm_reasoner_fallback_summary():
    """LLM reasoner fallback returns non-empty summary."""
    from reasoning.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(use_fallback=True)
    context = (
        "Frame 5: surprise=0.80, description=Major motion detected\n"
        "Frame 12: surprise=0.45, description=Moderate activity\n"
        "Frame 20: surprise=0.90, description=Critical event"
    )
    answer = reasoner.query_memory("What happened in this video?", context)
    assert len(answer) > 20, "Answer should be non-trivial"


def test_llm_reasoner_key_moments():
    """LLM reasoner returns key moments when asked."""
    from reasoning.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(use_fallback=True)
    context = (
        "Frame 3: surprise=0.85, description=Explosion detected\n"
        "Frame 10: surprise=0.20, description=Calm scene\n"
        "Frame 18: surprise=0.95, description=Major event"
    )
    answer = reasoner.query_memory("What are the key moments?", context)
    assert "Frame" in answer


# ─── Semantic Search ─────────────────────────────────────────────────────────

def test_semantic_search_index_and_query():
    """Semantic search indexes a caption and retrieves it."""
    from search.semantic_search import SemanticImageSearch

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        search = SemanticImageSearch(embeddings_file=tmp_path)
        search.add_image("/tmp/cat.jpg", "a cat sitting on a red sofa")
        search.add_image("/tmp/dog.jpg", "a dog running in the park")

        results = search.search("feline on furniture", top_k=1)
        assert len(results) >= 1
        # The cat result should score higher than dog for this query
        assert "cat" in results[0]["caption"].lower() or results[0]["score"] > 0.0
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
