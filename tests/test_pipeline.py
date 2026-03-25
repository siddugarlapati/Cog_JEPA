"""
COG-JEPA Test Suite

Pytest tests for each component.
"""

import pytest
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_encoder_output_shape():
    """Test that encoder produces 512-dim float32 output."""
    from encoder.smolvlm_encoder import SmolVLMEncoder

    # Create random image
    random_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    encoder = SmolVLMEncoder()
    latent = encoder.encode(random_image)

    assert latent.shape == (512,), f"Expected (512,), got {latent.shape}"
    assert latent.dtype == np.float32, f"Expected float32, got {latent.dtype}"


def test_surprise_score_range():
    """Test that surprise scores are in valid range."""
    from jepa.surprise import SurpriseComputer

    computer = SurpriseComputer(ema_alpha=0.1)

    # Test with identical vectors (should be 0)
    z1 = np.random.randn(512).astype(np.float32)
    z2 = z1.copy()

    surprise, l2 = computer.compute_surprise(z1, z2)
    assert 0 <= surprise <= 2, f"Surprise {surprise} out of range [0, 2]"

    # Test with different vectors
    z3 = np.random.randn(512).astype(np.float32)
    surprise2, l2_2 = computer.compute_surprise(z1, z3)
    assert 0 <= surprise2 <= 2, f"Surprise {surprise2} out of range [0, 2]"

    # High surprise should be higher than low surprise
    assert surprise2 > surprise, "Different vectors should have higher surprise"


def test_memory_gate_threshold():
    """Test memory gate stores high surprise and discards low."""
    from memory.memory_gate import MemoryGate

    gate = MemoryGate(base_threshold=0.3, ema_alpha=0.1)

    # Low surprise should be discarded
    for _ in range(5):
        result = gate.should_store(0.1)
        assert result == False, "Low surprise should be discarded"

    # Update EMA to allow some storage
    gate.ema_baseline = 0.5

    # High surprise should be stored
    result = gate.should_store(0.8)
    assert result == True, "High surprise should be stored"

    # Check statistics
    stats = gate.get_statistics()
    assert stats["stored_frames"] >= 1
    assert stats["discarded_frames"] >= 5


def test_context_window_rolling():
    """Test context window stays at size 8 after 20 frames."""
    from collections import deque
    from config import config

    context_window = deque(maxlen=config.context_window_size)

    # Add 20 frames
    for i in range(20):
        latent = np.random.randn(config.latent_dim).astype(np.float32)
        context_window.append(latent)

    # Window should be full (8)
    assert len(context_window) == config.context_window_size

    # Oldest should be gone
    oldest = context_window[0]

    # Add more frames
    for i in range(5):
        latent = np.random.randn(config.latent_dim).astype(np.float32)
        context_window.append(latent)

    # Still 8
    assert len(context_window) == config.context_window_size


def test_cognee_store_async():
    """Test cognee store add and query round trip."""
    import asyncio
    from datetime import datetime
    from memory.cognee_store import CogneeMemoryStore

    store = CogneeMemoryStore()

    async def test_async():
        # Add events
        for i in range(5):
            await store.add_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "frame_index": i,
                    "surprise_score": 0.5 + i * 0.1,
                    "description": f"Test event {i}",
                    "latent_hash": f"hash_{i}",
                    "context_window": [0.1, 0.2, 0.3],
                }
            )

        # Query
        result = await store.query("What happened?")
        assert len(result) > 0

        # Get recent
        recent = await store.get_recent_events(n=3)
        assert len(recent) == 3

        # Summary
        summary = await store.get_summary()
        assert "5" in summary  # Should mention 5 events

    asyncio.run(test_async())


def test_full_pipeline_synthetic():
    """Test full pipeline with 50 synthetic frames."""
    from pipeline.video_pipeline import VideoPipeline

    pipeline = VideoPipeline(
        video_source="test",
        use_llm=False,  # Skip LLM for faster test
    )

    # Run 50 frames
    pipeline.run(max_frames=50)

    # Check results
    stats = pipeline.memory_gate.get_statistics()

    assert pipeline.frame_count == 50
    assert stats["total_frames"] == 50
    assert stats["stored_frames"] > 0, "Should store at least some events"

    # Check memory store has events
    events = pipeline.memory_store.get_all_events()
    assert len(events) > 0


def test_config_defaults():
    """Test config has expected defaults."""
    from config import config

    assert config.latent_dim == 512
    assert config.context_window_size == 8
    assert config.d_model == 512
    assert config.nhead == 8
    assert config.base_surprise_threshold == 0.3


def test_predictor_output_shape():
    """Test predictor produces correct output shape."""
    import torch
    from jepa.predictor import Predictor

    predictor = Predictor()
    predictor = predictor.to("cpu")
    predictor.eval()

    # Single input
    test_input = torch.randn(512, dtype=torch.float16)

    with torch.no_grad():
        output = predictor(test_input)

    assert output.shape == (512,), f"Expected (512,), got {output.shape}"


def test_context_encoder_output_shape():
    """Test context encoder produces correct output shape."""
    import torch
    from jepa.context_encoder import ContextEncoder
    from config import config

    encoder = ContextEncoder()
    encoder = encoder.to("cpu")
    encoder.eval()

    # Full context window
    test_input = torch.randn(
        config.context_window_size, config.latent_dim, dtype=torch.float16
    )

    with torch.no_grad():
        output = encoder(test_input)

    assert output.shape == (config.latent_dim,), (
        f"Expected ({config.latent_dim},), got {output.shape}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
