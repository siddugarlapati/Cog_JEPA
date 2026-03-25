"""
COG-JEPA Configuration Module

All hyperparameters and paths in one place.
"""

from dataclasses import dataclass
import os


@dataclass
class Config:
    """Main configuration class for COG-JEPA system."""

    # Encoder
    smolvlm_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    frame_size: tuple = (224, 224)
    latent_dim: int = 512

    # JEPA
    context_window_size: int = 8
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 1024
    num_layers: int = 2

    # Surprise
    base_surprise_threshold: float = 0.3
    ema_alpha: float = 0.1

    # LLM
    llm_model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 4096

    # Paths
    db_path: str = "data/cognee_memory.db"
    log_path: str = "data/session.log"
    models_dir: str = "models"
    data_dir: str = "data"

    # Pipeline
    target_fps: int = 10
    online_learning_lr: float = 1e-4
    grad_clip_norm: float = 1.0

    # Device
    device: str = "mps"

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path) or "data", exist_ok=True)


config = Config()
