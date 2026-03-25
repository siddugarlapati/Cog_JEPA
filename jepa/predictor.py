"""
Predictor Module

Predicts next latent Z(t+1) from context embedding using a 3-layer MLP.
"""

import logging

import torch
import torch.nn as nn

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor(nn.Module):
    """
    3-layer MLP that predicts next latent vector from context.

    Architecture: 512 -> 512 -> 512 -> 512 with GELU and LayerNorm.
    """

    def __init__(self):
        super().__init__()

        self.latent_dim = config.latent_dim

        # First linear layer
        self.fc1 = nn.Linear(config.latent_dim, config.latent_dim)
        self.norm1 = nn.LayerNorm(config.latent_dim)

        # Second linear layer
        self.fc2 = nn.Linear(config.latent_dim, config.latent_dim)
        self.norm2 = nn.LayerNorm(config.latent_dim)

        # Third linear layer (output)
        self.fc3 = nn.Linear(config.latent_dim, config.latent_dim)

        self.activation = nn.GELU()

        logger.info(
            f"Predictor initialized: 3-layer MLP with {config.latent_dim} hidden dims"
        )

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent vector from context.

        Args:
            context_embedding: Context embedding of shape [latent_dim]
                             or [batch, latent_dim]

        Returns:
            Predicted next latent Z_hat of shape [latent_dim] or [batch, latent_dim]
        """
        # Handle both single and batch input
        squeeze = False
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)
            squeeze = True

        # First layer
        x = self.fc1(context_embedding)
        x = self.norm1(x)
        x = self.activation(x)

        # Second layer
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Third layer (output)
        x = self.fc3(x)

        if squeeze:
            x = x.squeeze(0)

        return x

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


def test_predictor():
    """Test function for standalone testing."""
    logger.info("Testing Predictor...")

    predictor = Predictor()
    predictor = predictor.to("mps" if torch.backends.mps.is_available() else "cpu")
    predictor.eval()

    # Test with random context embedding
    test_input = torch.randn(config.d_model, dtype=torch.float16)
    test_input = test_input.to(predictor.device)

    with torch.no_grad():
        output = predictor(test_input)

    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test with batch
    test_batch = torch.randn(4, config.d_model, dtype=torch.float16)
    test_batch = test_batch.to(predictor.device)

    with torch.no_grad():
        batch_output = predictor(test_batch)

    logger.info(f"Batch input shape: {test_batch.shape}")
    logger.info(f"Batch output shape: {batch_output.shape}")

    return output


if __name__ == "__main__":
    test_predictor()
