"""
Context Encoder Module

Encodes past context window using a lightweight Transformer.
"""

import logging
from typing import List

import torch
import torch.nn as nn

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """
    Lightweight Transformer encoder for context representation.

    Takes a sequence of N latent vectors and outputs a single context embedding.
    """

    def __init__(self):
        super().__init__()

        self.d_model = config.d_model
        self.nhead = config.nhead
        self.dim_feedforward = config.dim_feedforward
        self.num_layers = config.num_layers
        self.context_window_size = config.context_window_size

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, config.context_window_size, config.d_model) * 0.1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            dtype=torch.float16,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        logger.info(
            f"ContextEncoder initialized: d_model={config.d_model}, "
            f"nhead={config.nhead}, layers={config.num_layers}"
        )

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask to prevent attending to future tokens."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device),
            diagonal=1,
        )
        return mask

    def forward(self, context_window: torch.Tensor) -> torch.Tensor:
        """
        Encode context window to context embedding.

        Args:
            context_window: Tensor of shape [context_size, latent_dim]
                          or [batch, context_size, latent_dim]

        Returns:
            Context embedding of shape [latent_dim] or [batch, latent_dim]
        """
        # Handle both single sequence and batch
        if context_window.dim() == 2:
            # Single sequence: [context_size, latent_dim]
            context_window = context_window.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = context_window.shape

        # Add positional encoding
        pos_enc = self.pos_encoder[:, :seq_len, :]
        context_window = context_window + pos_enc

        # Create causal mask
        causal_mask = self._get_causal_mask(seq_len)

        # Apply transformer
        encoded = self.transformer(context_window, mask=causal_mask)

        # Mean pool all positions for context representation
        context_embedding = encoded.mean(dim=1)

        # Project to output dimension
        context_embedding = self.output_proj(context_embedding)

        if squeeze_output:
            context_embedding = context_embedding.squeeze(0)

        return context_embedding

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device


def test_context_encoder():
    """Test function for standalone testing."""
    logger.info("Testing ContextEncoder...")

    encoder = ContextEncoder()
    encoder = encoder.to("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.eval()

    # Test with context window of 8 vectors
    test_input = torch.randn(
        config.context_window_size, config.d_model, dtype=torch.float16
    )
    test_input = test_input.to(encoder.device)

    with torch.no_grad():
        output = encoder(test_input)

    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test rolling behavior (multiple calls)
    for i in range(3):
        test_input = torch.randn(
            config.context_window_size, config.d_model, dtype=torch.float16
        )
        test_input = test_input.to(encoder.device)
        with torch.no_grad():
            output = encoder(test_input)
        logger.info(f"Call {i + 1} output shape: {output.shape}")

    return output


if __name__ == "__main__":
    test_context_encoder()
