"""
Projection Head for NTILC: Maps 1024-D embeddings to 128-D similarity space.

This projection head is used only for similarity computation and metric learning,
not for storage. Outputs are L2-normalized for cosine similarity objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projects embeddings from 1024-D to 128-D for similarity computation.

    Optimized for contrastive and Circle Loss objectives.
    Uses LayerNorm instead of BatchNorm1d so that:
      - batch size 1 is safe (BatchNorm1d errors on single-sample batches)
      - per-sample normalization is consistent with metric learning objectives
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ):
        """
        Args:
            input_dim: Input embedding dimension (1024)
            output_dim: Output embedding dimension (128)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # FIX: Use LayerNorm instead of BatchNorm1d.
        # BatchNorm1d requires batch_size > 1 during training and interacts
        # poorly with metric learning — it normalizes across the batch, which
        # can interfere with the similarity structure the loss is trying to
        # build. LayerNorm normalizes per sample and works at any batch size.
        #
        # FIX: Do not add a norm layer before the final F.normalize call.
        # A norm layer immediately before L2 normalization is redundant:
        # the norm layer scales activations, then normalize discards the scale.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            # No LayerNorm here — F.normalize in forward handles unit-norm output
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to 128-D unit-normalized space.

        Args:
            embeddings: (batch_size, input_dim) or (input_dim,) input embeddings

        Returns:
            projected: (batch_size, output_dim) L2-normalized projected embeddings
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        projected = self.projection(embeddings)

        # L2-normalize for cosine similarity
        projected = F.normalize(projected, p=2, dim=-1)

        return projected

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
