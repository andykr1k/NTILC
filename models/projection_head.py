"""
Projection Head for NTILC: Maps 1024-D embeddings to 128-D similarity space.

This projection head is optimized for similarity computation and metric learning.
It's not used for storage - only for similarity and loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projects embeddings from 1024-D to 128-D for similarity computation.
    
    Optimized with contrastive and Circle Loss objectives.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.15,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input embedding dimension (1024)
            output_dim: Output embedding dimension (128)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        
        # First projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Second projection
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to 128-D space.
        
        Args:
            embeddings: (batch_size, 1024) input embeddings
        
        Returns:
            projected: (batch_size, 128) projected embeddings
        """
        # Handle 1D input
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        projected = self.projection(embeddings)
        
        # Normalize to unit sphere for cosine similarity
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
