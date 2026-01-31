"""
Query encoder for NTILC cluster retrieval.

Encodes natural language queries into 128-D embeddings for similarity search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


class QueryEncoder(nn.Module):
    """Encodes natural language queries to 128-D embeddings."""

    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        output_dim: int = 128,
        dropout: float = 0.15,
        torch_dtype: str = "float32"
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)
        self.dtype = dtype

        self.encoder = T5EncoderModel.from_pretrained(base_model, torch_dtype=dtype)
        config = self.encoder.config
        hidden_dim = config.d_model if hasattr(config, "d_model") else config.hidden_size

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Projection to output_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.attention_pool = self.attention_pool.to(dtype)
        self.projection = self.projection.to(dtype)

    def forward(self, input_ids, attention_mask):
        """Encode query to output_dim embedding."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        attention_weights = self.attention_pool(hidden_states)
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        attention_weights = attention_weights * mask_expanded
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (hidden_states * attention_weights).sum(dim=1)

        projected = self.projection(pooled)
        projected = F.normalize(projected, p=2, dim=1)
        return projected
