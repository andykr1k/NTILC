"""
Query encoder for NTILC cluster retrieval.

Encodes natural language queries into 128-D embeddings for similarity search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional
from transformers import AutoModel, AutoTokenizer


class QueryEncoder(nn.Module):
    """Encodes natural language queries to 128-D embeddings."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3.5-9B",
        output_dim: int = 128,
        dropout: float = 0.15,
        torch_dtype: str = "float32"
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)
        self.dtype = dtype

        self.encoder = AutoModel.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        config = self.encoder.config
        hidden_dim = self._infer_hidden_dim(config, base_model)

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

    def _candidate_backbones(self):
        """Return likely transformer backbone modules, deduplicated."""
        modules = [self.encoder]
        for attr in ("language_model", "model", "transformer", "text_model", "backbone"):
            module = getattr(self.encoder, attr, None)
            if isinstance(module, nn.Module):
                modules.append(module)

        seen = set()
        unique_modules = []
        for module in modules:
            module_id = id(module)
            if module_id not in seen:
                seen.add(module_id)
                unique_modules.append(module)
        return unique_modules

    def _infer_hidden_dim(self, config: Any, model_name: str) -> int:
        """Infer hidden dimension from model config, including nested layouts."""
        def _search_obj(obj: Any) -> Optional[int]:
            for attr in ("hidden_size", "d_model", "n_embd"):
                dim = getattr(obj, attr, None)
                if isinstance(dim, int) and dim > 0:
                    return dim
            return None

        def _search_dict(data: dict) -> Optional[int]:
            for attr in ("hidden_size", "d_model", "n_embd"):
                dim = data.get(attr)
                if isinstance(dim, int) and dim > 0:
                    return dim
            return None

        dim = _search_obj(config)
        if dim:
            return dim

        for nested_attr in ("text_config", "language_config", "llm_config",
                            "model_config", "decoder", "encoder"):
            nested = getattr(config, nested_attr, None)
            if nested is not None:
                dim = _search_obj(nested)
                if dim:
                    return dim

        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
            dim = _search_dict(config_dict)
            if dim:
                return dim
            for nested_attr in ("text_config", "language_config", "llm_config",
                                "model_config", "decoder", "encoder"):
                nested_dict = config_dict.get(nested_attr)
                if isinstance(nested_dict, dict):
                    dim = _search_dict(nested_dict)
                    if dim:
                        return dim

        for backbone in self._candidate_backbones():
            backbone_config = getattr(backbone, "config", None)
            if backbone_config is not None:
                dim = _search_obj(backbone_config)
                if dim:
                    return dim
            for embed_attr in ("embed_tokens", "wte"):
                embed = getattr(backbone, embed_attr, None)
                if embed is not None and hasattr(embed, "weight"):
                    return int(embed.weight.shape[-1])

        raise ValueError(f"Could not determine hidden size for model: {model_name}")

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
