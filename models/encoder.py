"""
Encoder module for NTILC: Transforms tool invocation strings into continuous embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
from typing import Optional


class ToolInvocationEncoder(nn.Module):
    """
    Encoder that maps tool invocation strings to d-dimensional embeddings.

    Architecture:
    - Tokenize input string
    - Transformer encoder processes tokens
    - Pool hidden states (mean/attention/CLS)
    - Project to d-dimensional embedding
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        embedding_dim: int = 256,
        pooling_strategy: str = "attention",
        dropout: float = 0.15,
        freeze_base: bool = False,
        freeze_layers: int = 0,
        torch_dtype: str = "bfloat16",
        max_length: int = 256
    ):
        """
        Args:
            model_name: HuggingFace model name for base encoder
            embedding_dim: Dimension of output embedding (d)
            pooling_strategy: One of ["mean", "cls", "max", "attention"]
            dropout: Dropout rate
            freeze_base: Whether to freeze all base transformer weights
            freeze_layers: Number of early layers to freeze (0 = none, -1 = all)
            torch_dtype: Data type for model weights
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length

        # Load pretrained encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        self.dtype = dtype

        # Load T5 encoder
        if 't5' in model_name.lower() or 'flan-t5' in model_name.lower():
            self.transformer = T5EncoderModel.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
        else:
            from transformers import AutoModelForSeq2SeqLM
            full_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
            if hasattr(full_model, 'encoder'):
                self.transformer = full_model.encoder
            elif hasattr(full_model, 'model') and hasattr(full_model.model, 'encoder'):
                self.transformer = full_model.model.encoder
            else:
                raise ValueError(f"Could not extract encoder from {model_name}")
            del full_model

        # Apply freezing strategy
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

        # Get hidden dimension
        if hasattr(config, 'd_model'):
            hidden_dim = config.d_model
        elif hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'n_embd'):
            hidden_dim = config.n_embd
        else:
            raise ValueError(f"Could not determine hidden dimension for model {model_name}")

        self.hidden_dim = hidden_dim

        # Pooling layer (if attention pooling)
        if pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.attention_pool = None

        # Projection to embedding dimension with residual connections
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Initialize projection layers properly
        self._init_projection_weights()

        # Convert to correct dtype
        if self.attention_pool is not None:
            self.attention_pool = self.attention_pool.to(dtype)
        self.projection = self.projection.to(dtype)

    def _freeze_early_layers(self, num_layers: int):
        """Freeze the first num_layers of the transformer."""
        # Freeze embeddings
        if hasattr(self.transformer, 'embed_tokens'):
            for param in self.transformer.embed_tokens.parameters():
                param.requires_grad = False
        
        # Freeze early encoder blocks
        if hasattr(self.transformer, 'block'):
            for i, block in enumerate(self.transformer.block):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
                    print(f"Froze encoder layer {i}")

    def _init_projection_weights(self):
        """Initialize projection weights for stable training."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tool_calls: list[str]) -> torch.Tensor:
        """
        Encode tool invocation strings to embeddings.
        """
        # Tokenize
        encoded = self.tokenizer(
            tool_calls,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get transformer outputs
        outputs = self.transformer(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]

        # Pool hidden states
        pooled = self._pool(hidden_states, attention_mask)

        # Convert to correct dtype before projection
        pooled = pooled.to(self.dtype)

        # Project to embedding dimension
        embeddings = self.projection(pooled)
        
        # Normalize embeddings to unit sphere
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence of hidden states into single vector."""
        mask_dtype = hidden_states.dtype
        
        if self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "max":
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            masked_hidden = hidden_states * mask_expanded - (1 - mask_expanded) * 1e9
            return masked_hidden.max(dim=1)[0]

        elif self.pooling_strategy == "attention":
            attention_weights = self.attention_pool(hidden_states)
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            attention_weights = attention_weights * mask_expanded
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
            pooled = (hidden_states * attention_weights).sum(dim=1)
            return pooled

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def encode_string(self, tool_call: str) -> torch.Tensor:
        """Convenience method to encode a single tool call string."""
        self.eval()
        with torch.no_grad():
            embedding = self.forward([tool_call])[0]
        return embedding
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
