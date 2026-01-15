"""
Encoder module for NTILC: Transforms tool invocation strings into continuous embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig


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
        embedding_dim: int = 512,
        pooling_strategy: str = "attention",
        dropout: float = 0.1,
        freeze_base: bool = False,
        torch_dtype: str = "bfloat16"
    ):
        """
        Args:
            model_name: HuggingFace model name for base encoder (should be encoder-decoder like T5)
            embedding_dim: Dimension of output embedding (d)
            pooling_strategy: One of ["mean", "cls", "max", "attention"]
            dropout: Dropout rate
            freeze_base: Whether to freeze base transformer weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy

        # Load pretrained encoder-decoder model and extract encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Store dtype for later use
        self.dtype = dtype

        # For T5/encoder-decoder models, load the encoder part
        # T5 models have an encoder attribute
        try:
            # Try loading T5 encoder directly
            if 't5' in model_name.lower() or 'flan-t5' in model_name.lower():
                self.transformer = T5EncoderModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
            else:
                # For other encoder-decoder models, try to get encoder
                from transformers import AutoModelForSeq2SeqLM
                full_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
                # Extract encoder from the model
                if hasattr(full_model, 'encoder'):
                    self.transformer = full_model.encoder
                elif hasattr(full_model, 'model') and hasattr(full_model.model, 'encoder'):
                    self.transformer = full_model.model.encoder
                else:
                    raise ValueError(f"Could not extract encoder from {model_name}")
                # Delete full model to save memory
                del full_model
        except Exception as e:
            raise ValueError(
                f"Failed to load encoder from {model_name}: {e}. "
                "Please use an encoder-decoder model like T5 or BART."
            )

        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Get hidden dimension
        if hasattr(config, 'd_model'):
            hidden_dim = config.d_model  # T5 uses d_model
        elif hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'n_embd'):
            hidden_dim = config.n_embd
        else:
            raise ValueError(
                f"Could not determine hidden dimension for model {model_name}")

        # Pooling layer (if attention pooling)
        if pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )


        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.attention_pool = self.attention_pool.to(dtype)
        self.projection = self.projection.to(dtype)

    def forward(self, tool_calls: list[str]) -> torch.Tensor:
        """
        Encode tool invocation strings to embeddings.

        Args:
            tool_calls: List of tool invocation strings

        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            tool_calls,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get transformer outputs
        # For T5 encoder, we pass input_ids and attention_mask
        outputs = self.transformer(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )
        # (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]  # (batch_size, seq_len)

        # Pool hidden states
        pooled = self._pool(hidden_states, attention_mask)

        # Convert pooled tensor to correct dtype before projection
        pooled = pooled.to(self.dtype)

        # Project to embedding dimension
        embeddings = self.projection(pooled)
        
        # Clamp to prevent extreme values that could cause NaN
        embeddings = torch.clamp(embeddings, min=-10.0, max=10.0)

        return embeddings

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence of hidden states into single vector.

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # Get dtype from hidden_states to maintain consistency
        mask_dtype = hidden_states.dtype
        
        if self.pooling_strategy == "mean":
            # Mean pooling (masked)
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "max":
            # Max pooling (masked)
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            # Set padding to very negative value
            masked_hidden = hidden_states * \
                mask_expanded - (1 - mask_expanded) * 1e9
            return masked_hidden.max(dim=1)[0]

        elif self.pooling_strategy == "attention":
            # Attention-weighted pooling
            attention_weights = self.attention_pool(
                hidden_states)  # (batch_size, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            attention_weights = attention_weights * mask_expanded
            attention_weights = attention_weights / \
                (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
            pooled = (hidden_states * attention_weights).sum(dim=1)
            return pooled

        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling_strategy}")

    def encode_string(self, tool_call: str) -> torch.Tensor:
        """
        Convenience method to encode a single tool call string.

        Args:
            tool_call: Single tool invocation string

        Returns:
            embedding: Tensor of shape (embedding_dim,)
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward([tool_call])[0]
        return embedding
