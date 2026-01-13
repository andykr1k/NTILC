"""
Encoder module for NTILC: Transforms tool invocation strings into continuous embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


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
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embedding_dim: int = 512,
        pooling_strategy: str = "attention",
        dropout: float = 0.1,
        freeze_base: bool = False,
        torch_dtype: str = "float32"
    ):
        """
        Args:
            model_name: HuggingFace model name for base transformer
            embedding_dim: Dimension of output embedding (d)
            pooling_strategy: One of ["mean", "cls", "max", "attention"]
            dropout: Dropout rate
            freeze_base: Whether to freeze base transformer weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy

        # Load pretrained transformer
        # Qwen2.5 is decoder-only, so we use AutoModelForCausalLM and access transformer
        from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Determine if it's a decoder-only model (Qwen, GPT) or encoder model (BERT)
        is_decoder_only = (
            'qwen' in model_name.lower() or
            'gpt' in model_name.lower() or
            hasattr(config, 'n_embd')  # GPT-2 style
        )

        if is_decoder_only:
            # Decoder-only model - load only the base model, not the full CausalLM (saves memory)
            # Use AutoModel to avoid loading the LM head we don't need
            try:
                # Try loading base model directly (Qwen2.5 supports this)
                self.transformer = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
            except:
                # Fallback: load CausalLM but only keep transformer
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
                self.transformer = model.model  # Qwen2.5 uses 'model' attribute
                if not hasattr(self.transformer, 'layers'):
                    # Try 'transformer' if 'model' doesn't work (GPT-2 style)
                    self.transformer = model.transformer
                # Delete the full model to free memory (LM head not needed)
                del model
        else:
            # Encoder model (BERT-style)
            self.transformer = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype
            )

        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Get hidden dimension
        if hasattr(config, 'hidden_size'):
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
        outputs = self.transformer(**encoded)
        # (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]  # (batch_size, seq_len)

        # Pool hidden states
        pooled = self._pool(hidden_states, attention_mask)

        # Project to embedding dimension
        embeddings = self.projection(pooled)

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
        if self.pooling_strategy == "mean":
            # Mean pooling (masked)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "max":
            # Max pooling (masked)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Set padding to very negative value
            masked_hidden = hidden_states * \
                mask_expanded - (1 - mask_expanded) * 1e9
            return masked_hidden.max(dim=1)[0]

        elif self.pooling_strategy == "attention":
            # Attention-weighted pooling
            attention_weights = self.attention_pool(
                hidden_states)  # (batch_size, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).float()
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
