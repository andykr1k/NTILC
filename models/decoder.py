"""
Decoder module for NTILC: Transforms continuous embeddings back into tool invocation strings.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ToolInvocationDecoder(nn.Module):
    """
    Decoder that maps d-dimensional embeddings to tool invocation strings.

    Architecture:
    - Project embedding to decoder hidden dimension
    - Use as initial hidden state for autoregressive transformer
    - Generate tokens autoregressively
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = False,
        torch_dtype: str = "float32",
        use_gradient_checkpointing: bool = False
    ):
        """
        Args:
            embedding_dim: Dimension of input embedding (d)
            model_name: HuggingFace model name for base decoder
            max_length: Maximum generation length
            dropout: Dropout rate
            freeze_base: Whether to freeze base transformer weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Load pretrained decoder (supports GPT-2, GPT-Neo, OPT, etc.)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        # Enable gradient checkpointing to save memory (trades compute for memory)
        if use_gradient_checkpointing:
            if hasattr(self.decoder, 'gradient_checkpointing_enable'):
                self.decoder.gradient_checkpointing_enable()
            elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'gradient_checkpointing_enable'):
                self.decoder.model.gradient_checkpointing_enable()

        if freeze_base:
            for param in self.decoder.parameters():
                param.requires_grad = False

        # Get hidden dimension from decoder (different models use different attribute names)
        if hasattr(self.decoder.config, 'n_embd'):  # GPT-2 style
            hidden_dim = self.decoder.config.n_embd
        elif hasattr(self.decoder.config, 'hidden_size'):  # BERT/OPT style
            hidden_dim = self.decoder.config.hidden_size
        elif hasattr(self.decoder.config, 'd_model'):  # T5 style
            hidden_dim = self.decoder.config.d_model
        else:
            raise ValueError(
                f"Could not determine hidden dimension for model {model_name}")

        # Projection from embedding to decoder hidden dimension
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Initialize projection layers properly
        for module in self.embedding_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Learnable initial token embedding (replaces start token)
        self.initial_token_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        target_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        Decode embeddings to tool invocation tokens.

        Args:
            embeddings: (batch_size, embedding_dim)
            target_ids: (batch_size, seq_len) - for training
            attention_mask: (batch_size, seq_len) - for training

        Returns:
            Dictionary with:
                - logits: (batch_size, seq_len, vocab_size) - for training
                - generated_ids: (batch_size, seq_len) - for inference
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Project embedding to decoder hidden dimension
        projected = self.embedding_projection(
            embeddings)  # (batch_size, hidden_dim)

        if self.training and target_ids is not None:
            # Training: teacher forcing
            return self._forward_train(projected, target_ids, attention_mask)
        else:
            # Inference: autoregressive generation
            return self._forward_inference(projected)

    def _forward_train(
        self,
        projected_emb: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        Uses the projected embedding to condition the first token prediction.
        """
        batch_size = projected_emb.shape[0]
        device = projected_emb.device
        seq_len = target_ids.shape[1]

        # For teacher forcing: input is target_ids[:-1], we predict target_ids[1:]
        input_ids = target_ids[:, :-1]  # (batch_size, seq_len-1)
        target_mask = attention_mask[:, 1:]  # (batch_size, seq_len-1)

        # Get token embeddings for input sequence
        # Different models have different embedding layers
        if hasattr(self.decoder, 'transformer'):  # GPT-2 style
            wte = self.decoder.transformer.wte
        elif hasattr(self.decoder, 'model'):  # OPT/GPT-Neo style
            wte = self.decoder.model.embed_tokens
        elif hasattr(self.decoder, 'gpt_neox'):  # GPT-NeoX style
            wte = self.decoder.gpt_neox.embed_in
        else:
            # Fallback: use model's embedding layer
            wte = self.decoder.get_input_embeddings()

        # (batch_size, seq_len-1, hidden_dim)
        token_embeddings = wte(input_ids)

        # Add projected embedding to the first token's embedding
        # This conditions the generation on the embedding
        token_embeddings[:, 0, :] = token_embeddings[:, 0, :] + projected_emb

        # Forward through decoder transformer
        # Different models have different transformer attribute names
        if hasattr(self.decoder, 'transformer'):  # GPT-2 style
            transformer = self.decoder.transformer
        elif hasattr(self.decoder, 'model'):  # OPT/GPT-Neo style
            transformer = self.decoder.model
        elif hasattr(self.decoder, 'gpt_neox'):  # GPT-NeoX style
            transformer = self.decoder.gpt_neox
        else:
            transformer = self.decoder

        outputs = transformer(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask[:, :-1]
        )
        # (batch_size, seq_len-1, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # Get logits for next token prediction
        # (batch_size, seq_len-1, vocab_size)
        logits = self.decoder.lm_head(hidden_states)

        return {"logits": logits}

    def _forward_inference(self, projected_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Inference forward pass with autoregressive generation.
        """
        batch_size = projected_emb.shape[0]
        device = projected_emb.device

        # Start with projected embedding
        past_key_values = None
        generated_ids = []

        # Use projected embedding as initial state
        # We'll prepend it to the sequence
        current_emb = projected_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Get transformer and embedding layers (same logic as training)
        if hasattr(self.decoder, 'transformer'):  # GPT-2 style
            transformer = self.decoder.transformer
            wte = self.decoder.transformer.wte
        elif hasattr(self.decoder, 'model'):  # OPT/GPT-Neo style
            transformer = self.decoder.model
            wte = self.decoder.model.embed_tokens
        elif hasattr(self.decoder, 'gpt_neox'):  # GPT-NeoX style
            transformer = self.decoder.gpt_neox
            wte = self.decoder.gpt_neox.embed_in
        else:
            transformer = self.decoder
            wte = self.decoder.get_input_embeddings()

        # Generate tokens autoregressively
        for _ in range(self.max_length):
            # Forward through decoder
            outputs = transformer(
                inputs_embeds=current_emb,
                past_key_values=past_key_values,
                use_cache=True
            )

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            # Get logits for next token
            logits = self.decoder.lm_head(
                hidden_states[:, -1, :])  # (batch_size, vocab_size)

            # Sample next token (greedy for now)
            next_token_id = logits.argmax(
                dim=-1, keepdim=True)  # (batch_size, 1)
            generated_ids.append(next_token_id)

            # Check for EOS
            if (next_token_id == self.tokenizer.eos_token_id).all():
                break

            # Get embedding for next token
            next_emb = wte(next_token_id)  # (batch_size, 1, hidden_dim)
            current_emb = next_emb

        # Concatenate generated tokens
        # (batch_size, seq_len)
        generated_ids = torch.cat(generated_ids, dim=1)

        return {"generated_ids": generated_ids}

    def decode_embedding(self, embedding: torch.Tensor) -> str:
        """
        Convenience method to decode a single embedding to string.

        Args:
            embedding: Tensor of shape (embedding_dim,)

        Returns:
            tool_call: Decoded tool invocation string
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(embedding.unsqueeze(0))
            generated_ids = result["generated_ids"][0]

            # Convert to string
            tool_call = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True)

        return tool_call
