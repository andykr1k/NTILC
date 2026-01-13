"""
Autoencoder module for NTILC: Complete encoder-decoder system for tool invocation compression.
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import ToolInvocationEncoder
from .decoder import ToolInvocationDecoder


class ToolInvocationAutoencoder(nn.Module):
    """
    Complete autoencoder that compresses tool invocations to embeddings and reconstructs them.

    Architecture:
    - Encoder: ToolCall → R^d
    - Decoder: R^d → ToolCall
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        encoder_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        decoder_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        pooling_strategy: str = "attention",
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        torch_dtype: str = "float32",
        use_gradient_checkpointing: bool = False
    ):
        """
        Args:
            embedding_dim: Dimension of embedding space (d)
            encoder_model: Base model for encoder
            decoder_model: Base model for decoder
            pooling_strategy: Encoder pooling method
            max_length: Maximum generation length
            dropout: Dropout rate
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Initialize encoder and decoder
        self.encoder = ToolInvocationEncoder(
            model_name=encoder_model,
            embedding_dim=embedding_dim,
            pooling_strategy=pooling_strategy,
            dropout=dropout,
            freeze_base=freeze_encoder,
            torch_dtype=torch_dtype
        )

        self.decoder = ToolInvocationDecoder(
            embedding_dim=embedding_dim,
            model_name=decoder_model,
            max_length=max_length,
            dropout=dropout,
            freeze_base=freeze_decoder,
            torch_dtype=torch_dtype,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

    def forward(
        self,
        tool_calls: list[str],
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: encode tool calls to embeddings, then decode back.

        Args:
            tool_calls: List of tool invocation strings
            target_ids: (batch_size, seq_len) - tokenized targets for training
            attention_mask: (batch_size, seq_len) - attention mask for training

        Returns:
            Dictionary with:
                - embeddings: (batch_size, embedding_dim)
                - logits: (batch_size, seq_len, vocab_size) - if training
                - generated_ids: (batch_size, seq_len) - if inference
        """
        # Encode to embeddings
        embeddings = self.encoder(tool_calls)

        # Decode back to tokens
        decoder_output = self.decoder(
            embeddings,
            target_ids=target_ids,
            attention_mask=attention_mask
        )

        return {
            "embeddings": embeddings,
            **decoder_output
        }

    def encode(self, tool_calls: list[str]) -> torch.Tensor:
        """
        Encode tool calls to embeddings only.

        Args:
            tool_calls: List of tool invocation strings

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.encoder(tool_calls)

    def decode(self, embeddings: torch.Tensor) -> list[str]:
        """
        Decode embeddings to tool call strings.

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            tool_calls: List of decoded tool invocation strings
        """
        self.eval()
        with torch.no_grad():
            result = self.decoder(embeddings)
            generated_ids = result["generated_ids"]

            # Convert to strings
            tool_calls = []
            for ids in generated_ids:
                tool_call = self.decoder.tokenizer.decode(
                    ids, skip_special_tokens=True)
                tool_calls.append(tool_call)

        return tool_calls

    def reconstruct(self, tool_calls: list[str]) -> list[str]:
        """
        Reconstruct tool calls: encode then decode.

        Args:
            tool_calls: List of tool invocation strings

        Returns:
            reconstructed: List of reconstructed tool invocation strings
        """
        embeddings = self.encode(tool_calls)
        return self.decode(embeddings)
