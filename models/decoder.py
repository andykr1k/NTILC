"""
Decoder module for NTILC: Transforms continuous embeddings back into tool invocation strings.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
        model_name: str = "gpt2",
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = False
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
        
        # Load pretrained decoder
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.decoder = GPT2LMHeadModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        # Get hidden dimension from decoder
        hidden_dim = self.decoder.config.n_embd
        
        # Projection from embedding to decoder hidden dimension
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
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
        projected = self.embedding_projection(embeddings)  # (batch_size, hidden_dim)
        
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
        """
        batch_size = projected_emb.shape[0]
        device = projected_emb.device
        
        # Shift target_ids for teacher forcing (predict next token)
        input_ids = target_ids[:, :-1]  # Remove last token
        targets = target_ids[:, 1:]  # What we're predicting
        
        # Get decoder embeddings for input sequence
        decoder_embeddings = self.decoder.transformer.wte(input_ids)  # (batch_size, seq_len-1, hidden_dim)
        
        # Prepend projected embedding as initial hidden state
        projected_expanded = projected_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        decoder_embeddings = torch.cat([projected_expanded, decoder_embeddings], dim=1)
        
        # Create attention mask (include projected embedding position)
        extended_mask = torch.cat([
            torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype),
            attention_mask[:, :-1]
        ], dim=1)
        
        # Forward through decoder
        outputs = self.decoder.transformer(
            inputs_embeds=decoder_embeddings,
            attention_mask=extended_mask
        )
        hidden_states = outputs.last_hidden_state
        
        # Get logits (skip the first position which is the embedding)
        logits = self.decoder.lm_head(hidden_states[:, 1:, :])  # (batch_size, seq_len-1, vocab_size)
        
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
        
        # Generate tokens autoregressively
        for _ in range(self.max_length):
            # Forward through decoder
            outputs = self.decoder.transformer(
                inputs_embeds=current_emb,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            
            # Get logits for next token
            logits = self.decoder.lm_head(hidden_states[:, -1, :])  # (batch_size, vocab_size)
            
            # Sample next token (greedy for now)
            next_token_id = logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
            generated_ids.append(next_token_id)
            
            # Check for EOS
            if (next_token_id == self.tokenizer.eos_token_id).all():
                break
            
            # Get embedding for next token
            next_emb = self.decoder.transformer.wte(next_token_id)  # (batch_size, 1, hidden_dim)
            current_emb = next_emb
        
        # Concatenate generated tokens
        generated_ids = torch.cat(generated_ids, dim=1)  # (batch_size, seq_len)
        
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
            tool_call = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        return tool_call
