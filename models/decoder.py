"""
Decoder module for NTILC: Transforms continuous embeddings back into tool invocation strings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig

class ToolInvocationDecoder(nn.Module):
    """Decoder with improved numerical stability."""

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "google/flan-t5-base",
        max_length: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = False,
        torch_dtype: str = "bfloat16",
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model_name = model_name

        # Load pretrained encoder-decoder model (T5, BART, etc.)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Store dtype for later use
        self.dtype = dtype

        # Load T5 or other encoder-decoder model
        try:
            if 't5' in model_name.lower() or 'flan-t5' in model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
            else:
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load encoder-decoder model from {model_name}: {e}. "
                "Please use an encoder-decoder model like T5 or BART."
            )

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
                self.model.model.gradient_checkpointing_enable()

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get hidden dimension from model config
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'd_model'):
            hidden_dim = config.d_model
        elif hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'n_embd'):
            hidden_dim = config.n_embd
        else:
            raise ValueError(
                f"Could not determine hidden dimension for model {model_name}")

        # FIXED: Better initialization for projection layers
        # Use smaller initialization scale to prevent extreme values
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # FIXED: Proper initialization with smaller scale
        # This is critical for numerical stability
        for module in self.embedding_projection:
            if isinstance(module, nn.Linear):
                # Use smaller gain for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # Reduced from 0.02
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm properly
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Convert projection layer to correct dtype
        self.embedding_projection = self.embedding_projection.to(dtype)

        # REMOVED: initial_token_embedding is not needed with T5's built-in decoder_start_token_id

    def forward(
        self,
        embeddings: torch.Tensor,
        target_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """Decode embeddings to tool invocation tokens."""
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # ADDED: Check for NaN in input embeddings
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"Warning: NaN/Inf in input embeddings")
            print(f"Embeddings stats: min={embeddings.min()}, max={embeddings.max()}, mean={embeddings.mean()}")
            # Clamp to prevent propagation
            embeddings = torch.clamp(embeddings, min=-10.0, max=10.0)

        # Convert embeddings to correct dtype before projection
        embeddings = embeddings.to(self.dtype)

        # Project embedding to encoder hidden dimension
        projected = self.embedding_projection(embeddings)
        
        # ADDED: Check projection output
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            print(f"Warning: NaN/Inf after projection")
            print(f"Projected stats: min={projected.min()}, max={projected.max()}, mean={projected.mean()}")
            projected = torch.clamp(projected, min=-10.0, max=10.0)
        
        encoder_outputs = projected.unsqueeze(1)
        encoder_attention_mask = torch.ones(
            batch_size, 1, dtype=torch.long, device=device
        )

        if self.training and target_ids is not None:
            return self._forward_train(encoder_outputs, encoder_attention_mask, target_ids, attention_mask)
        else:
            return self._forward_inference(encoder_outputs, encoder_attention_mask)


    def _forward_train(
        self,
        encoder_outputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        Uses encoder outputs (from embeddings) to condition decoder generation.
        """
        batch_size = target_ids.shape[0]
        device = target_ids.device
        
        # For T5, decoder_input_ids must start with decoder_start_token_id
        # T5 uses pad_token_id as decoder_start_token_id by default
        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id
        
        # Create decoder_input_ids: [decoder_start_token_id, target_ids[:-1]]
        # This shifts the sequence right and prepends the start token
        decoder_input_ids = torch.cat([
            torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device),
            target_ids[:, :-1]
        ], dim=1)  # (batch_size, seq_len)
        
        # Labels are target_ids (for next token prediction)
        labels = target_ids  # (batch_size, seq_len)
        
        # Decoder attention mask: all ones for decoder_input_ids
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long, device=device)

        # Forward through encoder-decoder model
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs_obj = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        
        outputs = self.model(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        # Get logits for next token prediction
        # (batch_size, seq_len, vocab_size)
        logits = outputs.logits

        return {"logits": logits}

    def _forward_inference(
        self,
        encoder_outputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Inference forward pass with autoregressive generation.
        Uses encoder outputs (from embeddings) to condition decoder generation.
        Uses T5's built-in generate method for efficient generation.
        """
        # Format encoder outputs for T5
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs_obj = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        
        # Use T5's built-in generate method for efficient autoregressive generation
        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            max_new_tokens=self.max_length,
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

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
