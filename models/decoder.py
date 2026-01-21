"""
Decoder module for NTILC: Transforms continuous embeddings back into tool invocation strings.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
from typing import Optional, Dict


class ToolInvocationDecoder(nn.Module):
    """Decoder with improved numerical stability and partial freezing support."""

    def __init__(
        self,
        embedding_dim: int = 256,
        model_name: str = "google/flan-t5-base",
        max_length: int = 256,
        dropout: float = 0.15,
        freeze_base: bool = False,
        freeze_layers: int = 0,
        torch_dtype: str = "bfloat16",
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model_name = model_name

        # Load pretrained encoder-decoder model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Convert torch_dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        self.dtype = dtype

        # Load model
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

        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        # Apply freezing strategy
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

        # Get hidden dimension
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, 'd_model'):
            hidden_dim = config.d_model
        elif hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'n_embd'):
            hidden_dim = config.n_embd
        else:
            raise ValueError(f"Could not determine hidden dimension for model {model_name}")

        self.hidden_dim = hidden_dim

        # Embedding projection: embedding_dim -> hidden_dim
        # Use a more expressive projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Initialize projection properly
        self._init_projection_weights()

        # Convert to correct dtype
        self.embedding_projection = self.embedding_projection.to(dtype)

    def _freeze_early_layers(self, num_layers: int):
        """Freeze the first num_layers of encoder and decoder."""
        # Freeze encoder layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'block'):
            for i, block in enumerate(self.model.encoder.block):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
                    print(f"Froze decoder's encoder layer {i}")
        
        # Freeze decoder layers
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'block'):
            for i, block in enumerate(self.model.decoder.block):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
                    print(f"Froze decoder layer {i}")

    def _init_projection_weights(self):
        """Initialize projection weights for stable training."""
        for module in self.embedding_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        target_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Decode embeddings to tool invocation tokens."""
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Check for NaN in input
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"Warning: NaN/Inf in input embeddings")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

        # Convert and project embeddings
        embeddings = embeddings.to(self.dtype)
        projected = self.embedding_projection(embeddings)
        
        # Check projection output
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            print(f"Warning: NaN/Inf after projection")
            projected = torch.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Create encoder outputs (single token representing the embedding)
        encoder_outputs = projected.unsqueeze(1)  # (batch, 1, hidden_dim)
        encoder_attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)

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
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing."""
        batch_size = target_ids.shape[0]
        device = target_ids.device
        
        # Get decoder start token
        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id
        
        # Create decoder_input_ids: shift right
        decoder_input_ids = torch.cat([
            torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device),
            target_ids[:, :-1]
        ], dim=1)
        
        labels = target_ids
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long, device=device)

        # Forward through model
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
        
        return {"logits": outputs.logits}

    def _forward_inference(
        self,
        encoder_outputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass with autoregressive generation."""
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs_obj = BaseModelOutput(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
            attentions=None
        )
        
        # Use model's generate method
        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs_obj,
            attention_mask=encoder_attention_mask,
            max_new_tokens=self.max_length,
            do_sample=False,  # Greedy decoding for deterministic output
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return {"generated_ids": generated_ids}

    def decode_embedding(self, embedding: torch.Tensor) -> str:
        """Convenience method to decode a single embedding to string."""
        self.eval()
        with torch.no_grad():
            result = self.forward(embedding.unsqueeze(0))
            generated_ids = result["generated_ids"][0]
            tool_call = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return tool_call
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
