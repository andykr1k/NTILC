"""
LLM Integration module for NTILC Phase 2.

This module trains an LLM to predict tool call embeddings from natural language queries.
The predicted embeddings are then decoded by the trained autoencoder's decoder.

Architecture:
- Base LLM (Flan-T5) processes natural language query
- Tool prediction head projects hidden states to embedding space
- Frozen autoencoder decoder converts embeddings to tool call strings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
from typing import Dict, List, Optional, Tuple
import os


class ToolPredictionHead(nn.Module):
    """
    Prediction head that projects LLM hidden states to tool embedding space.
    
    Includes an auxiliary tool classification head for multi-task learning.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        num_tools: int = 6,
        dropout: float = 0.15
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_tools = num_tools
        
        # Main embedding prediction head
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Auxiliary tool classification head
        self.tool_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tools)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project hidden states to embedding space.
        
        Args:
            hidden_states: (batch_size, hidden_dim) pooled LLM output
            
        Returns:
            Dictionary with:
                - embedding: (batch_size, embedding_dim) predicted tool embedding
                - tool_logits: (batch_size, num_tools) tool classification logits
        """
        embedding = self.embedding_head(hidden_states)
        # Normalize to unit sphere (match autoencoder's embedding space)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        tool_logits = self.tool_classifier(hidden_states)
        
        return {
            "embedding": embedding,
            "tool_logits": tool_logits
        }


class ToolPredictionLLM(nn.Module):
    """
    LLM that predicts tool call embeddings from natural language queries.
    
    Architecture:
    1. Encode NL query with T5 encoder
    2. Pool encoder outputs
    3. Project to embedding space via prediction head
    4. (Optional) Decode embedding with frozen autoencoder decoder
    """
    
    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        embedding_dim: int = 256,
        num_tools: int = 6,
        dropout: float = 0.15,
        freeze_base_layers: int = 6,
        torch_dtype: str = "bfloat16",
        max_length: int = 256
    ):
        """
        Args:
            base_model: HuggingFace model name for base LLM
            embedding_dim: Dimension of tool embedding space
            num_tools: Number of tools for auxiliary classification
            dropout: Dropout rate
            freeze_base_layers: Number of early layers to freeze
            torch_dtype: Data type for model weights
            max_length: Maximum input sequence length
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Data type
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.dtype = dtype
        
        # Load tokenizer and encoder
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.encoder = T5EncoderModel.from_pretrained(base_model, torch_dtype=dtype)
        
        # Get hidden dimension
        config = AutoConfig.from_pretrained(base_model)
        self.hidden_dim = config.d_model if hasattr(config, 'd_model') else config.hidden_size
        
        # Freeze early layers
        if freeze_base_layers > 0:
            self._freeze_early_layers(freeze_base_layers)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softmax(dim=1)
        ).to(dtype)
        
        # Tool prediction head
        self.prediction_head = ToolPredictionHead(
            hidden_dim=self.hidden_dim,
            embedding_dim=embedding_dim,
            num_tools=num_tools,
            dropout=dropout
        ).to(dtype)
        
        # Autoencoder decoder (loaded separately)
        self.decoder = None
    
    def _freeze_early_layers(self, num_layers: int):
        """Freeze early encoder layers."""
        if hasattr(self.encoder, 'embed_tokens'):
            for param in self.encoder.embed_tokens.parameters():
                param.requires_grad = False
        
        if hasattr(self.encoder, 'block'):
            for i, block in enumerate(self.encoder.block):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def load_decoder(self, autoencoder_checkpoint: str, device: torch.device = None):
        """
        Load decoder from trained autoencoder checkpoint.
        
        Args:
            autoencoder_checkpoint: Path to autoencoder checkpoint
            device: Device to load decoder to
        """
        from models.autoencoder import ToolInvocationAutoencoder
        
        # Load checkpoint
        checkpoint = torch.load(autoencoder_checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Create autoencoder with same config
        autoencoder = ToolInvocationAutoencoder(
            embedding_dim=config.get('embedding_dim', self.embedding_dim),
            encoder_model=config.get('encoder_model', 'google/flan-t5-base'),
            decoder_model=config.get('decoder_model', 'google/flan-t5-base'),
            max_length=config.get('max_length', 256),
            torch_dtype=config.get('torch_dtype', 'bfloat16')
        )
        
        # Load weights
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract decoder and freeze
        self.decoder = autoencoder.decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        if device:
            self.decoder = self.decoder.to(device)
        
        self.decoder.eval()
        print(f"Loaded decoder from {autoencoder_checkpoint}")
    
    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool sequence hidden states using attention pooling."""
        attention_weights = self.attention_pool(hidden_states)
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        attention_weights = attention_weights * mask_expanded
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        return pooled
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        nl_queries: List[str] = None,
        target_embeddings: torch.Tensor = None,
        target_tool_labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode NL query and predict tool embedding.
        
        Args:
            input_ids: (batch_size, seq_len) tokenized inputs
            attention_mask: (batch_size, seq_len) attention mask
            nl_queries: List of NL query strings (alternative to input_ids)
            target_embeddings: (batch_size, embedding_dim) target embeddings for training
            target_tool_labels: (batch_size,) target tool labels for auxiliary loss
            
        Returns:
            Dictionary with predictions and losses
        """
        device = next(self.parameters()).device
        
        # Tokenize if strings provided
        if nl_queries is not None:
            encoded = self.tokenizer(
                nl_queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        
        # Encode with T5
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Pool to single vector
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        pooled = pooled.to(self.dtype)
        
        # Predict embedding and tool
        predictions = self.prediction_head(pooled)
        
        result = {
            "embedding": predictions["embedding"],
            "tool_logits": predictions["tool_logits"]
        }
        
        # Compute losses if targets provided
        if target_embeddings is not None:
            # MSE loss for embedding prediction
            embedding_loss = F.mse_loss(predictions["embedding"], target_embeddings)
            
            # Cosine similarity loss (additional)
            cosine_sim = F.cosine_similarity(predictions["embedding"], target_embeddings, dim=1)
            cosine_loss = (1 - cosine_sim).mean()
            
            result["embedding_loss"] = embedding_loss
            result["cosine_loss"] = cosine_loss
        
        if target_tool_labels is not None:
            # Cross-entropy loss for tool classification
            tool_loss = F.cross_entropy(predictions["tool_logits"], target_tool_labels)
            result["tool_loss"] = tool_loss
        
        return result
    
    def predict_embedding(self, nl_query: str) -> torch.Tensor:
        """Predict embedding for a single NL query."""
        self.eval()
        with torch.no_grad():
            result = self.forward(nl_queries=[nl_query])
            return result["embedding"][0]
    
    def predict_tool_call(self, nl_query: str) -> str:
        """
        Predict tool call string from NL query.
        
        Requires decoder to be loaded via load_decoder().
        """
        if self.decoder is None:
            raise ValueError("Decoder not loaded. Call load_decoder() first.")
        
        self.eval()
        with torch.no_grad():
            # Predict embedding
            embedding = self.predict_embedding(nl_query)
            
            # Decode to tool call string
            result = self.decoder(embedding.unsqueeze(0))
            generated_ids = result["generated_ids"][0]
            tool_call = self.decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return tool_call
    
    def predict_batch(self, nl_queries: List[str]) -> List[str]:
        """Predict tool calls for multiple NL queries."""
        if self.decoder is None:
            raise ValueError("Decoder not loaded. Call load_decoder() first.")
        
        self.eval()
        with torch.no_grad():
            # Predict embeddings
            result = self.forward(nl_queries=nl_queries)
            embeddings = result["embedding"]
            
            # Decode all
            decoder_result = self.decoder(embeddings)
            generated_ids = decoder_result["generated_ids"]
            
            tool_calls = []
            for ids in generated_ids:
                tool_call = self.decoder.tokenizer.decode(ids, skip_special_tokens=True)
                tool_calls.append(tool_call)
            
            return tool_calls
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class NLToolCallDataset(torch.utils.data.Dataset):
    """
    Dataset for (NL query, tool call embedding) pairs.
    
    Uses a trained autoencoder to compute target embeddings.
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        autoencoder,
        tokenizer,
        max_length: int = 256,
        tool_to_idx: Dict[str, int] = None
    ):
        """
        Args:
            data: List of dicts with 'query', 'tool_call', 'tool' keys
            autoencoder: Trained autoencoder for computing target embeddings
            tokenizer: Tokenizer for NL queries
            max_length: Maximum sequence length
            tool_to_idx: Tool name to index mapping
        """
        self.data = data
        self.autoencoder = autoencoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if tool_to_idx is None:
            tools = ["search", "calculate", "database_query", "send_email", "web_fetch", "file_read"]
            self.tool_to_idx = {t: i for i, t in enumerate(tools)}
        else:
            self.tool_to_idx = tool_to_idx
        
        # Precompute embeddings
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute target embeddings using autoencoder."""
        print("Precomputing target embeddings...")
        self.autoencoder.eval()
        
        self.embeddings = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i+batch_size]
                tool_calls = [item['tool_call'] for item in batch]
                embeddings = self.autoencoder.encode(tool_calls)
                self.embeddings.append(embeddings.cpu())
        
        self.embeddings = torch.cat(self.embeddings, dim=0)
        print(f"Precomputed {len(self.embeddings)} embeddings")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize NL query
        encoded = self.tokenizer(
            item['query'],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get precomputed embedding
        target_embedding = self.embeddings[idx]
        
        # Get tool label
        tool_label = self.tool_to_idx.get(item.get('tool', 'unknown'), 0)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_embedding": target_embedding,
            "tool_label": tool_label,
            "query": item['query'],
            "tool_call": item['tool_call']
        }
