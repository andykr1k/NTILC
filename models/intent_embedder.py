"""
Intent Embedder for NTILC: Maps tool intents to 1024-D embedding space.

Embeds canonicalized intent objects built from NL-command pair samples.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Any, Optional
import json



class ToolIntentEmbedder(nn.Module):
    """
    Embeds tool intents into 1024-D space.

    Each sample is represented as a canonicalized intent object including:
    - Tool name
    - Natural language query
    - Command and arguments (from dataset rows)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        embedding_dim: int = 1024,
        pooling_strategy: str = "attention",
        dropout: float = 0.15,
        freeze_base: bool = False,
        freeze_layers: int = 0,
        torch_dtype: str = "bfloat16",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model name for base encoder
            embedding_dim: Dimension of output embedding (1024-D)
            pooling_strategy: One of ["mean", "cls", "max", "attention"]
            dropout: Dropout rate
            freeze_base: Whether to freeze all base transformer weights
            freeze_layers: Number of early layers to freeze
            torch_dtype: Data type for model weights
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length

        # Load pretrained encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Convert torch_dtype string to torch dtype
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.dtype = dtype

        # Load base model in requested dtype.
        loaded_model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.transformer = self._select_text_backbone(loaded_model)
        if self.transformer is not loaded_model:
            del loaded_model

        # FIX: Enable gradient checkpointing to save activation memory for large models
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()
        # Avoid repeated runtime warnings and KV-cache overhead during training.
        transformer_config = getattr(self.transformer, "config", None)
        if transformer_config is not None and hasattr(transformer_config, "use_cache"):
            transformer_config.use_cache = False

        # Apply freezing strategy
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

        hidden_dim = self._infer_hidden_dim(config, model_name)
        self.hidden_dim = hidden_dim

        # FIX: Attention pooling — do NOT apply Softmax inside Sequential.
        # Softmax must be applied after masking in _pool to avoid
        # distributing probability mass to padding tokens.
        if pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                # No Softmax here — applied after masking in _pool
            )
        else:
            self.attention_pool = None

        # Projection to embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self._init_projection_weights()

        if self.attention_pool is not None:
            self.attention_pool = self.attention_pool.to(dtype)
        self.projection = self.projection.to(dtype)

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def _freeze_early_layers(self, num_layers: int) -> None:
        """Freeze token embeddings and the first num_layers transformer blocks."""
        # Freeze token embeddings
        for backbone in self._candidate_backbones():
            for embed_attr in ("embed_tokens", "wte", "embeddings"):
                embed = getattr(backbone, embed_attr, None)
                if embed is not None:
                    for param in embed.parameters():
                        param.requires_grad = False
                    break  # only freeze once per backbone

        # Freeze early transformer blocks
        # FIX: Only return (stop searching) after we have successfully frozen
        # layers, not after the first non-None candidate regardless of whether
        # it contained the real transformer blocks.
        frozen = False
        for backbone in self._candidate_backbones():
            if frozen:
                break
            layer_candidates = [
                getattr(backbone, "layers", None),
                getattr(backbone, "h", None),
                getattr(backbone, "block", None),
            ]
            if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
                layer_candidates.append(backbone.encoder.layer)

            for layers in layer_candidates:
                if layers is None or len(layers) == 0:
                    continue
                for i, layer in enumerate(layers):
                    if i < num_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                frozen = True
                break  # successfully froze; stop trying other candidates

    def _candidate_backbones(self) -> List[nn.Module]:
        """Return likely transformer backbone modules, deduplicated."""
        modules = [self.transformer]
        for attr in ("language_model", "model", "transformer", "text_model", "backbone"):
            module = getattr(self.transformer, attr, None)
            if module is not None:
                modules.append(module)

        seen: set = set()
        unique: List[nn.Module] = []
        for module in modules:
            mid = id(module)
            if mid not in seen:
                seen.add(mid)
                unique.append(module)
        return unique

    def _select_text_backbone(self, model: nn.Module) -> nn.Module:
        """
        Extract text backbone from multimodal wrappers where applicable.
        Avoids registering unused visual parameters with DDP.
        """
        for attr in ("language_model", "text_model"):
            module = getattr(model, attr, None)
            if isinstance(module, nn.Module):
                return module
        return model

    # ------------------------------------------------------------------
    # Config / dim inference
    # ------------------------------------------------------------------

    def _infer_hidden_dim(self, config: Any, model_name: str) -> int:
        """Infer hidden dimension from config, handling nested layouts."""
        def _search_obj(obj: Any) -> Optional[int]:
            for attr in ("hidden_size", "d_model", "n_embd"):
                dim = getattr(obj, attr, None)
                if isinstance(dim, int) and dim > 0:
                    return dim
            return None

        def _search_dict(d: dict) -> Optional[int]:
            for attr in ("hidden_size", "d_model", "n_embd"):
                dim = d.get(attr)
                if isinstance(dim, int) and dim > 0:
                    return dim
            return None

        # Top-level config attributes
        dim = _search_obj(config)
        if dim:
            return dim

        # Nested config objects
        for nested_attr in ("text_config", "language_config", "llm_config",
                            "model_config", "decoder", "encoder"):
            nested = getattr(config, nested_attr, None)
            if nested is not None:
                dim = _search_obj(nested)
                if dim:
                    return dim

        # Fallback: config.to_dict()
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

        # Fallback: inspect backbone modules directly
        for backbone in self._candidate_backbones():
            backbone_config = getattr(backbone, "config", None)
            if backbone_config is not None:
                dim = _search_obj(backbone_config)
                if dim:
                    return dim
            # Last resort: infer from embedding weight shape
            embed = getattr(backbone, "embed_tokens", None)
            if embed is not None and hasattr(embed, "weight"):
                return int(embed.weight.shape[-1])

        raise ValueError(f"Could not determine hidden dimension for model: {model_name}")

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_projection_weights(self) -> None:
        """Initialize projection weights for stable early training."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Intent canonicalization
    # ------------------------------------------------------------------

    def _canonicalize_intent(
        self,
        tool_name: str,
        tool_call: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
    ) -> str:
        """
        Build a canonicalized text representation of a tool intent.

        Uses only sample-level fields from NL-command pair data.
        """
        parts: List[str] = [f"Tool: {tool_name}"]

        if query:
            query_text = str(query).strip()
            if query_text:
                parts.append(f"Query: {query_text}")

        args: Dict[str, Any] = {}
        if isinstance(tool_call, dict):
            raw_args = tool_call.get("arguments", {})
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                raw_args = raw_args.strip()
                if raw_args:
                    try:
                        parsed = json.loads(raw_args)
                        if isinstance(parsed, dict):
                            args = parsed
                        else:
                            args = {"command": raw_args}
                    except json.JSONDecodeError:
                        args = {"command": raw_args}
            elif raw_args not in (None, ""):
                args = {"value": raw_args}

            if not args:
                raw_command = tool_call.get("command")
                if raw_command is not None:
                    command_text = str(raw_command).strip()
                    if command_text:
                        args = {"command": command_text}

        command = args.get("command")
        if command is not None:
            command_text = str(command).strip()
            if command_text:
                parts.append(f"Command: {command_text}")

        extra_args = {k: v for k, v in args.items() if k != "command"}
        if extra_args:
            parts.append(f"Arguments: {json.dumps(extra_args, sort_keys=True, ensure_ascii=False)}")

        if tool_call:
            normalized_call = {"tool": tool_name}
            if args:
                normalized_call["arguments"] = args
            parts.append(f"ToolCall: {json.dumps(normalized_call, sort_keys=True, ensure_ascii=False)}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tool_intents: Optional[List[str]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        queries: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Embed tool intents into 1024-D unit-normalized space.

        Args:
            tool_intents: Pre-canonicalized intent strings (skip canonicalization)
            tool_calls: Tool call dicts; will be canonicalized if tool_intents is None
            queries: Optional natural language queries, aligned with tool_calls

        Returns:
            embeddings: (batch_size, embedding_dim), L2-normalized
        """
        if tool_intents is None:
            if tool_calls is None:
                raise ValueError("Must provide either tool_intents or tool_calls")
            tool_intents = [
                self._canonicalize_intent(
                    tc.get("tool", "unknown"),
                    tc,
                    queries[i] if queries and i < len(queries) else None,
                )
                for i, tc in enumerate(tool_calls)
            ]

        encoded = self.tokenizer(
            tool_intents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = self.transformer(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        attention_mask = encoded["attention_mask"]  # (B, T)

        pooled = self._pool(hidden_states, attention_mask)  # (B, H)

        pooled = pooled.to(self.dtype)
        embeddings = self.projection(pooled)  # (B, embedding_dim)

        # FIX: Normalize AFTER projection and BEFORE returning.
        # Noise injection (if used) must happen in the training loop
        # BEFORE this call, not after, so it perturbs the input to
        # projection rather than the already-normalized output.
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce (B, T, H) hidden states to (B, H)."""
        mask_dtype = hidden_states.dtype

        if self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)  # (B, T, 1)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "max":
            mask_expanded = attention_mask.unsqueeze(-1).to(dtype=mask_dtype)
            # Fill padding positions with a large negative value before max
            masked_hidden = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            return masked_hidden.max(dim=1).values

        elif self.pooling_strategy == "attention":
            # FIX: Compute raw scores, mask padding to -inf, THEN softmax.
            # The old code applied Softmax inside nn.Sequential before masking,
            # which leaked probability mass onto padding tokens.
            raw_scores = self.attention_pool(hidden_states)  # (B, T, 1)
            # Mask padding positions before softmax
            pad_mask = (attention_mask == 0).unsqueeze(-1)  # (B, T, 1)
            raw_scores = raw_scores.masked_fill(pad_mask, float("-inf"))
            attention_weights = torch.softmax(raw_scores, dim=1)  # (B, T, 1)
            # Guard against all-padding rows (shouldn't happen, but be safe)
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            pooled = (hidden_states * attention_weights).sum(dim=1)  # (B, H)
            return pooled

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
