"""
Baseline 2: Cross-Attention LLM (Fixed)
Add cross-attention mechanism to make LLM aware of tool embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import math
from tqdm import tqdm
from multiprocessing import Process, Queue
import multiprocessing

from .tool_schemas import TOOL_SCHEMAS

# Import the extraction function from naive baseline for consistency
from .baseline_naive import _extract_tool_call


def _cross_attention_worker_process(gpu_id, queries, prompts_list, result_queue, progress_queue, model_name, num_tools, cross_attention_layers, max_length, temperature, batch_size=8):
    """Worker process that runs on a specific GPU (module-level for pickling)."""
    import torch
    from transformers import AutoTokenizer
    
    # Set CUDA device for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # Import CrossAttentionLLM - use absolute import for multiprocessing
    try:
        from ablation.baseline_cross_attention import CrossAttentionLLM
    except ImportError:
        # Fallback: try relative import
        import sys
        import os
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ablation.baseline_cross_attention import CrossAttentionLLM
    
    # Load model and tokenizer in this process
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set left padding for decoder-only models
    tokenizer.padding_side = 'left'
    model = CrossAttentionLLM(
        base_model_name=model_name,
        num_tools=num_tools,
        cross_attention_layers=cross_attention_layers
    )
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process queries in batches for better GPU utilization
    predictions = []
    
    for i in range(0, len(prompts_list), batch_size):
        batch_prompts = prompts_list[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Generate for batch
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            temperature=temperature
        )
        
        # Decode and extract for each in batch
        batch_predictions = []
        for prompt, gen_ids in zip(batch_prompts, generated):
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Use improved extraction function
            tool_call = _extract_tool_call(generated_text, prompt)
            batch_predictions.append((generated_text, tool_call))
        
        predictions.extend(batch_predictions)
        
        # Send progress update after each batch
        if progress_queue is not None:
            progress_queue.put(len(batch_predictions))
    
    # Send final results back
    result_queue.put((gpu_id, predictions))


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer that attends to tool embeddings.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_model)
        key_value: torch.Tensor,  # (batch, num_tools, d_model)
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query from LLM hidden states
            key_value: Tool embeddings to attend to
            attention_mask: Optional mask for tools
        """
        batch_size, seq_len, _ = query.shape
        num_tools = key_value.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (batch, seq_len, d_model)
        K = self.k_proj(key_value)  # (batch, num_tools, d_model)
        V = self.v_proj(key_value)  # (batch, num_tools, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tools, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tools, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(attn_output)
        
        return output


class ToolEmbeddingEncoder(nn.Module):
    """
    Encodes tool schemas into embeddings using Qwen.
    Creates embeddings from example tool calls for each tool type.
    """
    
    def __init__(self, d_model: int, num_tools: int, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        self.num_tools = num_tools
        self.d_model = d_model
        self.base_model_name = base_model_name
        
        # Load Qwen tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Get the hidden size from the model config
        if hasattr(self.qwen_model.config, 'hidden_size'):
            encoder_dim = self.qwen_model.config.hidden_size
        elif hasattr(self.qwen_model.config, 'n_embd'):
            encoder_dim = self.qwen_model.config.n_embd
        else:
            encoder_dim = self.qwen_model.config.d_model if hasattr(self.qwen_model.config, 'd_model') else 2048
        
        # Project to desired dimension
        self.projection = nn.Linear(encoder_dim, d_model)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create example tool calls for each tool (will be encoded once)
        self._example_tool_calls = None
        self._tool_calls_by_type = None  # Store all tool calls grouped by type
        self._cached_embeddings = None
        self._average_embeddings = True  # Whether to average multiple tool calls per tool type
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.qwen_model = self.qwen_model.to(device)
        self.projection = self.projection.to(device)
        # Clear cache when moving devices
        self._cached_embeddings = None
        # Note: _tool_calls_by_type doesn't need to be cleared as it's just data
        return self
    
    def _load_tool_calls_from_test_data(self) -> Dict[str, List[str]]:
        """Load tool calls from test data file, grouped by tool type."""
        import json
        import os
        
        # Try to load from test data file
        test_data_paths = [
            'data/ablation/train_data.jsonl',
            '../data/ablation/train_data.jsonl',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/ablation/train_data.jsonl')
        ]
        
        tool_calls_by_type = {}
        
        # Try each possible path
        test_data_path = None
        for path in test_data_paths:
            if os.path.exists(path):
                test_data_path = path
                break
        
        if test_data_path:
            try:
                with open(test_data_path, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        tool_call = data.get('ground_truth', '')
                        tool_name = data.get('tool', '')
                        
                        if tool_call and tool_name:
                            if tool_name not in tool_calls_by_type:
                                tool_calls_by_type[tool_name] = []
                            tool_calls_by_type[tool_name].append(tool_call)
                
                # print(f"Loaded {sum(len(calls) for calls in tool_calls_by_type.values())} tool calls from test data for {len(tool_calls_by_type)} tool types")
            except Exception as e:
                print(f"Warning: Could not load test data: {e}")
                tool_calls_by_type = {}
        
        return tool_calls_by_type
    
    def _get_tool_calls_for_encoding(self) -> Dict[str, List[str]]:
        """Get tool calls for each tool type (from test data or examples)."""
        # Load from test data
        tool_calls_by_type = self._load_tool_calls_from_test_data()
        
        # Fallback to examples if test data not available
        if not tool_calls_by_type:
            from .tool_schemas import get_tool_examples
            tool_examples = get_tool_examples()
            tool_calls_by_type = {tool: examples for tool, examples in tool_examples.items()}
        
        # Ensure all tools are represented
        from .tool_schemas import TOOL_SCHEMAS
        tool_names = list(TOOL_SCHEMAS.keys())
        for tool_name in tool_names:
            if tool_name not in tool_calls_by_type or not tool_calls_by_type[tool_name]:
                # Fallback: use first example from schema
                from .tool_schemas import get_tool_examples
                tool_examples = get_tool_examples()
                if tool_name in tool_examples and tool_examples[tool_name]:
                    tool_calls_by_type[tool_name] = tool_examples[tool_name]
                else:
                    # Last resort: create a minimal tool call
                    tool_calls_by_type[tool_name] = [f"{tool_name}()"]
        
        return tool_calls_by_type
    
    def _encode_tool_call(self, tool_call_str: str, device: torch.device) -> torch.Tensor:
        """
        Encode a tool call string into an embedding using Qwen.
        
        Args:
            tool_call_str: String like "search(query='machine learning', max_results=10)"
            device: Device to run on
            
        Returns:
            (d_model,) tensor
        """
        # Tokenize the tool call
        inputs = self.tokenizer(
            tool_call_str,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.qwen_model(**inputs, output_hidden_states=True)
            
            # Use mean pooling over all token embeddings
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Mean pooling over sequence length
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            else:
                embedding = hidden_states.mean(dim=1)
            
            # Project to desired dimension
            embedding = self.projection(embedding.squeeze(0))
        
        return embedding
    
    def forward(self) -> torch.Tensor:
        """
        Returns tool embeddings encoded from tool calls in test data.
        If multiple tool calls exist for a tool type, averages their embeddings.
        
        Returns:
            (num_tools, d_model) tensor of tool embeddings
        """
        device = next(self.projection.parameters()).device
        
        # Ensure Qwen model is on the correct device
        if next(self.qwen_model.parameters()).device != device:
            self.qwen_model = self.qwen_model.to(device)
        
        # Cache embeddings if not already computed
        if self._cached_embeddings is None or self._cached_embeddings.device != device:
            if self._tool_calls_by_type is None:
                self._tool_calls_by_type = self._get_tool_calls_for_encoding()
            
            # Get tool names in consistent order
            from .tool_schemas import TOOL_SCHEMAS
            tool_names = list(TOOL_SCHEMAS.keys())
            
            # Encode tool calls for each tool type
            embeddings = []
            for tool_name in tool_names:
                tool_calls = self._tool_calls_by_type.get(tool_name, [])
                if not tool_calls:
                    # Fallback
                    tool_calls = [f"{tool_name}()"]
                
                if self._average_embeddings and len(tool_calls) > 1:
                    # Average embeddings from multiple tool calls
                    tool_embeddings = []
                    for tool_call in tool_calls:
                        emb = self._encode_tool_call(tool_call, device)
                        tool_embeddings.append(emb)
                    # Average the embeddings
                    avg_emb = torch.stack(tool_embeddings).mean(dim=0)
                    embeddings.append(avg_emb)
                else:
                    # Use first tool call
                    emb = self._encode_tool_call(tool_calls[0], device)
                    embeddings.append(emb)
            
            self._cached_embeddings = torch.stack(embeddings)
        
        return self._cached_embeddings


class CrossAttentionLLM(nn.Module):
    """
    LLM with cross-attention to tool embeddings.
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        num_tools: int = 6,
        cross_attention_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        Args:
            base_model_name: Base LLM model name
            num_tools: Number of tools
            cross_attention_layers: Number of cross-attention layers to add
            num_heads: Number of attention heads
            dropout: Dropout rate
            freeze_base: Whether to freeze base LLM weights
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.num_tools = num_tools
        
        # Load base model
        config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.d_model = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Tool embedding encoder (uses Qwen to encode example tool calls)
        self.tool_encoder = ToolEmbeddingEncoder(self.d_model, num_tools, base_model_name=base_model_name)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(self.d_model, num_heads, dropout)
            for _ in range(cross_attention_layers)
        ])
        
        # Initialize cross-attention layers to be near-identity to avoid corrupting base model
        # when not trained. This makes the model work as a baseline even without training.
        self._initialize_cross_attention_near_identity()
        
        # Layer norm for residual connection
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def _initialize_cross_attention_near_identity(self):
        """
        Initialize cross-attention layers to be near-identity so they don't corrupt
        the base model's outputs when used without training.
        """
        for layer in self.cross_attention_layers:
            # Initialize output projection with very small values (near-zero)
            # This makes the cross-attention output small, so residual connection dominates
            nn.init.normal_(layer.out_proj.weight, mean=0.0, std=0.001)
            nn.init.zeros_(layer.out_proj.bias)
            
            # Initialize Q, K, V projections with small random values
            # This allows the model to learn if trained, but minimizes corruption if not
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj]:
                nn.init.normal_(proj.weight, mean=0.0, std=0.01)
                nn.init.zeros_(proj.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-attention to tools.
        
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) attention mask
            labels: (batch, seq_len) labels for training
            
        Returns:
            Dictionary with logits and loss
        """
        # Get base model hidden states
        # FIX: Use getattr with proper fallback
        if hasattr(self.base_model, 'model'):
            transformer = self.base_model.model
        elif hasattr(self.base_model, 'transformer'):
            transformer = self.base_model.transformer
        else:
            raise AttributeError(f"Model {self.base_model_name} does not have 'model' or 'transformer' attribute")
        
        transformer_outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states (last layer)
        hidden_states = transformer_outputs.last_hidden_state  # (batch, seq_len, d_model)
        
        # Get tool embeddings
        tool_embeddings = self.tool_encoder()  # (num_tools, d_model)
        tool_embeddings = tool_embeddings.unsqueeze(0).expand(
            hidden_states.shape[0], -1, -1
        )  # (batch, num_tools, d_model)
        
        # Apply cross-attention layers
        x = hidden_states
        for cross_attn in self.cross_attention_layers:
            attn_output = cross_attn(x, tool_embeddings, attention_mask=None)
            x = self.layer_norm(x + attn_output)  # Residual connection
        
        # Project to vocabulary
        logits = self.base_model.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with tool awareness.
        
        Args:
            input_ids: (batch, seq_len) input token ids
            attention_mask: (batch, seq_len) attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated token ids
        """
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            generated = input_ids.clone()
            eos_token_id = self.base_model.config.eos_token_id
            
            # Track which sequences are finished
            finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"][:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                    ], dim=1)
                
                # Check for EOS tokens (handle batched case)
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                
                # Stop if all sequences are finished
                if finished.all():
                    break
            
            return generated


class CrossAttentionBaseline:
    """
    Wrapper class for cross-attention baseline.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_tools: int = 6,
        cross_attention_layers: int = 1,
        max_length: int = 256,
        temperature: float = 0.7,
        num_gpus: int = 1
    ):
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        # Determine actual number of GPUs to use
        if num_gpus > 1 and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            self.num_gpus = min(num_gpus, available_gpus)
            if self.num_gpus < num_gpus:
                print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. Using {self.num_gpus} GPUs.")
        else:
            self.num_gpus = 1
        
        print(f"Loading cross-attention model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set left padding for decoder-only models
        self.tokenizer.padding_side = 'left'
        self.model_name = model_name
        self.num_tools = num_tools
        self.cross_attention_layers = cross_attention_layers
        
        # Create model instances for each GPU (only for single GPU case, multi-GPU loads in processes)
        self.models = []
        self.devices = []
        
        if self.num_gpus == 1:
            # Single GPU or CPU - load model here
            model = CrossAttentionLLM(
                base_model_name=model_name,
                num_tools=num_tools,
                cross_attention_layers=cross_attention_layers
            )
            model.to(device)
            model.eval()
            self.models = [model]
            self.devices = [device]
        else:
            # Multi-GPU - models will be loaded in worker processes
            # Just store device info
            for gpu_id in range(self.num_gpus):
                self.devices.append(f"cuda:{gpu_id}")
            # Create a dummy model for backward compatibility (won't be used in multi-GPU)
            self.models = [None] * self.num_gpus
        
        # For backward compatibility
        if self.num_gpus == 1:
            self.model = self.models[0]
        else:
            self.model = None  # Will be created in worker processes
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def predict(self, user_query: str, tools_prompt: str = "") -> str:
        """
        Predict tool call for user query.
        
        Args:
            user_query: User's request
            tools_prompt: Optional prompt describing tools
            
        Returns:
            Predicted tool call string
        """
        # Build prompt
        prompt = f"{tools_prompt}\n\nUser request: {user_query}\nTool call:"
        
        # For single queries, use single GPU path
        if self.num_gpus == 1:
            device = self.devices[0]
            model = self.models[0]
        else:
            # Multi-GPU: create a temporary model for single query
            device = self.devices[0]
            model = CrossAttentionLLM(
                base_model_name=self.model_name,
                num_tools=self.num_tools,
                cross_attention_layers=self.cross_attention_layers
            )
            model.to(device)
            model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=self.max_length,
            temperature=self.temperature
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract tool call using improved parsing
        tool_call = _extract_tool_call(generated_text, prompt)
        
        return tool_call
    
    
    def predict_batch(self, user_queries: List[str], batch_size: int = None) -> List[tuple]:
        """
        Predict tool calls for multiple queries using multi-GPU if available.
        
        Args:
            user_queries: List of user queries
            batch_size: Batch size per GPU. If None, uses 8 as default
            
        Returns:
            List of tuples (generated_text, tool_call)
        """
        if batch_size is None:
            batch_size = 8
        
        # If single GPU, use simple batching
        if self.num_gpus == 1:
            prompts = [f"\n\nUser request: {query}\nTool call:" for query in user_queries]
            predictions = []
            num_batches = (len(user_queries) + batch_size - 1) // batch_size
            device = self.devices[0]
            model = self.models[0]
            
            for i in tqdm(range(0, len(user_queries), batch_size), desc="Predicting tool calls", total=num_batches):
                batch_prompts = prompts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)
                
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=self.max_length,
                    temperature=self.temperature
                )
                
                for prompt, gen_ids in zip(batch_prompts, generated):
                    generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    # Use improved extraction function
                    tool_call = _extract_tool_call(generated_text, prompt)
                    predictions.append((generated_text, tool_call))
            
            return predictions
        
        # Multi-GPU: Process queries in parallel using multiprocessing
        # Build prompts first (before forking processes)
        prompts = [f"\n\nUser request: {query}\nTool call:" for query in user_queries]
        
        # Split queries and prompts into chunks for each GPU
        chunk_size = len(user_queries) // self.num_gpus
        query_chunks = []
        prompt_chunks = []
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            if i == self.num_gpus - 1:
                end_idx = len(user_queries)
            else:
                end_idx = (i + 1) * chunk_size
            query_chunks.append(user_queries[start_idx:end_idx])
            prompt_chunks.append(prompts[start_idx:end_idx])
        
        # Set multiprocessing start method (required for CUDA)
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        # Start processes for each GPU
        result_queue = Queue()
        progress_queue = Queue()  # Separate queue for progress updates
        processes = []
        
        # Get model parameters from stored values
        num_tools = self.num_tools
        cross_attention_layers = self.cross_attention_layers
        
        for gpu_id, (queries, prompts_list) in enumerate(zip(query_chunks, prompt_chunks)):
            if not queries:
                continue
            p = Process(
                target=_cross_attention_worker_process,
                args=(
                    gpu_id,
                    queries,
                    prompts_list,
                    result_queue,
                    progress_queue,  # Pass progress queue
                    self.model_name,
                    num_tools,
                    cross_attention_layers,
                    self.max_length,
                    self.temperature,
                    batch_size  # Pass batch_size to worker
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results with real-time progress updates
        all_predictions = [None] * len(user_queries)
        completed = 0
        
        with tqdm(total=len(user_queries), desc="Predicting tool calls") as pbar:
            # Update progress bar as batches complete
            while completed < len(user_queries):
                try:
                    # Check for progress updates (non-blocking)
                    while not progress_queue.empty():
                        batch_count = progress_queue.get_nowait()
                        completed += batch_count
                        pbar.update(batch_count)
                    
                    # Check for final results (non-blocking)
                    if not result_queue.empty():
                        gpu_id, chunk_predictions = result_queue.get_nowait()
                        chunk_start = gpu_id * chunk_size
                        all_predictions[chunk_start:chunk_start + len(chunk_predictions)] = chunk_predictions
                    
                    # Small sleep to avoid busy waiting
                    import time
                    time.sleep(0.1)
                except:
                    # If queue is empty, wait a bit
                    import time
                    time.sleep(0.1)
            
            # Collect any remaining results
            for _ in range(len(processes)):
                if not result_queue.empty():
                    gpu_id, chunk_predictions = result_queue.get_nowait()
                    chunk_start = gpu_id * chunk_size
                    if all_predictions[chunk_start] is None:  # Only if not already set
                        all_predictions[chunk_start:chunk_start + len(chunk_predictions)] = chunk_predictions
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        return all_predictions
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate on test data."""
        from evaluation.metrics import exact_match_accuracy, tool_accuracy, parameter_accuracy
        
        queries = [item["query"] for item in test_data]
        ground_truth = [item["ground_truth"] for item in test_data]
        
        predictions_with_original = self.predict_batch(queries)
        # Extract just the tool calls for metrics
        predictions = [tool_call for _, tool_call in predictions_with_original]
        original_predictions = [orig for orig, _ in predictions_with_original]
        
        metrics = {
            "exact_match_accuracy": exact_match_accuracy(ground_truth, predictions),
            "tool_accuracy": tool_accuracy(ground_truth, predictions),
            "predictions": predictions,  # Store extracted tool calls for reuse
            "original_predictions": original_predictions  # Store original generated text
        }
        
        param_metrics = parameter_accuracy(ground_truth, predictions)
        metrics.update(param_metrics)
        
        return metrics