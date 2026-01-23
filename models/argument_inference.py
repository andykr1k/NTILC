"""
Argument Inference System for NTILC.

Handles arguments separately from tool selection:
- Argument necessity detection (required, optional, irrelevant)
- Argument value generation (deterministic, autoregressive, diffusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import json
import re

from ablation.tool_schemas import TOOL_SCHEMAS


class ArgumentNecessityClassifier(nn.Module):
    """
    Classifies arguments as required, optional, or irrelevant for a given tool.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_tools: int = 6,
        dropout: float = 0.15
    ):
        """
        Args:
            hidden_dim: Hidden dimension for classifier
            num_tools: Number of tools
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_tools = num_tools
        
        # Tool embedding
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim)
        
        # Argument necessity classifier
        # Input: tool embedding + query embedding
        # Output: (num_args, 3) logits for [irrelevant, optional, required]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Per-argument classifier heads
        self.arg_heads = nn.ModuleDict()
        for tool_name, schema in TOOL_SCHEMAS.items():
            num_args = len(schema["parameters"])
            self.arg_heads[tool_name] = nn.Linear(hidden_dim // 2, num_args * 3)
    
    def forward(
        self,
        tool_ids: torch.Tensor,
        query_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Classify argument necessity.
        
        Args:
            tool_ids: (batch_size,) tool IDs
            query_embeddings: (batch_size, embedding_dim) query embeddings
        
        Returns:
            Dictionary with necessity logits per tool
        """
        batch_size = tool_ids.shape[0]
        
        # Get tool embeddings
        tool_embeds = self.tool_embedding(tool_ids)  # (batch_size, hidden_dim)
        
        # Concatenate tool and query embeddings
        combined = torch.cat([tool_embeds, query_embeddings], dim=1)  # (batch_size, hidden_dim * 2)
        
        # Get shared features
        features = self.classifier(combined)  # (batch_size, hidden_dim // 2)
        
        # Get per-tool predictions
        results = {}
        for tool_name in TOOL_SCHEMAS.keys():
            tool_idx = list(TOOL_SCHEMAS.keys()).index(tool_name)
            mask = tool_ids == tool_idx
            
            if mask.sum() == 0:
                continue
            
            # Get logits for this tool
            logits = self.arg_heads[tool_name](features[mask])  # (num_samples, num_args * 3)
            num_args = len(TOOL_SCHEMAS[tool_name]["parameters"])
            logits = logits.view(-1, num_args, 3)  # (num_samples, num_args, 3)
            
            results[tool_name] = logits
        
        return results


class ArgumentValueGenerator(nn.Module):
    """
    Generates argument values using appropriate mechanisms:
    - Deterministic extraction (IDs, strings)
    - Autoregressive generation (enums, short text)
    - Diffusion (continuous values, coordinates, layouts)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        vocab_size: int = 32000,
        max_length: int = 128
    ):
        """
        Args:
            embedding_dim: Dimension of query/tool embeddings
            vocab_size: Vocabulary size for text generation
            max_length: Maximum generation length
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Shared encoder for query + tool context
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # Deterministic extractor (for IDs, strings)
        self.deterministic_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)  # Output: extracted value
        )
        
        # Autoregressive generator (for enums, short text)
        # Using simple linear projection for now (can be extended to transformer)
        self.autoregressive_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, vocab_size)
        )
    
    def generate_deterministic(
        self,
        context: torch.Tensor,
        arg_type: str,
        arg_name: str
    ) -> Any:
        """
        Generate deterministic argument value (IDs, strings).
        
        Args:
            context: (batch_size, embedding_dim) context embedding
            arg_type: Type of argument
            arg_name: Name of argument
        
        Returns:
            Generated value
        """
        # For deterministic extraction, we use simple heuristics
        # In practice, this could use NER, regex, or rule-based extraction
        
        # Placeholder: return None (will be filled by extraction logic)
        return None
    
    def generate_autoregressive(
        self,
        context: torch.Tensor,
        arg_type: str,
        arg_name: str,
        max_length: int = 32
    ) -> str:
        """
        Generate argument value using autoregressive generation.
        
        Args:
            context: (batch_size, embedding_dim) context embedding
            arg_type: Type of argument
            arg_name: Name of argument
            max_length: Maximum generation length
        
        Returns:
            Generated string value
        """
        # Simple generation (can be extended to full transformer)
        logits = self.autoregressive_head(context)
        # Placeholder: would use actual generation logic
        return ""
    
    def extract_from_query(
        self,
        query: str,
        arg_name: str,
        arg_type: str,
        tool_name: str
    ) -> Optional[Any]:
        """
        Extract argument value from query using deterministic methods.
        
        Args:
            query: Natural language query
            arg_name: Argument name
            arg_type: Argument type
            tool_name: Tool name
        
        Returns:
            Extracted value or None
        """
        # Simple extraction logic
        # For email addresses
        if arg_type == "email" or "email" in arg_name.lower():
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = re.findall(email_pattern, query)
            if matches:
                return matches[0]
        
        # For URLs
        if arg_type == "str" and ("url" in arg_name.lower() or "link" in arg_name.lower()):
            url_pattern = r'https?://[^\s]+'
            matches = re.findall(url_pattern, query)
            if matches:
                return matches[0]
        
        # For numbers
        if arg_type in ["int", "float"]:
            number_pattern = r'\d+(?:\.\d+)?'
            matches = re.findall(number_pattern, query)
            if matches:
                try:
                    if arg_type == "int":
                        return int(float(matches[0]))
                    else:
                        return float(matches[0])
                except ValueError:
                    pass
        
        # For enum types, check if query contains enum value
        schema = TOOL_SCHEMAS.get(tool_name, {})
        param_info = schema.get("parameters", {}).get(arg_name, {})
        if "options" in param_info:
            options = param_info["options"]
            query_lower = query.lower()
            for option in options:
                if option.lower() in query_lower:
                    return option
        
        return None
    
    def forward(
        self,
        query: str,
        tool_name: str,
        necessary_args: List[str],
        query_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate argument values for necessary arguments.
        
        Args:
            query: Natural language query
            tool_name: Tool name
            necessary_args: List of argument names that are necessary
            query_embedding: Optional query embedding
        
        Returns:
            Dictionary mapping argument names to values
        """
        schema = TOOL_SCHEMAS.get(tool_name, {})
        arguments = {}
        
        for arg_name in necessary_args:
            if arg_name not in schema["parameters"]:
                continue
            
            param_info = schema["parameters"][arg_name]
            arg_type = param_info.get("type", "str")
            
            # Try deterministic extraction first
            value = self.extract_from_query(query, arg_name, arg_type, tool_name)
            
            if value is None:
                # Try autoregressive generation for text/enum
                if arg_type in ["str", "enum"]:
                    # Placeholder: would use actual generation
                    value = ""
                else:
                    # Use default if available
                    value = param_info.get("default", None)
            
            if value is not None:
                arguments[arg_name] = value
        
        return arguments
