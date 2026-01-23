"""
NTILC Inference Pipeline

End-to-end inference from natural language queries to tool call execution.

Usage:
    from inference import ToolCallingSystem
    
    # Load system
    system = ToolCallingSystem.from_pretrained(
        autoencoder_path="checkpoints/best_model.pt",
        llm_path="checkpoints/llm_integration/best_model.pt"
    )
    
    # Predict tool call
    result = system.predict("Get the last 10 orders from California")
    print(result)
    # -> {"tool": "database_query", "arguments": {"sql": "SELECT * FROM orders WHERE state = 'CA' LIMIT 10", "timeout": 30}}
"""

import torch
import json
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ToolCallResult:
    """Result of tool call prediction."""
    tool_call: str  # Raw tool call string
    tool_name: str  # Extracted tool name
    arguments: Dict  # Extracted arguments
    embedding: torch.Tensor  # Predicted embedding
    confidence: float  # Tool classification confidence
    
    def to_dict(self) -> Dict:
        return {
            "tool_call": self.tool_call,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "confidence": self.confidence
        }
    
    def __repr__(self):
        return f"ToolCallResult(tool={self.tool_name}, args={self.arguments}, conf={self.confidence:.3f})"


class ToolCallingSystem:
    """
    End-to-end tool calling system.
    
    Takes natural language queries and produces structured tool calls.
    
    Pipeline:
    1. NL Query -> LLM Encoder -> Hidden States
    2. Hidden States -> Tool Prediction Head -> Embedding
    3. Embedding -> Autoencoder Decoder -> Tool Call String
    4. Tool Call String -> Parser -> Structured Output
    """
    
    def __init__(
        self,
        llm_model=None,
        autoencoder=None,
        device: str = None
    ):
        """
        Initialize with models.
        
        Use from_pretrained() for loading saved models.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model = llm_model
        self.autoencoder = autoencoder
        
        self.tool_names = ["search", "calculate", "database_query", "send_email", "web_fetch", "file_read"]
    
    @classmethod
    def from_pretrained(
        cls,
        autoencoder_path: str,
        llm_path: str = None,
        device: str = None
    ) -> "ToolCallingSystem":
        """
        Load system from saved checkpoints.
        
        Args:
            autoencoder_path: Path to autoencoder checkpoint
            llm_path: Path to LLM integration checkpoint (optional)
            device: Device to load models to
            
        Returns:
            Initialized ToolCallingSystem
        """
        from models.autoencoder import ToolInvocationAutoencoder
        from models.llm_integration import ToolPredictionLLM
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load autoencoder
        print(f"Loading autoencoder from {autoencoder_path}...")
        ae_checkpoint = torch.load(autoencoder_path, map_location='cpu')
        ae_config = ae_checkpoint.get('config', {})
        
        autoencoder = ToolInvocationAutoencoder(
            embedding_dim=ae_config.get('embedding_dim', 256),
            encoder_model=ae_config.get('encoder_model', 'google/flan-t5-base'),
            decoder_model=ae_config.get('decoder_model', 'google/flan-t5-base'),
            max_length=ae_config.get('max_length', 256),
            torch_dtype=ae_config.get('torch_dtype', 'bfloat16')
        )
        autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        
        # Load LLM if provided
        llm_model = None
        if llm_path:
            print(f"Loading LLM from {llm_path}...")
            llm_checkpoint = torch.load(llm_path, map_location='cpu')
            llm_config = llm_checkpoint.get('config', {})
            
            llm_model = ToolPredictionLLM(
                base_model=llm_config.get('base_model', 'google/flan-t5-base'),
                embedding_dim=ae_config.get('embedding_dim', 256),
                num_tools=6,
                torch_dtype=llm_config.get('torch_dtype', 'bfloat16')
            )
            llm_model.load_state_dict(llm_checkpoint['model_state_dict'])
            llm_model.decoder = autoencoder.decoder  # Share decoder
            llm_model = llm_model.to(device)
            llm_model.eval()
        
        return cls(
            llm_model=llm_model,
            autoencoder=autoencoder,
            device=device
        )
    
    def _parse_tool_call(self, tool_call_str: str) -> Dict:
        """Parse tool call string to structured format."""
        try:
            # Try JSON format first
            data = json.loads(tool_call_str)
            return {
                "tool_name": data.get("tool", "unknown"),
                "arguments": data.get("arguments", {})
            }
        except json.JSONDecodeError:
            pass
        
        # Try Python format: tool_name(arg1='val1', arg2=val2)
        try:
            import re
            match = re.match(r'(\w+)\((.*)\)', tool_call_str, re.DOTALL)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                
                # Simple argument parsing
                arguments = {}
                # Match key='value' or key=value patterns
                arg_pattern = r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\d+(?:\.\d+)?)|(\w+))"
                for m in re.finditer(arg_pattern, args_str):
                    key = m.group(1)
                    # Try each capture group for value
                    value = m.group(2) or m.group(3) or m.group(4) or m.group(5)
                    # Try to convert to number
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass
                    arguments[key] = value
                
                return {
                    "tool_name": tool_name,
                    "arguments": arguments
                }
        except Exception:
            pass
        
        return {
            "tool_name": "unknown",
            "arguments": {}
        }
    
    def predict(
        self,
        query: str,
        return_embedding: bool = False
    ) -> Union[ToolCallResult, str]:
        """
        Predict tool call from natural language query.
        
        Args:
            query: Natural language query
            return_embedding: Whether to include embedding in result
            
        Returns:
            ToolCallResult with prediction
        """
        if self.llm_model is None:
            raise ValueError("LLM model not loaded. Provide llm_path to from_pretrained()")
        
        self.llm_model.eval()
        
        with torch.no_grad():
            # Get prediction
            result = self.llm_model(nl_queries=[query])
            embedding = result["embedding"][0]
            tool_logits = result["tool_logits"][0]
            
            # Get tool classification confidence
            tool_probs = torch.softmax(tool_logits, dim=0)
            pred_tool_idx = tool_probs.argmax().item()
            confidence = tool_probs[pred_tool_idx].item()
            
            # Decode embedding to tool call
            decoder_result = self.llm_model.decoder(embedding.unsqueeze(0))
            generated_ids = decoder_result["generated_ids"][0]
            tool_call_str = self.llm_model.decoder.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Parse tool call
            parsed = self._parse_tool_call(tool_call_str)
            
            return ToolCallResult(
                tool_call=tool_call_str,
                tool_name=parsed["tool_name"],
                arguments=parsed["arguments"],
                embedding=embedding.cpu() if return_embedding else None,
                confidence=confidence
            )
    
    def predict_batch(
        self,
        queries: List[str],
        return_embeddings: bool = False
    ) -> List[ToolCallResult]:
        """
        Predict tool calls for multiple queries.
        
        Args:
            queries: List of natural language queries
            return_embeddings: Whether to include embeddings in results
            
        Returns:
            List of ToolCallResult
        """
        if self.llm_model is None:
            raise ValueError("LLM model not loaded")
        
        self.llm_model.eval()
        
        with torch.no_grad():
            # Get predictions
            result = self.llm_model(nl_queries=queries)
            embeddings = result["embedding"]
            tool_logits = result["tool_logits"]
            
            # Decode all embeddings
            decoder_result = self.llm_model.decoder(embeddings)
            generated_ids = decoder_result["generated_ids"]
            
            results = []
            for i, (emb, logits, ids) in enumerate(zip(embeddings, tool_logits, generated_ids)):
                # Tool confidence
                tool_probs = torch.softmax(logits, dim=0)
                pred_tool_idx = tool_probs.argmax().item()
                confidence = tool_probs[pred_tool_idx].item()
                
                # Decode
                tool_call_str = self.llm_model.decoder.tokenizer.decode(
                    ids, skip_special_tokens=True
                )
                
                # Parse
                parsed = self._parse_tool_call(tool_call_str)
                
                results.append(ToolCallResult(
                    tool_call=tool_call_str,
                    tool_name=parsed["tool_name"],
                    arguments=parsed["arguments"],
                    embedding=emb.cpu() if return_embeddings else None,
                    confidence=confidence
                ))
            
            return results
    
    def encode_tool_call(self, tool_call: str) -> torch.Tensor:
        """
        Encode a tool call string to embedding using the autoencoder.
        
        Useful for analyzing the embedding space.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            embedding = self.autoencoder.encode([tool_call])[0]
        return embedding.cpu()
    
    def decode_embedding(self, embedding: torch.Tensor) -> str:
        """
        Decode an embedding to tool call string.
        
        Useful for exploring the embedding space.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            embedding = embedding.to(self.device)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            tool_calls = self.autoencoder.decode(embedding)
        return tool_calls[0]
    
    def reconstruct_tool_call(self, tool_call: str) -> str:
        """
        Reconstruct a tool call through the autoencoder.
        
        Useful for testing autoencoder quality.
        """
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder.reconstruct([tool_call])
        return reconstructed[0]


def demo():
    """Demo the tool calling system."""
    import sys
    
    print("=" * 60)
    print("NTILC Tool Calling System Demo")
    print("=" * 60)
    
    # Check if models exist
    autoencoder_path = "checkpoints/best_model.pt"
    llm_path = "checkpoints/llm_integration/best_model.pt"
    
    import os
    if not os.path.exists(autoencoder_path):
        print(f"\nAutoencoder not found at {autoencoder_path}")
        print("Please train the autoencoder first:")
        print("  python training/train_autoencoder.py")
        return
    
    # Load system (autoencoder only if LLM not available)
    if os.path.exists(llm_path):
        print("\nLoading full system (autoencoder + LLM)...")
        system = ToolCallingSystem.from_pretrained(
            autoencoder_path=autoencoder_path,
            llm_path=llm_path
        )
    else:
        print(f"\nLLM not found at {llm_path}")
        print("Loading autoencoder only...")
        system = ToolCallingSystem.from_pretrained(
            autoencoder_path=autoencoder_path
        )
    
    # Demo autoencoder reconstruction
    print("\n" + "=" * 40)
    print("Autoencoder Reconstruction Demo")
    print("=" * 40)
    
    test_tool_calls = [
        '{"tool": "search", "arguments": {"query": "machine learning", "max_results": 10}}',
        '{"tool": "calculate", "arguments": {"expression": "sqrt(144)"}}',
        '{"tool": "database_query", "arguments": {"sql": "SELECT * FROM users LIMIT 5", "timeout": 30}}'
    ]
    
    for tc in test_tool_calls:
        reconstructed = system.reconstruct_tool_call(tc)
        match = tc == reconstructed
        print(f"\nOriginal:      {tc[:80]}...")
        print(f"Reconstructed: {reconstructed[:80]}...")
        print(f"Match: {'✓' if match else '✗'}")
    
    # Demo full inference if LLM available
    if system.llm_model is not None:
        print("\n" + "=" * 40)
        print("Natural Language to Tool Call Demo")
        print("=" * 40)
        
        test_queries = [
            "Find me information about machine learning",
            "What is 25 plus 37?",
            "Get the last 10 orders from California",
            "Send an email to test@example.com about the meeting",
            "Fetch data from the GitHub API"
        ]
        
        for query in test_queries:
            result = system.predict(query)
            print(f"\nQuery: {query}")
            print(f"Tool: {result.tool_name} (confidence: {result.confidence:.2f})")
            print(f"Arguments: {result.arguments}")
            print(f"Raw: {result.tool_call[:80]}...")
    else:
        print("\nTo enable NL -> Tool Call inference, train the LLM integration:")
        print("  python training/train_llm_integration.py")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
