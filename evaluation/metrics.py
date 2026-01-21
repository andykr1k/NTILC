"""
Evaluation metrics for NTILC autoencoder.
Supports both JSON and Python format tool calls.
"""

import torch
import re
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict


def extract_tool_from_call(tool_call: str) -> str:
    """
    Extract tool name from tool call string.
    Handles both JSON and Python formats.
    
    Args:
        tool_call: Tool call string
        
    Returns:
        Tool name or empty string if not found
    """
    tool_call = tool_call.strip()
    
    # Try JSON format first
    if tool_call.startswith('{'):
        try:
            parsed = json.loads(tool_call)
            return parsed.get("tool", "")
        except json.JSONDecodeError:
            pass
    
    # Try Python format: tool_name(...)
    match = re.match(r'(\w+)\s*\(', tool_call)
    return match.group(1) if match else ""


def extract_parameters_from_call(tool_call: str) -> Dict[str, Any]:
    """
    Extract parameters from tool call string.
    Handles both JSON and Python formats.
    
    Args:
        tool_call: Tool call string
        
    Returns:
        Dictionary mapping parameter names to (type, value) tuples
    """
    tool_call = tool_call.strip()
    params = {}
    
    # Try JSON format first
    if tool_call.startswith('{'):
        try:
            parsed = json.loads(tool_call)
            arguments = parsed.get("arguments", {})
            for name, value in arguments.items():
                if isinstance(value, str):
                    params[name] = ("str", value)
                elif isinstance(value, int):
                    params[name] = ("int", value)
                elif isinstance(value, float):
                    params[name] = ("float", value)
                elif isinstance(value, bool):
                    params[name] = ("bool", value)
                elif isinstance(value, list):
                    params[name] = ("list", value)
                elif isinstance(value, dict):
                    params[name] = ("dict", value)
            return params
        except json.JSONDecodeError:
            pass
    
    # Python format extraction
    # Extract string parameters: param='value' or param="value"
    string_params = re.findall(r"(\w+)='([^']*)'", tool_call)
    for name, value in string_params:
        params[name] = ("str", value)
    
    string_params_dq = re.findall(r'(\w+)="([^"]*)"', tool_call)
    for name, value in string_params_dq:
        if name not in params:
            params[name] = ("str", value)
    
    # Extract integer parameters: param=123
    int_params = re.findall(r"(\w+)=(\d+)(?![.\w])", tool_call)
    for name, value in int_params:
        if name not in params:
            params[name] = ("int", int(value))
    
    return params


def exact_match_accuracy(original: List[str], reconstructed: List[str]) -> float:
    """
    Compute exact string match accuracy.
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        
    Returns:
        accuracy: Fraction of exact matches
    """
    assert len(original) == len(reconstructed)
    
    matches = sum(1 for orig, recon in zip(original, reconstructed) if orig == recon)
    return matches / len(original) if len(original) > 0 else 0.0


def tool_accuracy(original: List[str], reconstructed: List[str]) -> float:
    """
    Compute tool selection accuracy (ignoring parameters).
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        
    Returns:
        accuracy: Fraction of correct tool selections
    """
    assert len(original) == len(reconstructed)
    
    correct = 0
    for orig, recon in zip(original, reconstructed):
        orig_tool = extract_tool_from_call(orig)
        recon_tool = extract_tool_from_call(recon)
        if orig_tool == recon_tool and orig_tool != "":
            correct += 1
    
    return correct / len(original) if len(original) > 0 else 0.0


def parameter_accuracy(original: List[str], reconstructed: List[str]) -> Dict[str, float]:
    """
    Compute parameter accuracy by type.
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        
    Returns:
        Dictionary with accuracy for each parameter type
    """
    assert len(original) == len(reconstructed)
    
    param_counts = defaultdict(int)
    param_correct = defaultdict(int)
    
    for orig, recon in zip(original, reconstructed):
        orig_params = extract_parameters_from_call(orig)
        recon_params = extract_parameters_from_call(recon)
        
        for param_name, (param_type, param_value) in orig_params.items():
            param_counts[param_type] += 1
            if param_name in recon_params:
                recon_type, recon_value = recon_params[param_name]
                if param_type == recon_type and param_value == recon_value:
                    param_correct[param_type] += 1
    
    accuracies = {}
    for param_type in param_counts:
        accuracies[f"param_{param_type}_accuracy"] = (
            param_correct[param_type] / param_counts[param_type]
            if param_counts[param_type] > 0 else 0.0
        )
    
    return accuracies


def embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about embedding space.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        
    Returns:
        Dictionary with embedding statistics
    """
    stats = {}
    
    # Norm statistics
    norms = torch.norm(embeddings, dim=1)
    stats["embedding_mean_norm"] = norms.mean().item()
    stats["embedding_std_norm"] = norms.std().item()
    stats["embedding_min_norm"] = norms.min().item()
    stats["embedding_max_norm"] = norms.max().item()
    
    # Variance statistics
    stats["embedding_mean_variance"] = embeddings.var(dim=0).mean().item()
    stats["embedding_std_variance"] = embeddings.var(dim=0).std().item()
    
    # Per-dimension statistics
    stats["embedding_mean_per_dim"] = embeddings.mean(dim=0).mean().item()
    stats["embedding_std_per_dim"] = embeddings.std(dim=0).mean().item()
    
    return stats


def compute_metrics(
    original: List[str],
    reconstructed: List[str],
    embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        embeddings: Tensor of embeddings (batch_size, embedding_dim)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Reconstruction metrics
    metrics["exact_match_accuracy"] = exact_match_accuracy(original, reconstructed)
    metrics["tool_accuracy"] = tool_accuracy(original, reconstructed)
    
    # Parameter metrics
    param_metrics = parameter_accuracy(original, reconstructed)
    metrics.update(param_metrics)
    
    # Embedding statistics
    embedding_stats = embedding_statistics(embeddings)
    metrics.update(embedding_stats)
    
    return metrics


def per_tool_metrics(original: List[str], reconstructed: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per tool type.
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        
    Returns:
        Dictionary mapping tool name to metrics
    """
    # Group by tool
    tool_groups = defaultdict(lambda: {"original": [], "reconstructed": []})
    for orig, recon in zip(original, reconstructed):
        tool = extract_tool_from_call(orig)
        if not tool:
            tool = "unknown"
        tool_groups[tool]["original"].append(orig)
        tool_groups[tool]["reconstructed"].append(recon)
    
    # Compute metrics per tool
    per_tool = {}
    for tool, data in tool_groups.items():
        if len(data["original"]) > 0:
            per_tool[tool] = {
                "count": len(data["original"]),
                "exact_match_accuracy": exact_match_accuracy(
                    data["original"], data["reconstructed"]
                ),
                "tool_accuracy": tool_accuracy(
                    data["original"], data["reconstructed"]
                )
            }
    
    return per_tool


def semantic_similarity(original: List[str], reconstructed: List[str]) -> Dict[str, float]:
    """
    Compute semantic similarity metrics (useful for string parameters).
    
    Args:
        original: List of original tool calls
        reconstructed: List of reconstructed tool calls
        
    Returns:
        Dictionary with semantic similarity metrics
    """
    from difflib import SequenceMatcher
    
    # Extract all string parameters and compute similarity
    similarities = []
    
    for orig, recon in zip(original, reconstructed):
        orig_params = extract_parameters_from_call(orig)
        recon_params = extract_parameters_from_call(recon)
        
        for param_name, (param_type, param_value) in orig_params.items():
            if param_type == "str" and param_name in recon_params:
                recon_type, recon_value = recon_params[param_name]
                if recon_type == "str":
                    ratio = SequenceMatcher(None, str(param_value), str(recon_value)).ratio()
                    similarities.append(ratio)
    
    if similarities:
        return {
            "string_param_similarity_mean": sum(similarities) / len(similarities),
            "string_param_similarity_min": min(similarities),
            "string_param_similarity_max": max(similarities)
        }
    return {
        "string_param_similarity_mean": 0.0,
        "string_param_similarity_min": 0.0,
        "string_param_similarity_max": 0.0
    }
