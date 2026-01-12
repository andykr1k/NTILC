"""
Evaluation package for NTILC.
"""

from .metrics import (
    compute_metrics,
    exact_match_accuracy,
    tool_accuracy,
    parameter_accuracy,
    embedding_statistics,
    per_tool_metrics
)
from .visualizations import (
    visualize_embeddings_2d,
    visualize_embedding_distances,
    visualize_embedding_norms,
    analyze_embedding_space
)

__all__ = [
    "compute_metrics",
    "exact_match_accuracy",
    "tool_accuracy",
    "parameter_accuracy",
    "embedding_statistics",
    "per_tool_metrics",
    "visualize_embeddings_2d",
    "visualize_embedding_distances",
    "visualize_embedding_norms",
    "analyze_embedding_space"
]
