"""
Ablation studies for NTILC: Baseline comparisons before full training.
"""

from .baseline_naive import NaivePromptingBaseline
from .baseline_cross_attention import CrossAttentionLLM
from .evaluate_baselines import evaluate_baseline

__all__ = [
    "NaivePromptingBaseline",
    "CrossAttentionLLM",
    "evaluate_baseline"
]
