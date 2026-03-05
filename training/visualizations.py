"""
Visualization tools for analyzing NTILC embedding space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import re


def extract_tool(tool_call) -> str:
    """Extract tool name from tool call string or dict."""
    if isinstance(tool_call, dict):
        return tool_call.get("tool", "unknown")
    tool_call = str(tool_call).strip()
    if tool_call.startswith("{"):
        try:
            import json
            parsed = json.loads(tool_call)
            return parsed.get("tool", "unknown")
        except json.JSONDecodeError:
            pass
    match = re.match(r'(\w+)\(', tool_call)
    return match.group(1) if match else "unknown"


def visualize_embeddings_2d(
    embeddings: torch.Tensor,
    tool_calls: List[str],
    method: str = "tsne",
    n_components: int = 2,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        tool_calls: List of tool call strings for coloring
        method: Reduction method ("pca", "tsne", "umap")
        n_components: Number of components (should be 2 for visualization)
        output_path: Path to save figure
        figsize: Figure size
    """
    embeddings_np = embeddings.cpu().numpy()
    
    # Reduce dimensionality
    if method == "pca":
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings_np)
        title = f"PCA Visualization (explained variance: {reducer.explained_variance_ratio_.sum():.2%})"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings_np)
        title = "t-SNE Visualization"
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings_np)
        title = "UMAP Visualization"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract tool names for coloring
    tool_names = [extract_tool(tc) for tc in tool_calls]
    unique_tools = list(set(tool_names))
    tool_to_idx = {tool: idx for idx, tool in enumerate(unique_tools)}
    colors = [tool_to_idx[tool] for tool in tool_names]
    
    # Create plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=colors,
        cmap="tab10",
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label="Tool Type")
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=plt.cm.tab10(i), markersize=10)
               for i in range(len(unique_tools))]
    plt.legend(handles, unique_tools, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_embedding_distances(
    embeddings: torch.Tensor,
    tool_calls: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize pairwise distances between embeddings, grouped by tool type.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        tool_calls: List of tool call strings
        output_path: Path to save figure
        figsize: Figure size
    """
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings).cpu().numpy()
    
    # Extract tool names
    tool_names = [extract_tool(tc) for tc in tool_calls]
    unique_tools = sorted(list(set(tool_names)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Distance matrix
    im = axes[0].imshow(distances, cmap='viridis', aspect='auto')
    axes[0].set_title("Pairwise Embedding Distances")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")
    plt.colorbar(im, ax=axes[0])
    
    # Plot 2: Distance distribution by tool type
    tool_distances = {tool: [] for tool in unique_tools}
    
    for i, tool_i in enumerate(tool_names):
        for j, tool_j in enumerate(tool_names):
            if i < j:  # Avoid duplicates
                if tool_i == tool_j:
                    tool_distances[tool_i].append(distances[i, j])
    
    # Box plot
    data_to_plot = [tool_distances[tool] for tool in unique_tools if tool_distances[tool]]
    labels = [tool for tool in unique_tools if tool_distances[tool]]
    
    axes[1].boxplot(data_to_plot, labels=labels)
    axes[1].set_title("Intra-Tool Distance Distribution")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_embedding_norms(
    embeddings: torch.Tensor,
    tool_calls: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Visualize embedding norms by tool type.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        tool_calls: List of tool call strings
        output_path: Path to save figure
        figsize: Figure size
    """
    # Compute norms
    norms = torch.norm(embeddings, dim=1).cpu().numpy()
    
    # Extract tool names
    tool_names = [extract_tool(tc) for tc in tool_calls]
    unique_tools = sorted(list(set(tool_names)))
    
    # Group norms by tool
    tool_norms = {tool: [] for tool in unique_tools}
    for norm, tool in zip(norms, tool_names):
        tool_norms[tool].append(norm)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    data_to_plot = [tool_norms[tool] for tool in unique_tools]
    positions = range(len(unique_tools))
    
    bp = plt.boxplot(data_to_plot, labels=unique_tools, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tools)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title("Embedding Norm Distribution by Tool Type")
    plt.ylabel("L2 Norm")
    plt.xlabel("Tool Type")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_embedding_space(
    embeddings: torch.Tensor,
    tool_calls: List[str],
    output_dir: str = "./visualizations"
):
    """
    Generate comprehensive embedding space analysis.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        tool_calls: List of tool call strings
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating embedding space visualizations...")
    
    # 2D visualizations
    print("  - PCA visualization...")
    visualize_embeddings_2d(
        embeddings, tool_calls, method="pca",
        output_path=os.path.join(output_dir, "pca_visualization.png")
    )
    
    print("  - t-SNE visualization...")
    visualize_embeddings_2d(
        embeddings, tool_calls, method="tsne",
        output_path=os.path.join(output_dir, "tsne_visualization.png")
    )
    
    if UMAP_AVAILABLE:
        print("  - UMAP visualization...")
        visualize_embeddings_2d(
            embeddings, tool_calls, method="umap",
            output_path=os.path.join(output_dir, "umap_visualization.png")
        )
    
    # Distance analysis
    print("  - Distance analysis...")
    visualize_embedding_distances(
        embeddings, tool_calls,
        output_path=os.path.join(output_dir, "distance_analysis.png")
    )
    
    # Norm analysis
    print("  - Norm analysis...")
    visualize_embedding_norms(
        embeddings, tool_calls,
        output_path=os.path.join(output_dir, "norm_analysis.png")
    )
    
    print(f"All visualizations saved to {output_dir}")
