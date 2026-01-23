"""
Cluster Retrieval System for NTILC.

At inference, retrieves cluster IDs instead of generating tool calls.
No decoder needed - just similarity computation against cluster centroids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ClusterRetrieval(nn.Module):
    """
    Retrieves cluster IDs based on similarity to cluster centroids.
    
    At inference:
    1. Query embedding (128-D) → compute similarity to all cluster centroids
    2. Select top cluster(s) → return cluster ID(s)
    3. No decoder or autoregressive generation
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_clusters: int = None,
        similarity_type: str = "cosine"
    ):
        """
        Args:
            embedding_dim: Dimension of projected embeddings (128)
            num_clusters: Number of clusters (can be None if using dynamic clusters)
            similarity_type: "cosine" or "euclidean"
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.similarity_type = similarity_type
        
        # Cluster centroids (learned or computed)
        if num_clusters is not None:
            self.register_buffer(
                'cluster_centroids',
                torch.randn(num_clusters, embedding_dim)
            )
            # Normalize centroids
            self.cluster_centroids = F.normalize(self.cluster_centroids, p=2, dim=1)
        else:
            self.cluster_centroids = None
    
    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        cluster_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between queries and clusters.
        
        Args:
            query_embeddings: (batch_size, embedding_dim)
            cluster_embeddings: (num_clusters, embedding_dim)
        
        Returns:
            similarities: (batch_size, num_clusters)
        """
        if self.similarity_type == "cosine":
            # Both should be normalized
            similarities = torch.matmul(query_embeddings, cluster_embeddings.T)
        else:  # euclidean
            # Convert to distance, then to similarity
            distances = torch.cdist(query_embeddings, cluster_embeddings, p=2)
            similarities = -distances
        
        return similarities
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        cluster_embeddings: Optional[torch.Tensor] = None,
        top_k: int = 1,
        threshold: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve top-k cluster IDs for queries.
        
        Args:
            query_embeddings: (batch_size, embedding_dim) projected query embeddings
            cluster_embeddings: (num_clusters, embedding_dim) cluster centroids (if None, uses learned)
            top_k: Number of top clusters to return
            threshold: Minimum similarity threshold
        
        Returns:
            Dictionary with:
                - cluster_ids: (batch_size, top_k) cluster IDs
                - similarities: (batch_size, top_k) similarity scores
                - all_similarities: (batch_size, num_clusters) all similarities
        """
        if cluster_embeddings is None:
            if self.cluster_centroids is None:
                raise ValueError("Must provide cluster_embeddings or initialize with num_clusters")
            cluster_embeddings = self.cluster_centroids
        
        # Compute similarities
        similarities = self.compute_similarity(query_embeddings, cluster_embeddings)
        
        # Get top-k
        top_k_similarities, top_k_indices = torch.topk(similarities, k=min(top_k, similarities.shape[1]), dim=1)
        
        # Apply threshold
        mask = top_k_similarities >= threshold
        top_k_indices = top_k_indices * mask + (-1) * (~mask)  # -1 for below threshold
        
        return {
            "cluster_ids": top_k_indices,
            "similarities": top_k_similarities,
            "all_similarities": similarities
        }
    
    def update_cluster_centroids(
        self,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor
    ):
        """
        Update cluster centroids from embeddings and assignments.
        
        Args:
            embeddings: (num_samples, embedding_dim)
            cluster_assignments: (num_samples,) cluster IDs
        """
        num_clusters = cluster_assignments.max().item() + 1
        
        # Compute centroids
        centroids = []
        for cluster_id in range(num_clusters):
            mask = cluster_assignments == cluster_id
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(dim=0)
                centroids.append(centroid)
            else:
                # Random initialization if no samples
                centroids.append(torch.randn(self.embedding_dim, device=embeddings.device))
        
        self.cluster_centroids = torch.stack(centroids)
        self.cluster_centroids = F.normalize(self.cluster_centroids, p=2, dim=1)
    
    def get_cluster_centroids(self) -> torch.Tensor:
        """Get current cluster centroids."""
        if self.cluster_centroids is None:
            raise ValueError("No cluster centroids initialized")
        return self.cluster_centroids.clone()
