"""
Loss functions for NTILC training.

Includes:
- Contrastive loss for embedding diversity
- Embedding regularization losses

FIXED: Corrected Circle Loss optimal points and proper gradient handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to prevent embedding collapse.
    
    Uses InfoNCE-style loss to push different tool calls apart
    while keeping similar tool calls close together.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        use_tool_labels: bool = True
    ):
        """
        Args:
            temperature: Temperature for softmax scaling
            margin: Margin for triplet-style loss
            use_tool_labels: If True, same tool = positive, different tool = negative
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_tool_labels = use_tool_labels
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss on embeddings.
        
        Args:
            embeddings: (batch_size, embedding_dim) tensor
            labels: (batch_size,) tensor of class labels
            
        Returns:
            Scalar loss tensor
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if batch_size < 2:
            return embeddings.new_zeros(1, requires_grad=True)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)  # (B, B)
        
        # Use labels if provided
        if self.use_tool_labels and labels is not None:
            # Create positive/negative masks based on tool type
            positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Same tool
            negative_mask = ~positive_mask
            
            # Remove diagonal (self-similarity)
            diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            positive_mask = positive_mask & ~diagonal_mask
            negative_mask = negative_mask & ~diagonal_mask
            
            # InfoNCE-style loss
            sim_scaled = similarity_matrix / self.temperature
            
            # Compute loss for samples that have both positives and negatives
            loss = embeddings.new_zeros(1, requires_grad=True)
            valid_samples = 0
            
            for i in range(batch_size):
                pos_mask_i = positive_mask[i]
                neg_mask_i = negative_mask[i]
                
                if pos_mask_i.sum() > 0 and neg_mask_i.sum() > 0:
                    # Get positive and negative similarities
                    pos_sim = sim_scaled[i][pos_mask_i]
                    neg_sim = sim_scaled[i][neg_mask_i]
                    
                    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                    # Average over all positives
                    for pos_s in pos_sim:
                        # Use log-sum-exp for numerical stability
                        logits = torch.cat([pos_s.unsqueeze(0), neg_sim])
                        loss = loss - F.log_softmax(logits, dim=0)[0]
                        valid_samples += 1
            
            if valid_samples > 0:
                loss = loss / valid_samples
            
            return loss
        
        else:
            # Without labels: use uniformity loss to spread embeddings
            # Penalize high similarity between different samples
            
            # Remove diagonal
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            off_diagonal_sim = similarity_matrix[mask].view(batch_size, batch_size - 1)
            
            # Uniformity loss: push all pairs apart
            uniformity_loss = torch.log(torch.exp(off_diagonal_sim / self.temperature).mean() + 1e-8)
            
            return uniformity_loss


class CircleLoss(nn.Module):
    """
    Circle Loss for metric learning and deep feature learning.
    
    Paper: "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    https://arxiv.org/abs/2002.10857
    
    Circle Loss unifies pair-based and class-based similarity optimization,
    providing flexible optimization boundaries for better feature learning.
    """
    
    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 64,
        similarity_type: str = "cosine"
    ):
        """
        Args:
            margin: Margin for similarity optimization (m in paper)
            gamma: Scale factor for logits (gamma in paper)
            similarity_type: Type of similarity measure ('cosine' or 'euclidean')
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.similarity_type = similarity_type
        
        # FIXED: Correct optimal similarity thresholds for cosine similarity [-1, 1]
        self.O_p = 1 - margin      # Optimal positive similarity (e.g., 0.75 for margin=0.25)
        self.O_n = -1 + margin     # Optimal negative similarity (e.g., -0.75 for margin=0.25)
        
        # Delta values for weighting
        self.Delta_p = 1 - margin  # 0.75
        self.Delta_n = margin      # 0.25
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Circle Loss.
        
        Args:
            embeddings: (batch_size, embedding_dim) embeddings (will be normalized)
            labels: (batch_size,) class labels for each embedding
            
        Returns:
            Scalar loss tensor
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if batch_size < 2:
            return embeddings.new_zeros(1, requires_grad=True)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix: (B, B)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # Create masks for positive and negative pairs
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()  # Same class
        negative_mask = (labels != labels.T).float()  # Different class
        
        # Remove diagonal (self-similarity)
        diagonal_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        positive_mask = positive_mask * (1 - diagonal_mask)
        negative_mask = negative_mask * (1 - diagonal_mask)
        
        # Check if we have valid pairs
        num_positives = positive_mask.sum()
        num_negatives = negative_mask.sum()
        
        if num_positives == 0 or num_negatives == 0:
            return embeddings.new_zeros(1, requires_grad=True)
        
        # Get positive and negative similarities
        sp = similarity_matrix * positive_mask  # Positive pairs
        sn = similarity_matrix * negative_mask  # Negative pairs
        
        # Compute alpha (weighting factors) - detach to stop gradients through alpha
        alpha_p = torch.clamp(self.O_p - sp.detach(), min=0.0)
        alpha_n = torch.clamp(sn.detach() - self.O_n, min=0.0)
        
        # Compute weighted logits
        logit_p = -self.gamma * alpha_p * (sp - self.Delta_p)
        logit_n = self.gamma * alpha_n * (sn - self.Delta_n)
        
        # Apply masks
        logit_p = logit_p * positive_mask
        logit_n = logit_n * negative_mask
        
        # Gather all positive and negative logits across the batch
        pos_logits = logit_p[positive_mask > 0]
        neg_logits = logit_n[negative_mask > 0]
        
        # Circle loss formulation: log(1 + exp(sum(logit_p) + sum(logit_n)))
        # Use logsumexp for numerical stability
        if len(pos_logits) > 0 and len(neg_logits) > 0:
            pos_term = torch.logsumexp(pos_logits, dim=0)
            neg_term = torch.logsumexp(neg_logits, dim=0)
            loss = F.softplus(pos_term + neg_term)
        else:
            loss = embeddings.new_zeros(1, requires_grad=True)
        
        return loss


class CircleLossV2(nn.Module):
    """
    Alternative Circle Loss implementation using per-sample computation.
    More similar to triplet loss formulation.
    """
    
    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 64,
        similarity_type: str = "cosine"
    ):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.similarity_type = similarity_type
        
        # FIXED: Correct optimal points
        self.O_p = 1 - margin
        self.O_n = -1 + margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute Circle Loss per sample and average."""
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if batch_size < 2:
            return embeddings.new_zeros(1, requires_grad=True)
        
        # Normalize
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # Create masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        diagonal_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        positive_mask = positive_mask * (1 - diagonal_mask)
        negative_mask = negative_mask * (1 - diagonal_mask)
        
        # Per-sample loss
        losses = []
        for i in range(batch_size):
            pos_mask_i = positive_mask[i] > 0
            neg_mask_i = negative_mask[i] > 0
            
            if pos_mask_i.sum() == 0 or neg_mask_i.sum() == 0:
                continue
            
            # Get similarities for this sample
            pos_sim = similarity_matrix[i][pos_mask_i]
            neg_sim = similarity_matrix[i][neg_mask_i]
            
            # Compute alpha weights with correct optimal points
            alpha_p = torch.clamp(self.O_p - pos_sim.detach(), min=0.0)
            alpha_n = torch.clamp(neg_sim.detach() - self.O_n, min=0.0)
            
            # Weighted logits
            logit_p = -self.gamma * alpha_p * (pos_sim - (1 - self.margin))
            logit_n = self.gamma * alpha_n * (neg_sim - (-1 + self.margin))
            
            # Circle loss for this sample
            pos_term = torch.logsumexp(logit_p, dim=0)
            neg_term = torch.logsumexp(logit_n, dim=0)
            sample_loss = F.softplus(pos_term + neg_term)
            losses.append(sample_loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return embeddings.new_zeros(1, requires_grad=True)


class EmbeddingRegularizationLoss(nn.Module):
    """
    Regularization losses for embeddings to encourage diversity
    and prevent collapse.
    """
    
    def __init__(
        self,
        l2_weight: float = 0.001,
        variance_weight: float = 0.01,
        target_norm: float = 1.0,
        target_variance: float = 0.5
    ):
        """
        Args:
            l2_weight: Weight for L2 regularization (keep embeddings bounded)
            variance_weight: Weight for variance loss (encourage diversity)
            target_norm: Target norm for embeddings
            target_variance: Target variance per dimension (for normalized embeddings, ~0.5 is good)
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.variance_weight = variance_weight
        self.target_norm = target_norm
        self.target_variance = target_variance
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        
        Args:
            embeddings: (batch_size, embedding_dim) tensor
            
        Returns:
            Dictionary with individual loss terms
        """
        device = embeddings.device
        batch_size, embed_dim = embeddings.shape
        
        losses = {}
        
        # L2 regularization: keep embeddings bounded
        norms = torch.norm(embeddings, p=2, dim=1)
        l2_loss = ((norms - self.target_norm) ** 2).mean()
        losses["l2_loss"] = l2_loss * self.l2_weight
        
        # Variance loss: encourage diversity across batch
        if batch_size > 1:
            # Per-dimension variance across batch
            var_per_dim = embeddings.var(dim=0)  # (embed_dim,)
            mean_var = var_per_dim.mean()
            
            # FIXED: Better variance loss that penalizes deviation from target
            # This prevents both collapse (too low) and explosion (too high)
            variance_loss = (mean_var - self.target_variance) ** 2
            losses["variance_loss"] = variance_loss * self.variance_weight
        else:
            losses["variance_loss"] = embeddings.new_zeros(1)
        
        # Total regularization loss
        losses["total_reg_loss"] = losses["l2_loss"] + losses["variance_loss"]
        
        return losses
