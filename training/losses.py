"""
Loss functions for NTILC training.

Includes:
- Contrastive loss for embedding diversity
- Embedding regularization losses
- Combined loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import json


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
        tool_calls: List[str] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss on embeddings.
        
        Args:
            embeddings: (batch_size, embedding_dim) tensor
            tool_calls: Optional list of tool call strings for extracting labels
            
        Returns:
            Scalar loss tensor
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)  # (B, B)
        
        # Extract tool labels if provided
        if self.use_tool_labels and tool_calls is not None:
            tool_labels = self._extract_tool_labels(tool_calls)
            
            # Create positive/negative masks based on tool type
            labels = torch.tensor([self._tool_to_idx(t) for t in tool_labels], device=device)
            positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Same tool
            negative_mask = ~positive_mask
            
            # Remove diagonal (self-similarity)
            diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            positive_mask = positive_mask & ~diagonal_mask
            negative_mask = negative_mask & ~diagonal_mask
            
            # InfoNCE-style loss
            # For each sample, maximize similarity to positives, minimize to negatives
            sim_scaled = similarity_matrix / self.temperature
            
            # Compute loss for samples that have both positives and negatives
            loss = torch.tensor(0.0, device=device)
            valid_samples = 0
            
            for i in range(batch_size):
                pos_mask_i = positive_mask[i]
                neg_mask_i = negative_mask[i]
                
                if pos_mask_i.sum() > 0 and neg_mask_i.sum() > 0:
                    # Get positive and negative similarities
                    pos_sim = sim_scaled[i][pos_mask_i]
                    neg_sim = sim_scaled[i][neg_mask_i]
                    
                    # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
                    # Average over all positives
                    for pos_s in pos_sim:
                        numerator = torch.exp(pos_s)
                        denominator = numerator + torch.exp(neg_sim).sum()
                        loss -= torch.log(numerator / (denominator + 1e-8))
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
            # Using Gaussian potential: exp(-t * ||x - y||^2)
            # Approximated via cosine similarity
            uniformity_loss = torch.log(torch.exp(off_diagonal_sim / self.temperature).mean() + 1e-8)
            
            return uniformity_loss
    
    def _extract_tool_labels(self, tool_calls: List[str]) -> List[str]:
        """Extract tool name from tool call strings."""
        labels = []
        for tc in tool_calls:
            try:
                # Try JSON format first
                data = json.loads(tc)
                labels.append(data.get("tool", "unknown"))
            except (json.JSONDecodeError, TypeError):
                # Try Python format: tool_name(...)
                if "(" in tc:
                    labels.append(tc.split("(")[0].strip())
                else:
                    labels.append("unknown")
        return labels
    
    def _tool_to_idx(self, tool_name: str) -> int:
        """Convert tool name to index."""
        tools = ["search", "calculate", "database_query", "send_email", "web_fetch", "file_read"]
        try:
            return tools.index(tool_name)
        except ValueError:
            return -1

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
        gamma: float = 256,
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
        
        # Optimal similarity thresholds
        self.O_p = 1 + margin  # Optimal positive similarity
        self.O_n = -margin      # Optimal negative similarity
        
        # Delta values for weighting
        self.Delta_p = 1 - margin
        self.Delta_n = margin
    
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
            return torch.tensor(0.0, device=device, requires_grad=True)
        
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
            # Fallback: use simple contrastive loss
            print(f"Warning: No valid pairs (pos={num_positives}, neg={num_negatives})")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Get positive and negative similarities
        sp = similarity_matrix * positive_mask  # Positive pairs
        sn = similarity_matrix * negative_mask  # Negative pairs
        
        # Compute alpha (weighting factors) - detach to stop gradients
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
        
        # Circle loss formulation: log(1 + sum_p(exp(logit_p)) * sum_n(exp(logit_n)))
        # Use logsumexp for numerical stability
        if len(pos_logits) > 0 and len(neg_logits) > 0:
            # Method 1: Global loss across all pairs
            pos_term = torch.logsumexp(pos_logits, dim=0)
            neg_term = torch.logsumexp(neg_logits, dim=0)
            loss = F.softplus(pos_term + neg_term)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class CircleLossV2(nn.Module):
    """
    Alternative Circle Loss implementation using per-sample computation.
    More similar to triplet loss formulation.
    """
    
    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 256,
        similarity_type: str = "cosine"
    ):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.similarity_type = similarity_type
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute Circle Loss per sample and average."""
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
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
            
            # Compute alpha weights
            alpha_p = torch.clamp(1 + self.margin - pos_sim.detach(), min=0.0)
            alpha_n = torch.clamp(neg_sim.detach() + self.margin, min=0.0)
            
            # Weighted logits
            logit_p = -self.gamma * alpha_p * (pos_sim - (1 - self.margin))
            logit_n = self.gamma * alpha_n * (neg_sim - self.margin)
            
            # Circle loss for this sample
            pos_term = torch.logsumexp(logit_p, dim=0)
            neg_term = torch.logsumexp(logit_n, dim=0)
            sample_loss = F.softplus(pos_term + neg_term)
            losses.append(sample_loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class EmbeddingRegularizationLoss(nn.Module):
    """
    Regularization losses for embeddings to encourage diversity
    and prevent collapse.
    """
    
    def __init__(
        self,
        l2_weight: float = 0.001,
        variance_weight: float = 0.01,
        target_norm: float = 1.0
    ):
        """
        Args:
            l2_weight: Weight for L2 regularization (keep embeddings bounded)
            variance_weight: Weight for variance loss (encourage diversity)
            target_norm: Target norm for embeddings
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.variance_weight = variance_weight
        self.target_norm = target_norm
    
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
            
            # Penalize low variance (collapse) - use negative log to encourage high variance
            # variance_loss = -torch.log(mean_var + 1e-8)
            # Alternative: penalize if variance is below a threshold
            target_var = 0.1  # Target variance per dimension
            variance_loss = F.relu(target_var - mean_var)
            losses["variance_loss"] = variance_loss * self.variance_weight
        else:
            losses["variance_loss"] = torch.tensor(0.0, device=device)
        
        # Total regularization loss
        losses["total_reg_loss"] = losses["l2_loss"] + losses["variance_loss"]
        
        return losses


class CombinedAutoencoderLoss(nn.Module):
    """
    Combined loss function for autoencoder training.
    
    Includes:
    - Reconstruction loss (cross-entropy)
    - Contrastive loss (embedding diversity)
    - Regularization losses
    """
    
    def __init__(
        self,
        pad_token_id: int,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        l2_weight: float = 0.001,
        variance_weight: float = 0.01,
        use_contrastive: bool = True
    ):
        super().__init__()
        
        self.reconstruction_loss = nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
        
        self.contrastive_loss = ContrastiveLoss(
            temperature=contrastive_temperature,
            use_tool_labels=True
        )
        
        self.regularization_loss = EmbeddingRegularizationLoss(
            l2_weight=l2_weight,
            variance_weight=variance_weight
        )
        
        self.contrastive_weight = contrastive_weight
        self.use_contrastive = use_contrastive
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
        tool_calls: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            embeddings: (batch_size, embedding_dim)
            tool_calls: List of tool call strings
            
        Returns:
            Dictionary with all loss terms
        """
        device = logits.device
        
        # Reconstruction loss
        loss_per_token = self.reconstruction_loss(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        mask_flat = attention_mask.reshape(-1).float()
        reconstruction_loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Contrastive loss
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss(embeddings, tool_calls)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        
        # Regularization losses
        reg_losses = self.regularization_loss(embeddings)
        
        # Total loss
        total_loss = (
            reconstruction_loss +
            self.contrastive_weight * contrastive_loss +
            reg_losses["total_reg_loss"]
        )
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "contrastive_loss": contrastive_loss,
            "l2_loss": reg_losses["l2_loss"],
            "variance_loss": reg_losses["variance_loss"]
        }


class ScheduledSampling:
    """
    Implements scheduled sampling for decoder training.
    
    Gradually reduces teacher forcing ratio during training
    to improve model robustness.
    """
    
    def __init__(
        self,
        start_ratio: float = 1.0,
        end_ratio: float = 0.5,
        warmup_steps: int = 1000,
        decay_steps: int = 10000
    ):
        """
        Args:
            start_ratio: Initial teacher forcing ratio (1.0 = full teacher forcing)
            end_ratio: Final teacher forcing ratio
            warmup_steps: Steps before starting decay
            decay_steps: Steps over which to decay
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def get_ratio(self, step: int) -> float:
        """Get teacher forcing ratio for current step."""
        if step < self.warmup_steps:
            return self.start_ratio
        
        progress = min(1.0, (step - self.warmup_steps) / self.decay_steps)
        ratio = self.start_ratio - progress * (self.start_ratio - self.end_ratio)
        return ratio
    
    def should_use_teacher_forcing(self, step: int) -> bool:
        """Randomly decide whether to use teacher forcing based on current ratio."""
        import random
        return random.random() < self.get_ratio(step)
