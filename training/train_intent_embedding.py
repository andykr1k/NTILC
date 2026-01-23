"""
Training script for NTILC intent embedding (NEW ARCHITECTURE).

Trains intent embedder to map tool intents to 1024-D embeddings,
with projection head to 128-D for similarity computation.
Uses Circle Loss for metric learning and cluster formation.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, List, Any
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from collections import defaultdict

from models.intent_embedder import ToolIntentEmbedder
from models.projection_head import ProjectionHead
from training.config import IntentEmbeddingConfig
from training.data_generator import ToolInvocationGenerator, NaturalLanguageToolCallGenerator, OutputFormat
from training.losses import CircleLoss, ContrastiveLoss, EmbeddingRegularizationLoss
from evaluation.metrics import compute_cluster_metrics
from ablation.tool_schemas import TOOL_SCHEMAS


class IntentDataset(Dataset):
    """Dataset for tool intent training."""
    
    def __init__(
        self,
        tool_calls: List[Dict[str, Any]],
        queries: List[str] = None
    ):
        """
        Args:
            tool_calls: List of tool call dicts with 'tool' and 'arguments'
            queries: Optional list of natural language queries
        """
        self.tool_calls = tool_calls
        self.queries = queries or [None] * len(tool_calls)
        
        # Extract tool labels
        self.tool_names = list(TOOL_SCHEMAS.keys())
        self.tool_to_idx = {tool: idx for idx, tool in enumerate(self.tool_names)}
        self.labels = [self.tool_to_idx.get(tc.get("tool", "unknown"), 0) for tc in tool_calls]
    
    def __len__(self):
        return len(self.tool_calls)
    
    def __getitem__(self, idx):
        return {
            "tool_call": self.tool_calls[idx],
            "query": self.queries[idx],
            "label": self.labels[idx]
        }


def train_epoch(
    intent_embedder: ToolIntentEmbedder,
    projection_head: ProjectionHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    circle_loss_fn: CircleLoss,
    contrastive_loss_fn: ContrastiveLoss,
    reg_loss_fn: EmbeddingRegularizationLoss,
    device: torch.device,
    config: IntentEmbeddingConfig,
    epoch: int,
    global_step: int,
    scheduler=None
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    intent_embedder.train()
    projection_head.train()
    
    total_losses = {
        "total_loss": 0.0,
        "circle_loss": 0.0,
        "contrastive_loss": 0.0,
        "l2_loss": 0.0,
        "variance_loss": 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        tool_calls = [item["tool_call"] for item in batch]
        queries = [item["query"] for item in batch if item["query"] is not None]
        labels = torch.tensor([item["label"] for item in batch], device=device)
        
        # Forward pass: intent embedding
        intent_embeddings = intent_embedder(
            tool_calls=tool_calls,
            queries=queries if queries else None
        )  # (batch_size, 1024)
        
        # Project to 128-D
        projected_embeddings = projection_head(intent_embeddings)  # (batch_size, 128)
        
        # Compute losses
        # Circle Loss on projected embeddings
        circle_loss = circle_loss_fn(projected_embeddings, labels)
        
        # Contrastive loss (optional)
        contrastive_loss = torch.tensor(0.0, device=device)
        if config.use_contrastive_loss:
            contrastive_loss = contrastive_loss_fn(projected_embeddings, labels)
        
        # Regularization losses
        reg_losses = reg_loss_fn(intent_embeddings)
        
        # Total loss
        total_loss = (
            config.circle_loss_weight * circle_loss +
            config.contrastive_loss_weight * contrastive_loss +
            reg_losses["total_reg_loss"]
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss = total_loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(intent_embedder.parameters()) + list(projection_head.parameters()),
                config.gradient_clip
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        # Accumulate losses
        total_losses["total_loss"] += total_loss.item()
        total_losses["circle_loss"] += circle_loss.item()
        total_losses["contrastive_loss"] += contrastive_loss.item()
        total_losses["l2_loss"] += reg_losses["l2_loss"].item()
        total_losses["variance_loss"] += reg_losses["variance_loss"].item()
        num_batches += 1
        
        current_step = global_step + batch_idx
        
        # Update progress bar
        if current_step % config.log_interval == 0:
            avg_loss = total_loss.item()
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "circle": f"{circle_loss.item():.4f}",
                "contrastive": f"{contrastive_loss.item():.4f}"
            })
            
            if config.use_wandb:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/circle_loss": circle_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/l2_loss": reg_losses["l2_loss"].item(),
                    "train/variance_loss": reg_losses["variance_loss"].item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step": current_step
                })
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses, num_batches


def evaluate(
    intent_embedder: ToolIntentEmbedder,
    projection_head: ProjectionHead,
    dataloader: DataLoader,
    device: torch.device,
    config: IntentEmbeddingConfig,
    circle_loss_fn: CircleLoss,
    contrastive_loss_fn: ContrastiveLoss,
    reg_loss_fn: EmbeddingRegularizationLoss
) -> Dict[str, float]:
    """Evaluate model."""
    intent_embedder.eval()
    projection_head.eval()
    
    all_embeddings = []
    all_labels = []
    all_tool_calls = []
    
    # Loss accumulators
    total_loss = 0.0
    circle_loss_total = 0.0
    contrastive_loss_total = 0.0
    l2_loss_total = 0.0
    variance_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tool_calls = [item["tool_call"] for item in batch]
            queries = [item["query"] for item in batch if item["query"] is not None]
            labels = torch.tensor([item["label"] for item in batch], device=device)
            
            # Forward pass
            intent_embeddings = intent_embedder(
                tool_calls=tool_calls,
                queries=queries if queries else None
            )
            projected_embeddings = projection_head(intent_embeddings)
            
            # Compute losses (same as training)
            circle_loss = circle_loss_fn(projected_embeddings, labels)
            if config.use_contrastive_loss:
                contrastive_loss = contrastive_loss_fn(projected_embeddings, labels)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            reg_losses = reg_loss_fn(intent_embeddings)
            
            val_loss = (
                config.circle_loss_weight * circle_loss +
                config.contrastive_loss_weight * contrastive_loss +
                reg_losses["total_reg_loss"]
            )
            
            # Accumulate losses
            total_loss += val_loss.item()
            circle_loss_total += circle_loss.item()
            contrastive_loss_total += contrastive_loss.item()
            l2_loss_total += reg_losses["l2_loss"].item()
            variance_loss_total += reg_losses["variance_loss"].item()
            num_batches += 1
            
            all_embeddings.append(projected_embeddings.cpu())
            all_labels.append(labels.cpu())
            all_tool_calls.extend(tool_calls)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_cluster_metrics(all_embeddings, all_labels, all_tool_calls)
    
    # Add loss metrics
    metrics["total_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
    metrics["circle_loss"] = circle_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["contrastive_loss"] = contrastive_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["l2_loss"] = l2_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["variance_loss"] = variance_loss_total / num_batches if num_batches > 0 else 0.0
    
    return metrics


def main():
    """Main training function."""
    config = IntentEmbeddingConfig()
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or "intent_embedding",
            config=config.__dict__
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Generate training data WITH natural language queries
    print("Generating training data with natural language queries...")
    nl_generator = NaturalLanguageToolCallGenerator(output_format=OutputFormat.JSON)
    
    train_data = []
    for _ in range(config.num_train_samples):
        # Generate (query, tool_call) pair
        pair = nl_generator.generate_pair()
        
        # Parse tool call
        try:
            tool_call_dict = json.loads(pair["tool_call"])
            train_data.append({
                "tool_call": tool_call_dict,
                "query": pair["query"]
            })
        except json.JSONDecodeError:
            continue
    
    # Create datasets WITH queries
    train_tool_calls = [item["tool_call"] for item in train_data]
    train_queries = [item["query"] for item in train_data]
    train_dataset = IntentDataset(train_tool_calls, queries=train_queries)
    
    val_tool_calls = [item["tool_call"] for item in train_data[:config.num_val_samples]]
    val_queries = [item["query"] for item in train_data[:config.num_val_samples]]
    val_dataset = IntentDataset(val_tool_calls, queries=val_queries)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize models
    print("Initializing models...")
    intent_embedder = ToolIntentEmbedder(
        model_name=config.encoder_model,
        embedding_dim=config.intent_embedding_dim,
        pooling_strategy=config.pooling_strategy,
        dropout=config.dropout,
        freeze_base=config.freeze_encoder,
        freeze_layers=config.freeze_encoder_layers,
        torch_dtype="bfloat16" if device.type == "cuda" else "float32",
        max_length=config.max_length
    ).to(device)
    
    projection_head = ProjectionHead(
        input_dim=config.intent_embedding_dim,
        output_dim=config.projection_dim,
        dropout=config.dropout
    ).to(device)
    
    # Loss functions
    circle_loss_fn = CircleLoss(
        margin=config.circle_loss_margin,
        gamma=config.circle_loss_gamma,
        similarity_type="cosine"
    )
    contrastive_loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)
    reg_loss_fn = EmbeddingRegularizationLoss(
        l2_weight=config.embedding_l2_weight,
        variance_weight=config.embedding_variance_weight
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(intent_embedder.parameters()) + list(projection_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # Train
        train_losses, num_batches = train_epoch(
            intent_embedder,
            projection_head,
            train_loader,
            optimizer,
            circle_loss_fn,
            contrastive_loss_fn,
            reg_loss_fn,
            device,
            config,
            epoch,
            global_step,
            scheduler
        )
        global_step += num_batches
        
        # Evaluate
        val_metrics = evaluate(
            intent_embedder, 
            projection_head, 
            val_loader, 
            device, 
            config,
            circle_loss_fn,
            contrastive_loss_fn,
            reg_loss_fn
        )
        
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics.get('total_loss', 0.0):.4f}")
        print(f"Val Metrics: {val_metrics}")
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "val/total_loss": val_metrics.get("total_loss", 0.0),
                "val/circle_loss": val_metrics.get("circle_loss", 0.0),
                "val/contrastive_loss": val_metrics.get("contrastive_loss", 0.0),
                "val/l2_loss": val_metrics.get("l2_loss", 0.0),
                "val/variance_loss": val_metrics.get("variance_loss", 0.0),
                "val/cluster_accuracy": val_metrics.get("cluster_accuracy", 0.0),
                "val/intra_cluster_similarity": val_metrics.get("intra_cluster_similarity", 0.0),
                "val/inter_cluster_similarity": val_metrics.get("inter_cluster_similarity", 0.0),
                "val/cluster_separation": val_metrics.get("cluster_separation", 0.0),
                "val/silhouette_score": val_metrics.get("silhouette_score", 0.0)
            })
        
        # Save checkpoint
        val_loss = val_metrics.get("total_loss", train_losses["total_loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "intent_embedder_state_dict": intent_embedder.state_dict(),
                "projection_head_state_dict": projection_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "val_metrics": val_metrics
            }
            torch.save(checkpoint, os.path.join(config.output_dir, "best_model.pt"))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining complete!")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
