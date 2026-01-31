"""
Training script for NTILC cluster retrieval (NEW ARCHITECTURE Phase 2).

Trains query encoder to predict cluster IDs from natural language queries.
Uses frozen intent embedder and projection head from Phase 1.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, List, Any
import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import numpy as np

from models.intent_embedder import ToolIntentEmbedder
from models.projection_head import ProjectionHead
from models.cluster_retrieval import ClusterRetrieval
from models.query_encoder import QueryEncoder
from training.config import IntentEmbeddingConfig
from training.data_generator import NaturalLanguageToolCallGenerator, OutputFormat
from models.tool_schemas import TOOL_SCHEMAS


class QueryClusterDataset(Dataset):
    """Dataset for (query, cluster_id) pairs."""
    
    def __init__(
        self,
        queries: List[str],
        tool_calls: List[Dict[str, Any]],
        tool_labels: List[int],
        intent_embedder: ToolIntentEmbedder,
        projection_head: ProjectionHead,
        tokenizer: AutoTokenizer,
        max_length: int = 256
    ):
        """
        Args:
            queries: Natural language queries
            tool_calls: Corresponding tool call dicts
            intent_embedder: Frozen intent embedder
            projection_head: Frozen projection head
            tokenizer: Tokenizer for queries
            max_length: Max sequence length
        """
        self.queries = queries
        self.tool_calls = tool_calls
        self.labels = tool_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Precompute target cluster embeddings
        print("Precomputing target cluster embeddings...")
        intent_embedder.eval()
        projection_head.eval()
        
        self.target_embeddings = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(tool_calls), batch_size):
                batch_tool_calls = tool_calls[i:i+batch_size]
                # Get intent embeddings
                intent_embeds = intent_embedder(tool_calls=batch_tool_calls)
                # Project to 128-D
                projected = projection_head(intent_embeds)
                self.target_embeddings.append(projected.cpu())
        
        self.target_embeddings = torch.cat(self.target_embeddings, dim=0)
        print(f"Precomputed {len(self.target_embeddings)} target embeddings")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        target_embedding = self.target_embeddings[idx]
        
        # Tokenize query
        encoded = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_embedding": target_embedding,
            "label": self.labels[idx]
        }



def train_epoch(
    query_encoder: QueryEncoder,
    cluster_retrieval: ClusterRetrieval,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: IntentEmbeddingConfig,
    epoch: int,
    global_step: int,
    scheduler=None
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    query_encoder.train()
    
    total_losses = {
        "total_loss": 0.0,
        "mse_loss": 0.0,
        "cosine_loss": 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_embeddings = batch["target_embedding"].to(device)
        
        # Forward pass
        query_embeddings = query_encoder(input_ids, attention_mask)
        if target_embeddings.dtype != query_embeddings.dtype:
            target_embeddings = target_embeddings.to(query_embeddings.dtype)
        
        # Compute losses
        # MSE loss
        mse_loss = F.mse_loss(query_embeddings, target_embeddings)
        
        # Cosine similarity loss
        cosine_sim = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
        cosine_loss = (1 - cosine_sim).mean()
        
        # Total loss
        total_loss = mse_loss + cosine_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss = total_loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        # Accumulate losses
        total_losses["total_loss"] += total_loss.item()
        total_losses["mse_loss"] += mse_loss.item()
        total_losses["cosine_loss"] += cosine_loss.item()
        num_batches += 1
        
        current_step = global_step + batch_idx
        
        # Update progress bar
        if current_step % config.log_interval == 0:
            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "mse": f"{mse_loss.item():.4f}",
                "cosine": f"{cosine_loss.item():.4f}"
            })
            
            if config.use_wandb:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/cosine_loss": cosine_loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step": current_step
                })
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses, num_batches


def evaluate(
    query_encoder: QueryEncoder,
    cluster_retrieval: ClusterRetrieval,
    dataloader: DataLoader,
    device: torch.device,
    cluster_centroids: torch.Tensor
) -> Dict[str, float]:
    """Evaluate model."""
    query_encoder.eval()
    
    correct_clusters = 0
    total_samples = 0
    all_similarities = []
    
    # Loss accumulators
    total_loss = 0.0
    mse_loss_total = 0.0
    cosine_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_embeddings = batch["target_embedding"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            query_embeddings = query_encoder(input_ids, attention_mask)
            if target_embeddings.dtype != query_embeddings.dtype:
                target_embeddings = target_embeddings.to(query_embeddings.dtype)
            
            # Compute losses (same as training)
            mse_loss = F.mse_loss(query_embeddings, target_embeddings)
            cosine_sim = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
            cosine_loss = (1 - cosine_sim).mean()
            val_loss = mse_loss + cosine_loss
            
            total_loss += val_loss.item()
            mse_loss_total += mse_loss.item()
            cosine_loss_total += cosine_loss.item()
            num_batches += 1
            
            # Retrieve clusters
            results = cluster_retrieval(
                query_embeddings,
                cluster_embeddings=cluster_centroids.to(device, dtype=query_embeddings.dtype),
                top_k=1
            )
            pred_ids = results["cluster_ids"].squeeze(1)
            valid_mask = pred_ids >= 0
            correct_clusters += (pred_ids[valid_mask] == labels[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()
            
            # Compute similarity to target
            similarities = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
            all_similarities.extend(similarities.cpu().tolist())
            
            # total_samples updated above with valid_mask
    
    avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
    
    return {
        "total_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "mse_loss": mse_loss_total / num_batches if num_batches > 0 else 0.0,
        "cosine_loss": cosine_loss_total / num_batches if num_batches > 0 else 0.0,
        "avg_similarity": avg_similarity,
        "cluster_accuracy": correct_clusters / total_samples if total_samples > 0 else 0.0
    }


def main():
    """Main training function."""
    config = IntentEmbeddingConfig()
    
    # Load Phase 1 models
    checkpoint_path = os.path.join(config.output_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {checkpoint_path}")
    
    print(f"Loading Phase 1 models from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:1", weights_only=False)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda:1")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    model_torch_dtype = config.torch_dtype
    use_dtype = dtype_map.get(model_torch_dtype, torch.float32)
    
    # Initialize frozen models
    intent_embedder = ToolIntentEmbedder(
        model_name=config.encoder_model,
        embedding_dim=config.intent_embedding_dim,
        pooling_strategy=config.pooling_strategy,
        dropout=config.dropout,
        torch_dtype=model_torch_dtype,
        max_length=config.max_length
    ).to(device)
    intent_embedder.load_state_dict(checkpoint["intent_embedder_state_dict"])
    for param in intent_embedder.parameters():
        param.requires_grad = False
    intent_embedder.eval()
    
    projection_head = ProjectionHead(
        input_dim=config.intent_embedding_dim,
        output_dim=config.projection_dim,
        dropout=config.dropout
    ).to(device).to(use_dtype)
    projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
    for param in projection_head.parameters():
        param.requires_grad = False
    projection_head.eval()
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or "cluster_retrieval",
            config=config.__dict__
        )
    
    # Generate training data
    print("Generating training data...")
    nl_generator = NaturalLanguageToolCallGenerator(output_format=OutputFormat.PYTHON)
    
    train_data = []
    for _ in range(config.num_train_samples):
        sample = nl_generator.generate_pair()
        train_data.append(sample)
    
    # Create datasets
    queries = [item["query"] for item in train_data]
    tool_calls = [item["tool_call_dict"] for item in train_data]
    tool_names = list(TOOL_SCHEMAS.keys())
    tool_labels = [
        tool_names.index(tc.get("tool", "unknown")) if tc.get("tool", "unknown") in tool_names else -1
        for tc in tool_calls
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = QueryClusterDataset(
        queries,
        tool_calls,
        tool_labels,
        intent_embedder,
        projection_head,
        tokenizer,
        max_length=256
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize query encoder
    query_encoder = QueryEncoder(
        base_model=config.encoder_model,
        output_dim=config.projection_dim,
        dropout=config.dropout,
        torch_dtype=model_torch_dtype
    ).to(device).to(use_dtype)
    
    # Initialize cluster retrieval
    # Compute cluster centroids from training data
    print("Computing cluster centroids...")
    all_embeddings = train_dataset.target_embeddings
    num_clusters = len(TOOL_SCHEMAS)
    cluster_retrieval = ClusterRetrieval(
        embedding_dim=config.projection_dim,
        num_clusters=num_clusters,
        similarity_type="cosine"
    )
    
    # Simple clustering: one cluster per tool
    cluster_assignments = torch.tensor(tool_labels, dtype=torch.long)
    
    cluster_retrieval.update_cluster_centroids(all_embeddings, cluster_assignments)
    cluster_centroids = cluster_retrieval.get_cluster_centroids()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        query_encoder.parameters(),
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
            query_encoder,
            cluster_retrieval,
            train_loader,
            optimizer,
            device,
            config,
            epoch,
            global_step,
            scheduler
        )
        global_step += num_batches
        
        # Evaluate
        val_metrics = evaluate(
            query_encoder,
            cluster_retrieval,
            train_loader,  # Use train for now
            device,
            cluster_centroids
        )
        
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics.get('total_loss', 0.0):.4f}")
        print(f"Val Metrics: {val_metrics}")
        
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "val/total_loss": val_metrics.get("total_loss", 0.0),
                "val/mse_loss": val_metrics.get("mse_loss", 0.0),
                "val/cosine_loss": val_metrics.get("cosine_loss", 0.0),
                "val/avg_similarity": val_metrics.get("avg_similarity", 0.0),
                "val/cluster_accuracy": val_metrics.get("cluster_accuracy", 0.0)
            })
        
        # Save checkpoint
        val_loss = val_metrics.get("total_loss", 1.0 - val_metrics.get("avg_similarity", 0.0))  # Use total_loss if available
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "query_encoder_state_dict": query_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cluster_centroids": cluster_centroids,
                "config": config.__dict__,
                "val_metrics": val_metrics
            }
            output_dir = os.path.join(config.output_dir, "cluster_retrieval")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining complete!")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
