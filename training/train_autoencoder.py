"""
Training script for NTILC autoencoder.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import wandb
from typing import Tuple, Dict

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.autoencoder import ToolInvocationAutoencoder
from training.config import AutoencoderConfig
from training.data_generator import ToolInvocationGenerator, DataGeneratorConfig
from evaluation.metrics import compute_metrics


class ToolInvocationDataset(Dataset):
    """Dataset for tool invocation strings."""
    
    def __init__(self, tool_calls: list[str], tokenizer: GPT2Tokenizer, max_length: int = 128):
        """
        Args:
            tool_calls: List of tool invocation strings
            tokenizer: Tokenizer for encoding targets
            max_length: Maximum sequence length
        """
        self.tool_calls = tool_calls
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tool_calls)
    
    def __getitem__(self, idx):
        tool_call = self.tool_calls[idx]
        
        # Tokenize target
        encoded = self.tokenizer(
            tool_call,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "tool_call": tool_call,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def train_epoch(
    model: ToolInvocationAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: AutoencoderConfig,
    epoch: int,
    global_step: int,
    scheduler=None
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_grad_norm = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        tool_calls = batch["tool_call"]
        target_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        outputs = model(
            tool_calls=tool_calls,
            target_ids=target_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs["logits"]  # (batch_size, seq_len-1, vocab_size)
        
        # Shift targets for next-token prediction (what we're predicting)
        targets = target_ids[:, 1:]  # (batch_size, seq_len-1)
        target_mask = attention_mask[:, 1:]  # (batch_size, seq_len-1)
        
        # Compute loss (only on non-padding tokens)
        # Reshape for cross entropy: (batch * seq, vocab_size) vs (batch * seq,)
        loss_per_token = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )  # (batch * seq,)
        
        # Mask out padding tokens and compute mean
        mask_flat = target_mask.reshape(-1).float()
        loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping and tracking
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        total_grad_norm += grad_norm.item()
        
        optimizer.step()
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        current_step = global_step + batch_idx
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Logging
        if batch_idx % config.log_interval == 0:
            if config.use_wandb:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": current_step,
                    "train/epoch": epoch
                }
                wandb.log(log_dict, step=current_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    
    new_global_step = global_step + len(dataloader)
    return {"loss": avg_loss, "grad_norm": avg_grad_norm}, new_global_step


def evaluate(
    model: ToolInvocationAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    config: AutoencoderConfig,
    split: str = "val"
) -> dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_tool_calls = []
    all_reconstructed = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            tool_calls = batch["tool_call"]
            
            # Reconstruct
            reconstructed = model.reconstruct(tool_calls)
            
            # Get embeddings
            embeddings = model.encode(tool_calls)
            
            all_tool_calls.extend(tool_calls)
            all_reconstructed.extend(reconstructed)
            all_embeddings.append(embeddings.cpu())
    
    # Compute metrics
    metrics = compute_metrics(all_tool_calls, all_reconstructed, torch.cat(all_embeddings, dim=0))
    
    # Add per-tool metrics if available
    try:
        from evaluation.metrics import per_tool_metrics
        per_tool = per_tool_metrics(all_tool_calls, all_reconstructed)
        for tool_name, tool_metrics in per_tool.items():
            for metric_name, metric_value in tool_metrics.items():
                if metric_name != "count":  # Skip count, log as separate metric
                    metrics[f"{tool_name}/{metric_name}"] = metric_value
                else:
                    metrics[f"{tool_name}/count"] = metric_value
    except Exception as e:
        print(f"Warning: Could not compute per-tool metrics: {e}")
    
    return metrics


def main():
    """Main training function."""
    # Load configuration
    config = AutoencoderConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Log device information to wandb (will be added after init)
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": torch.cuda.get_device_capability(0),
        })
    
    # Initialize wandb if enabled
    if config.use_wandb:
        # Generate run name if not provided
        run_name = config.wandb_run_name
        if run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"autoencoder_{timestamp}"
        
        # Organize hyperparameters into categories for better tracking
        # NOTE: When adding new config parameters, make sure to add them here!
        # This ensures all hyperparameters are tracked in wandb
        hyperparams = {
            # Model architecture
            "model/embedding_dim": config.embedding_dim,
            "model/encoder_model": config.encoder_model,
            "model/decoder_model": config.decoder_model,
            "model/pooling_strategy": config.pooling_strategy,
            "model/max_length": config.max_length,
            "model/dropout": config.dropout,
            "model/freeze_encoder": config.freeze_encoder,
            "model/freeze_decoder": config.freeze_decoder,
            
            # Training hyperparameters
            "training/batch_size": config.batch_size,
            "training/learning_rate": config.learning_rate,
            "training/weight_decay": config.weight_decay,
            "training/num_epochs": config.num_epochs,
            "training/warmup_steps": config.warmup_steps,
            "training/gradient_clip": config.gradient_clip,
            "training/use_lr_scheduler": config.use_lr_scheduler,
            "training/optimizer": "AdamW",
            "training/loss_function": "CrossEntropyLoss",
            
            # Data configuration
            "data/num_train_samples": config.num_train_samples,
            "data/num_val_samples": config.num_val_samples,
            "data/num_test_samples": config.num_test_samples,
            
            # Early stopping
            "early_stopping/patience": config.early_stopping_patience,
            "early_stopping/min_delta": config.early_stopping_min_delta,
            
            # Logging configuration
            "logging/log_interval": config.log_interval,
            "logging/eval_interval": config.eval_interval,
            "logging/save_interval": config.save_interval,
            "logging/use_wandb": config.use_wandb,
            "logging/wandb_project": config.wandb_project,
            "logging/wandb_entity": config.wandb_entity,
            
            # Paths
            "paths/output_dir": config.output_dir,
            "paths/log_dir": config.log_dir,
        }
        
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=hyperparams,
            name=run_name,
            tags=["autoencoder", "training"]
        )
        
        # Add device information
        wandb.config.update(device_info)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Generate or load data
    print("Generating training data...")
    data_config = DataGeneratorConfig()
    generator = ToolInvocationGenerator(data_config)
    
    train_data_path = os.path.join(config.log_dir, "train_data.txt")
    val_data_path = os.path.join(config.log_dir, "val_data.txt")
    test_data_path = os.path.join(config.log_dir, "test_data.txt")
    
    if not os.path.exists(train_data_path):
        print(f"Generating {config.num_train_samples} training samples...")
        train_tool_calls = generator.generate_dataset(config.num_train_samples)
        generator.save_dataset(train_tool_calls, train_data_path)
    else:
        train_tool_calls = generator.load_dataset(train_data_path)
    
    if not os.path.exists(val_data_path):
        print(f"Generating {config.num_val_samples} validation samples...")
        val_tool_calls = generator.generate_dataset(config.num_val_samples)
        generator.save_dataset(val_tool_calls, val_data_path)
    else:
        val_tool_calls = generator.load_dataset(val_data_path)
    
    if not os.path.exists(test_data_path):
        print(f"Generating {config.num_test_samples} test samples...")
        test_tool_calls = generator.generate_dataset(config.num_test_samples)
        generator.save_dataset(test_tool_calls, test_data_path)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = ToolInvocationDataset(train_tool_calls, tokenizer, config.max_length)
    val_dataset = ToolInvocationDataset(val_tool_calls, tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Update wandb config with computed hyperparameters
    if config.use_wandb:
        wandb.config.update({
            "data/actual_train_size": len(train_tool_calls),
            "data/actual_val_size": len(val_tool_calls),
            "data/vocab_size": len(tokenizer),
            "training/steps_per_epoch": len(train_loader),
            "training/total_training_steps": len(train_loader) * config.num_epochs,
            "training/num_workers": 4,
        })
        
        # Add data generator config
        wandb.config.update({
            "data_generator/format_style": data_config.format_style,
            "data_generator/min_query_length": data_config.min_query_length,
            "data_generator/max_query_length": data_config.max_query_length,
            "data_generator/min_max_results": data_config.min_max_results,
            "data_generator/max_max_results": data_config.max_max_results,
            "data_generator/num_tools": len(data_config.tools),
        })
    
    # Initialize model
    print("Initializing model...")
    model = ToolInvocationAutoencoder(
        embedding_dim=config.embedding_dim,
        encoder_model=config.encoder_model,
        decoder_model=config.decoder_model,
        pooling_strategy=config.pooling_strategy,
        max_length=config.max_length,
        dropout=config.dropout,
        freeze_encoder=config.freeze_encoder,
        freeze_decoder=config.freeze_decoder
    )
    model = model.to(device)
    
    # Watch model for gradients and parameters (after model is initialized)
    if config.use_wandb:
        wandb.watch(model, log="all", log_freq=config.log_interval)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    scheduler = None
    if config.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * config.num_epochs - config.warmup_steps,
            eta_min=config.learning_rate * 0.01
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps]
        )
    
    # Log optimizer-specific hyperparameters
    if config.use_wandb:
        optimizer_config = {
            "optimizer/betas": optimizer.param_groups[0].get('betas', (0.9, 0.999)),
            "optimizer/eps": optimizer.param_groups[0].get('eps', 1e-8),
            "optimizer/amsgrad": optimizer.param_groups[0].get('amsgrad', False),
        }
        if scheduler is not None:
            optimizer_config.update({
                "scheduler/type": "SequentialLR",
                "scheduler/warmup_type": "LinearLR",
                "scheduler/cosine_type": "CosineAnnealingLR",
                "scheduler/warmup_steps": config.warmup_steps,
                "scheduler/warmup_start_factor": 0.1,
                "scheduler/cosine_eta_min": config.learning_rate * 0.01,
            })
        wandb.config.update(optimizer_config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    # Log model parameter counts
    if config.use_wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/non_trainable_parameters": total_params - trainable_params,
        })
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch, global_step, scheduler
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Log epoch-level train metrics
        if config.use_wandb:
            wandb.log({
                "train/epoch_loss": train_metrics['loss'],
                "train/epoch_grad_norm": train_metrics.get('grad_norm', 0.0),
                "train/epoch": epoch + 1
            }, step=global_step)
        
        # Evaluate
        eval_should_run = (
            (epoch + 1) % max(1, config.eval_interval // len(train_loader)) == 0 
            or epoch == 0
        )
        if eval_should_run:
            val_metrics = evaluate(model, val_loader, device, config, split="val")
            print(f"Validation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Log to wandb with proper step
            if config.use_wandb:
                log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
                log_dict["val/epoch"] = epoch + 1
                wandb.log(log_dict, step=global_step)
            
            # Early stopping - use reconstruction error (1 - accuracy) as loss
            # Prefer exact_match_accuracy, fallback to tool_accuracy
            val_accuracy = val_metrics.get("exact_match_accuracy", val_metrics.get("tool_accuracy", 0.0))
            val_loss = 1.0 - val_accuracy  # Convert accuracy to "loss" for early stopping
            
            if val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(config.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                    "val_metrics": val_metrics
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
                
                # Save as wandb artifact
                if config.use_wandb:
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    if config.use_wandb:
                        wandb.log({"train/early_stopped": True, "train/epoch": epoch + 1}, step=global_step)
                    break
        
        # Periodic checkpoint
        checkpoint_should_run = (
            (epoch + 1) % max(1, config.save_interval // len(train_loader)) == 0
        )
        if checkpoint_should_run:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__
            }, checkpoint_path)
            
            # Save as wandb artifact
            if config.use_wandb:
                artifact = wandb.Artifact(f"checkpoint_epoch_{epoch + 1}", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
    
    print("\nTraining complete!")
    
    # Final evaluation on test set
    if os.path.exists(test_data_path):
        print("Evaluating on test set...")
        test_tool_calls = generator.load_dataset(test_data_path)
        test_dataset = ToolInvocationDataset(test_tool_calls, tokenizer, config.max_length)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        test_metrics = evaluate(model, test_loader, device, config, split="test")
        print("Test Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test metrics
        test_metrics_path = os.path.join(config.output_dir, "test_metrics.json")
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Log test metrics to wandb
        if config.use_wandb:
            log_dict = {f"test/{k}": v for k, v in test_metrics.items()}
            wandb.log(log_dict, step=global_step)
            
            # Save test metrics as artifact
            artifact = wandb.Artifact("test_metrics", type="metrics")
            artifact.add_file(test_metrics_path)
            wandb.log_artifact(artifact)
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
