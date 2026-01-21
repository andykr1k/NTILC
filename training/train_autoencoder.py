"""
Training script for NTILC autoencoder.
Includes contrastive loss, embedding regularization, and improved training dynamics.
"""
import sys
from pathlib import Path

# Add project root to path BEFORE any other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, Union, List
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import json
import os
import shutil

from models.autoencoder import ToolInvocationAutoencoder
from training.config import AutoencoderConfig
from training.data_generator import ToolInvocationGenerator, OutputFormat
from training.losses import CombinedAutoencoderLoss, ScheduledSampling
from evaluation.metrics import compute_metrics


class ToolInvocationDataset(Dataset):
    """Dataset for tool invocation strings."""

    def __init__(self, tool_calls: List[str], tokenizer: AutoTokenizer, max_length: int = 256):
        self.tool_calls = tool_calls
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tool_calls)

    def __getitem__(self, idx):
        tool_call = self.tool_calls[idx]

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
    loss_fn: CombinedAutoencoderLoss,
    device: torch.device,
    config: AutoencoderConfig,
    epoch: int,
    global_step: int,
    scheduler=None,
    scheduled_sampling: ScheduledSampling = None
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch with improved loss tracking."""
    model.train()
    
    total_losses = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "contrastive_loss": 0.0,
        "l2_loss": 0.0,
        "variance_loss": 0.0
    }
    num_batches = 0
    total_grad_norm = 0.0
    nan_count = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

    for batch_idx, batch in enumerate(progress_bar):
        tool_calls = batch["tool_call"]
        target_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Check for NaN in inputs
        if torch.isnan(target_ids.float()).any():
            print(f"Warning: NaN in input at batch {batch_idx}")
            continue

        # Forward pass
        outputs = model(
            tool_calls=tool_calls,
            target_ids=target_ids,
            attention_mask=attention_mask
        )
        logits = outputs["logits"]
        embeddings = outputs["embeddings"]
        
        # Check for NaN in outputs
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/Inf in logits at batch {batch_idx}")
            nan_count += 1
            if nan_count > 10:
                raise ValueError("Too many NaN occurrences")
            continue

        # Compute combined loss
        losses = loss_fn(
            logits=logits,
            targets=target_ids,
            attention_mask=attention_mask,
            embeddings=embeddings,
            tool_calls=tool_calls
        )
        
        loss = losses["total_loss"]
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at batch {batch_idx}")
            nan_count += 1
            if nan_count > 10:
                raise ValueError("Too many NaN losses")
            continue

        # Backward pass
        optimizer.zero_grad()
        
        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        # Only step optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Check gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: NaN/Inf gradient in {name}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                nan_count += 1
                optimizer.zero_grad()
                continue
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_grad_norm += grad_norm.item()

        # Accumulate losses
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        num_batches += 1
        current_step = global_step + batch_idx

        # Update progress bar
        progress_bar.set_postfix({
            "loss": losses["total_loss"].item(),
            "recon": losses["reconstruction_loss"].item(),
            "contrast": losses["contrastive_loss"].item() if config.use_contrastive_loss else 0.0,
        })

        # Logging
        if batch_idx % config.log_interval == 0 and config.use_wandb:
            log_dict = {
                "train/total_loss": losses["total_loss"].item(),
                "train/reconstruction_loss": losses["reconstruction_loss"].item(),
                "train/contrastive_loss": losses["contrastive_loss"].item(),
                "train/l2_loss": losses["l2_loss"].item(),
                "train/variance_loss": losses["variance_loss"].item(),
                "train/grad_norm": grad_norm.item() if 'grad_norm' in dir() else 0.0,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/step": current_step,
                "train/epoch": epoch,
                "train/embedding_norm_mean": embeddings.norm(dim=1).mean().item(),
                "train/embedding_norm_std": embeddings.norm(dim=1).std().item(),
            }
            if scheduled_sampling:
                log_dict["train/teacher_forcing_ratio"] = scheduled_sampling.get_ratio(current_step)
            wandb.log(log_dict, step=current_step)

    # Compute averages
    avg_losses = {k: v / num_batches if num_batches > 0 else 0.0 for k, v in total_losses.items()}
    avg_losses["grad_norm"] = total_grad_norm / num_batches if num_batches > 0 else 0.0
    avg_losses["nan_count"] = nan_count

    new_global_step = global_step + len(dataloader)
    return avg_losses, new_global_step


def evaluate(
    model: ToolInvocationAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    config: AutoencoderConfig,
    split: str = "val",
    return_data: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict]]:
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
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    metrics = compute_metrics(all_tool_calls, all_reconstructed, all_embeddings_tensor)

    # Per-tool metrics
    try:
        from evaluation.metrics import per_tool_metrics
        per_tool = per_tool_metrics(all_tool_calls, all_reconstructed)
        for tool_name, tool_metrics in per_tool.items():
            for metric_name, metric_value in tool_metrics.items():
                metrics[f"{tool_name}/{metric_name}"] = metric_value
    except Exception as e:
        print(f"Warning: Could not compute per-tool metrics: {e}")

    if return_data:
        return metrics, {
            "tool_calls": all_tool_calls,
            "reconstructed": all_reconstructed,
            "embeddings": all_embeddings_tensor
        }
    return metrics


def main():
    """Main training function."""
    config = AutoencoderConfig()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Device info
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
        })

    # Initialize wandb
    if config.use_wandb:
        import datetime
        run_name = config.wandb_run_name or f"autoencoder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "embedding_dim": config.embedding_dim,
                "encoder_model": config.encoder_model,
                "decoder_model": config.decoder_model,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "use_contrastive_loss": config.use_contrastive_loss,
                "contrastive_weight": config.contrastive_loss_weight,
                "label_smoothing": config.label_smoothing,
                "freeze_encoder": config.freeze_encoder,
                "freeze_decoder": config.freeze_decoder,
                "freeze_encoder_layers": config.freeze_encoder_layers,
                "freeze_decoder_layers": config.freeze_decoder_layers,
                "output_format": config.output_format,
            },
            name=run_name,
            tags=["autoencoder", "training", "v2"]
        )
        wandb.config.update(device_info)

    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    eval_results_dir = os.path.join(project_root, "output", "autoencoder_train_eval")
    os.makedirs(eval_results_dir, exist_ok=True)

    # Data generation
    print("Setting up data generation...")
    output_format = OutputFormat.JSON if config.output_format == "json" else OutputFormat.PYTHON
    generator = ToolInvocationGenerator(output_format=output_format)

    train_data_path = os.path.join(config.data_dir, "train_data.txt")
    val_data_path = os.path.join(config.data_dir, "val_data.txt")
    test_data_path = os.path.join(config.data_dir, "test_data.txt")

    # Regenerate data if needed (format change)
    if config.regenerate_data or not os.path.exists(train_data_path):
        print(f"Generating {config.num_train_samples} training samples in {config.output_format} format...")
        train_tool_calls = generator.generate_dataset(config.num_train_samples)
        generator.save_dataset(train_tool_calls, train_data_path)
        
        print(f"Generating {config.num_val_samples} validation samples...")
        val_tool_calls = generator.generate_dataset(config.num_val_samples)
        generator.save_dataset(val_tool_calls, val_data_path)
        
        print(f"Generating {config.num_test_samples} test samples...")
        test_tool_calls = generator.generate_dataset(config.num_test_samples)
        generator.save_dataset(test_tool_calls, test_data_path)
    else:
        train_tool_calls = generator.load_dataset(train_data_path)
        val_tool_calls = generator.load_dataset(val_data_path)

    print(f"Training samples: {len(train_tool_calls)}")
    print(f"Sample tool call: {train_tool_calls[0][:100]}...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.decoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = ToolInvocationDataset(train_tool_calls, tokenizer, config.max_length)
    val_dataset = ToolInvocationDataset(val_tool_calls, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

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
        freeze_decoder=config.freeze_decoder,
        freeze_encoder_layers=config.freeze_encoder_layers,
        freeze_decoder_layers=config.freeze_decoder_layers,
        torch_dtype=config.torch_dtype,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )
    model = model.to(device)

    # Print parameter counts
    param_counts = model.get_trainable_params()
    total_params = model.get_total_params()
    print(f"Total parameters: {total_params['total']:,}")
    print(f"Trainable parameters: {param_counts['total']:,}")
    print(f"  Encoder: {param_counts['encoder']:,}")
    print(f"  Decoder: {param_counts['decoder']:,}")

    if config.use_wandb:
        wandb.config.update({
            "total_params": total_params['total'],
            "trainable_params": param_counts['total'],
        })
        wandb.watch(model, log="gradients", log_freq=config.log_interval)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio) if config.warmup_ratio else config.warmup_steps
    
    scheduler = None
    if config.use_lr_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.learning_rate * 0.01
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    # Initialize loss function
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loss_fn = CombinedAutoencoderLoss(
        pad_token_id=pad_token_id,
        label_smoothing=config.label_smoothing,
        contrastive_weight=config.contrastive_loss_weight,
        contrastive_temperature=config.contrastive_temperature,
        l2_weight=config.embedding_l2_weight,
        variance_weight=config.embedding_variance_weight,
        use_contrastive=config.use_contrastive_loss
    )

    # Scheduled sampling
    scheduled_sampling = None
    if config.use_scheduled_sampling:
        scheduled_sampling = ScheduledSampling(
            start_ratio=config.scheduled_sampling_start,
            end_ratio=config.scheduled_sampling_end,
            warmup_steps=config.scheduled_sampling_warmup,
            decay_steps=total_steps // 2
        )

    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    global_step = 0

    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, 
            epoch, global_step, scheduler, scheduled_sampling
        )
        print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Recon: {train_metrics['reconstruction_loss']:.4f} | "
              f"Contrast: {train_metrics['contrastive_loss']:.4f}")

        # Log epoch metrics
        if config.use_wandb:
            wandb.log({
                "train/epoch_total_loss": train_metrics['total_loss'],
                "train/epoch_reconstruction_loss": train_metrics['reconstruction_loss'],
                "train/epoch_contrastive_loss": train_metrics['contrastive_loss'],
                "train/epoch": epoch + 1
            }, step=global_step)

        # Evaluate
        if (epoch + 1) % max(1, config.eval_interval // len(train_loader)) == 0 or epoch == 0:
            val_metrics = evaluate(model, val_loader, device, config, split="val")
            
            print(f"Validation Metrics:")
            print(f"  Exact Match: {val_metrics.get('exact_match_accuracy', 0):.4f}")
            print(f"  Tool Accuracy: {val_metrics.get('tool_accuracy', 0):.4f}")
            print(f"  Embedding Norm: {val_metrics.get('embedding_mean_norm', 0):.4f}")
            print(f"  Embedding Variance: {val_metrics.get('embedding_mean_variance', 0):.4f}")

            # Save metrics
            val_metrics_path = os.path.join(eval_results_dir, f"val_metrics_epoch_{epoch + 1}.json")
            with open(val_metrics_path, 'w') as f:
                json.dump({"epoch": epoch + 1, "metrics": val_metrics}, f, indent=2)

            if config.use_wandb:
                log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
                log_dict["val/epoch"] = epoch + 1
                wandb.log(log_dict, step=global_step)

            # Early stopping based on exact match accuracy
            val_accuracy = val_metrics.get("exact_match_accuracy", 0.0)

            if val_accuracy > best_val_accuracy + config.early_stopping_min_delta:
                best_val_accuracy = val_accuracy
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
                print(f"Saved best model (accuracy: {val_accuracy:.4f})")

                if config.use_wandb:
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        # Periodic checkpoint
        if (epoch + 1) % max(1, config.save_interval // len(train_loader)) == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__
            }, checkpoint_path)

    print("\nTraining complete!")

    # Final test evaluation
    if os.path.exists(test_data_path):
        print("Evaluating on test set...")
        test_tool_calls = generator.load_dataset(test_data_path)
        test_dataset = ToolInvocationDataset(test_tool_calls, tokenizer, config.max_length)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        test_metrics, test_data = evaluate(
            model, test_loader, device, config, split="test", return_data=True)
        
        print("Test Metrics:")
        print(f"  Exact Match: {test_metrics.get('exact_match_accuracy', 0):.4f}")
        print(f"  Tool Accuracy: {test_metrics.get('tool_accuracy', 0):.4f}")

        # Save results
        test_metrics_path = os.path.join(eval_results_dir, "test_metrics.json")
        with open(test_metrics_path, 'w') as f:
            json.dump({"metrics": test_metrics}, f, indent=2)

        # Save examples
        num_examples = min(100, len(test_data["tool_calls"]))
        examples = []
        for i in range(num_examples):
            examples.append({
                "original": test_data["tool_calls"][i],
                "reconstructed": test_data["reconstructed"][i],
                "match": test_data["tool_calls"][i] == test_data["reconstructed"][i]
            })

        examples_path = os.path.join(eval_results_dir, "test_examples.json")
        with open(examples_path, 'w') as f:
            json.dump({"examples": examples}, f, indent=2)

        if config.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
