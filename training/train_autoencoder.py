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
    config: AutoencoderConfig
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
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
        
        # Shift targets for next-token prediction
        targets = target_ids[:, 1:]  # (batch_size, seq_len-1)
        target_mask = attention_mask[:, 1:]  # (batch_size, seq_len-1)
        
        # Compute loss (only on non-padding tokens)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Mask out padding tokens
        loss = (loss * target_mask.reshape(-1)).sum() / target_mask.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Logging
        if batch_idx % config.log_interval == 0:
            if config.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": batch_idx
                })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"loss": avg_loss}


def evaluate(
    model: ToolInvocationAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    config: AutoencoderConfig
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    all_tool_calls = []
    all_reconstructed = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
    
    return metrics


def main():
    """Main training function."""
    # Load configuration
    config = AutoencoderConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=config.__dict__,
            name="autoencoder_training"
        )
    
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
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        if (epoch + 1) % (config.eval_interval // len(train_loader)) == 0 or epoch == 0:
            val_metrics = evaluate(model, val_loader, device, config)
            print(f"Validation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Log to wandb
            if config.use_wandb:
                wandb.log({
                    "val/" + k: v for k, v in val_metrics.items()
                })
            
            # Early stopping
            val_loss = val_metrics.get("loss", float('inf'))
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
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Periodic checkpoint
        if (epoch + 1) % (config.save_interval // len(train_loader)) == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__
            }, checkpoint_path)
    
    print("\nTraining complete!")
    
    # Final evaluation on test set
    if os.path.exists(test_data_path):
        print("Evaluating on test set...")
        test_tool_calls = generator.load_dataset(test_data_path)
        test_dataset = ToolInvocationDataset(test_tool_calls, tokenizer, config.max_length)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        test_metrics = evaluate(model, test_loader, device, config)
        print("Test Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test metrics
        with open(os.path.join(config.output_dir, "test_metrics.json"), 'w') as f:
            json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
