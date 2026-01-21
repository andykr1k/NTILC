"""
Training script for NTILC Phase 2: LLM Integration.

Trains an LLM to predict tool call embeddings from natural language queries.
Uses the trained autoencoder from Phase 1 to:
1. Compute target embeddings for training
2. Decode predicted embeddings for evaluation
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
import json
import os

from models.llm_integration import ToolPredictionLLM, NLToolCallDataset
from models.autoencoder import ToolInvocationAutoencoder
from training.config import LLMIntegrationConfig
from training.data_generator import NaturalLanguageToolCallGenerator, OutputFormat
from evaluation.metrics import exact_match_accuracy, tool_accuracy


def train_epoch(
    model: ToolPredictionLLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: LLMIntegrationConfig,
    epoch: int,
    global_step: int,
    scheduler=None
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    
    total_losses = {
        "total_loss": 0.0,
        "embedding_loss": 0.0,
        "cosine_loss": 0.0,
        "tool_loss": 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_embedding = batch["target_embedding"].to(device)
        tool_label = batch["tool_label"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_embeddings=target_embedding,
            target_tool_labels=tool_label
        )
        
        # Compute total loss
        embedding_loss = outputs["embedding_loss"]
        cosine_loss = outputs["cosine_loss"]
        tool_loss = outputs["tool_loss"]
        
        total_loss = (
            config.embedding_loss_weight * (embedding_loss + cosine_loss) +
            config.auxiliary_tool_cls_weight * tool_loss
        )
        
        # Backward pass
        optimizer.zero_grad()
        
        # Gradient accumulation
        loss = total_loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        # Accumulate losses
        total_losses["total_loss"] += total_loss.item()
        total_losses["embedding_loss"] += embedding_loss.item()
        total_losses["cosine_loss"] += cosine_loss.item()
        total_losses["tool_loss"] += tool_loss.item()
        num_batches += 1
        
        current_step = global_step + batch_idx
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": total_loss.item(),
            "emb_loss": embedding_loss.item(),
            "tool_loss": tool_loss.item()
        })
        
        # Logging
        if batch_idx % config.log_interval == 0 and config.use_wandb:
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/embedding_loss": embedding_loss.item(),
                "train/cosine_loss": cosine_loss.item(),
                "train/tool_loss": tool_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/step": current_step
            }, step=current_step)
    
    # Compute averages
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    new_global_step = global_step + len(dataloader)
    return avg_losses, new_global_step


def evaluate(
    model: ToolPredictionLLM,
    dataloader: DataLoader,
    device: torch.device,
    split: str = "val"
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_queries = []
    all_ground_truth = []
    all_predictions = []
    all_tool_labels = []
    all_tool_preds = []
    
    total_embedding_loss = 0.0
    total_cosine_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_embedding = batch["target_embedding"].to(device)
            tool_label = batch["tool_label"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_embeddings=target_embedding,
                target_tool_labels=tool_label
            )
            
            total_embedding_loss += outputs["embedding_loss"].item()
            total_cosine_loss += outputs["cosine_loss"].item()
            num_batches += 1
            
            # Get predicted tool calls (if decoder loaded)
            if model.decoder is not None:
                embeddings = outputs["embedding"]
                decoder_result = model.decoder(embeddings)
                generated_ids = decoder_result["generated_ids"]
                
                for ids in generated_ids:
                    pred = model.decoder.tokenizer.decode(ids, skip_special_tokens=True)
                    all_predictions.append(pred)
            
            # Tool classification predictions
            tool_preds = outputs["tool_logits"].argmax(dim=1)
            all_tool_preds.extend(tool_preds.cpu().tolist())
            all_tool_labels.extend(tool_label.cpu().tolist())
            
            all_queries.extend(batch["query"])
            all_ground_truth.extend(batch["tool_call"])
    
    metrics = {
        "embedding_loss": total_embedding_loss / num_batches,
        "cosine_loss": total_cosine_loss / num_batches,
    }
    
    # Tool classification accuracy
    correct = sum(p == l for p, l in zip(all_tool_preds, all_tool_labels))
    metrics["tool_cls_accuracy"] = correct / len(all_tool_labels)
    
    # Full tool call accuracy (if decoder loaded)
    if all_predictions:
        metrics["exact_match_accuracy"] = exact_match_accuracy(all_ground_truth, all_predictions)
        metrics["tool_accuracy"] = tool_accuracy(all_ground_truth, all_predictions)
    
    return metrics


def main():
    """Main training function."""
    config = LLMIntegrationConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config.use_wandb:
        import datetime
        run_name = config.wandb_run_name or f"llm_integration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=config.__dict__,
            name=run_name,
            tags=["llm_integration", "phase2"]
        )
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    # Load autoencoder
    print(f"Loading autoencoder from {config.autoencoder_checkpoint}...")
    checkpoint = torch.load(config.autoencoder_checkpoint, map_location='cpu')
    ae_config = checkpoint.get('config', {})
    
    autoencoder = ToolInvocationAutoencoder(
        embedding_dim=ae_config.get('embedding_dim', config.embedding_dim),
        encoder_model=ae_config.get('encoder_model', 'google/flan-t5-base'),
        decoder_model=ae_config.get('decoder_model', 'google/flan-t5-base'),
        max_length=ae_config.get('max_length', 256),
        torch_dtype=ae_config.get('torch_dtype', 'bfloat16')
    )
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Freeze autoencoder
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    print(f"Loaded autoencoder (embedding_dim={ae_config.get('embedding_dim', config.embedding_dim)})")
    
    # Generate or load NL-tool call pairs
    output_format = OutputFormat.JSON if config.output_format == "json" else OutputFormat.PYTHON
    generator = NaturalLanguageToolCallGenerator(output_format=output_format)
    
    train_data_path = os.path.join(config.data_dir, "train_data.jsonl")
    val_data_path = os.path.join(config.data_dir, "val_data.jsonl")
    test_data_path = os.path.join(config.data_dir, "test_data.jsonl")
    
    if not os.path.exists(train_data_path):
        print(f"Generating {config.num_train_samples} training samples...")
        train_data = generator.generate_dataset(config.num_train_samples)
        generator.save_dataset(train_data, train_data_path)
        
        print(f"Generating {config.num_val_samples} validation samples...")
        val_data = generator.generate_dataset(config.num_val_samples)
        generator.save_dataset(val_data, val_data_path)
        
        print(f"Generating {config.num_test_samples} test samples...")
        test_data = generator.generate_dataset(config.num_test_samples)
        generator.save_dataset(test_data, test_data_path)
    else:
        train_data = generator.load_dataset(train_data_path)
        val_data = generator.load_dataset(val_data_path)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Sample: {train_data[0]}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = NLToolCallDataset(train_data, autoencoder, tokenizer)
    val_dataset = NLToolCallDataset(val_data, autoencoder, tokenizer)
    
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
    model = ToolPredictionLLM(
        base_model=config.base_model,
        embedding_dim=ae_config.get('embedding_dim', config.embedding_dim),
        num_tools=6,
        dropout=0.15,
        torch_dtype=config.torch_dtype
    )
    model.load_decoder(config.autoencoder_checkpoint, device)
    model = model.to(device)
    
    print(f"Model parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    if config.use_wandb:
        wandb.config.update({
            "total_params": model.get_total_params(),
            "trainable_params": model.get_trainable_params()
        })
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
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
    
    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0
    global_step = 0
    
    print("Starting training...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, device, config,
            epoch, global_step, scheduler
        )
        print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Embedding: {train_metrics['embedding_loss']:.4f} | "
              f"Tool: {train_metrics['tool_loss']:.4f}")
        
        if config.use_wandb:
            wandb.log({
                "train/epoch_total_loss": train_metrics['total_loss'],
                "train/epoch_embedding_loss": train_metrics['embedding_loss'],
                "train/epoch": epoch + 1
            }, step=global_step)
        
        # Evaluate
        if (epoch + 1) % max(1, config.eval_interval // len(train_loader)) == 0:
            val_metrics = evaluate(model, val_loader, device, split="val")
            
            print(f"Validation Metrics:")
            print(f"  Tool Classification Accuracy: {val_metrics.get('tool_cls_accuracy', 0):.4f}")
            print(f"  Embedding Loss: {val_metrics.get('embedding_loss', 0):.4f}")
            if 'exact_match_accuracy' in val_metrics:
                print(f"  Exact Match Accuracy: {val_metrics.get('exact_match_accuracy', 0):.4f}")
                print(f"  Tool Accuracy: {val_metrics.get('tool_accuracy', 0):.4f}")
            
            if config.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
            
            # Early stopping
            val_accuracy = val_metrics.get('exact_match_accuracy', val_metrics.get('tool_cls_accuracy', 0))
            
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
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
    
    print("\nTraining complete!")
    
    # Final test evaluation
    if os.path.exists(test_data_path):
        print("Evaluating on test set...")
        test_data = generator.load_dataset(test_data_path)
        test_dataset = NLToolCallDataset(test_data, autoencoder, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        test_metrics = evaluate(model, test_loader, device, split="test")
        
        print("Test Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save results
        results_path = os.path.join(config.output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        if config.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
