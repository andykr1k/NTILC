"""
Training script for NTILC intent embedding (NEW ARCHITECTURE).

Trains intent embedder to map tool intents to 1024-D embeddings,
with projection head to 128-D for similarity computation.
Uses Circle Loss for metric learning and cluster formation.

FIXED: Added embedding noise and query augmentation to prevent collapse.
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Tuple, Dict, List, Any, Optional
import math
import wandb
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import random
import time
from datetime import timedelta

from models.intent_embedder import ToolIntentEmbedder
from models.projection_head import ProjectionHead
from training.config import IntentEmbeddingConfig
from training.losses import CircleLoss, ContrastiveLoss, EmbeddingRegularizationLoss
from evaluation.metrics import compute_cluster_metrics


def augment_query(query: str) -> str:
    """Add MORE variation to prevent collapse."""
    prefixes = [
        "", "Please ", "Could you ", "I need to ", "Help me ", "Can you ",
        "Would you ", "I'd like to ", "I want to ", "Let me ", "Go ahead and "
    ]
    suffixes = [
        "", " please", " thanks", " now", " ASAP", "?", ".",
        " for me", " right away", " if possible", " when you can"
    ]
    
    # Always augment (100% instead of 70%)
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    # Also randomly add/remove punctuation
    if random.random() < 0.3:
        query = query.rstrip('.?!')
    
    return prefix + query + suffix


class IntentDataset(Dataset):
    """Dataset for tool intent training."""
    
    def __init__(
        self,
        tool_calls: List[Dict[str, Any]],
        queries: Optional[List[str]] = None,
        tool_names: Optional[List[str]] = None,
    ):
        """
        Args:
            tool_calls: List of tool call dicts with 'tool' and 'arguments'
            queries: Optional list of natural language queries
        """
        self.tool_calls = tool_calls
        self.queries = queries or [None] * len(tool_calls)
        
        # Extract tool labels from provided list or from current data.
        if tool_names is not None:
            self.tool_names = list(tool_names)
        else:
            self.tool_names = sorted(
                {
                    str(tc.get("tool", "unknown")).strip()
                    for tc in tool_calls
                    if str(tc.get("tool", "")).strip()
                }
            )
        self.tool_to_idx = {tool: idx for idx, tool in enumerate(self.tool_names)}
        self.labels = []
        for tc in tool_calls:
            tool = str(tc.get("tool", "unknown")).strip()
            if tool not in self.tool_to_idx:
                # Add unseen tool to mapping to avoid invalid labels.
                self.tool_to_idx[tool] = len(self.tool_names)
                self.tool_names.append(tool)
            self.labels.append(self.tool_to_idx[tool])
    
    def __len__(self):
        return len(self.tool_calls)
    
    def __getitem__(self, idx):
        return {
            "tool_call": self.tool_calls[idx],
            "query": self.queries[idx],
            "label": self.labels[idx]
        }


def collate_intent_batch(batch):
    """
    Custom collate function for IntentDataset.
    
    Since tool_calls are dictionaries with varying structures,
    we cannot use PyTorch's default collate which tries to
    recursively batch nested dictionaries.
    
    Args:
        batch: List of dicts from IntentDataset.__getitem__
        
    Returns:
        List of dicts (preserves original structure)
    """
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train intent embeddings from NL-command pairs datasets."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/man/nl_command_pairs_flat_train_v2.jsonl",
        help="Path to NL-command dataset (json/jsonl). Supports flat or nested records.",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Optional separate validation data path (json/jsonl).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio when --val-data-path is not provided.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total samples loaded from file (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting/shuffling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (e.g., cuda:0).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs (e.g., '0,1').",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of DataLoader workers per process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-GPU batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override tokenizer max length.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Override warmup ratio when warmup-steps is not set.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the full base encoder.",
    )
    parser.add_argument(
        "--freeze-encoder-layers",
        type=int,
        default=None,
        help="Override number of early encoder layers to freeze.",
    )
    parser.add_argument(
        "--embedding-noise-std",
        type=float,
        default=None,
        help="Override embedding noise std.",
    )
    parser.add_argument(
        "--pooling-strategy",
        type=str,
        choices=["mean", "cls", "max", "attention"],
        default=None,
        help="Override pooling strategy.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Override model dtype.",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=60,
        help="DDP process-group timeout in minutes.",
    )
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    """Initialize DDP from torchrun env vars when WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
        device = torch.device("cuda", local_rank)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
        device = torch.device(args.device)

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}, got {type(data).__name__}")
    return data


def _rows_to_training_items(
    rows: List[Dict[str, Any]],
    augmentation_prob: float,
) -> List[Dict[str, Any]]:
    """
    Convert flat or nested NL-command rows to training items used by intent embedder.
    """
    items: List[Dict[str, Any]] = []

    def add_item(tool: str, query: str, command: str) -> None:
        tool = str(tool).strip()
        query = str(query).strip()
        command = str(command).strip()
        if not tool or not query or not command:
            return
        if random.random() < augmentation_prob:
            query = augment_query(query)
        items.append(
            {
                "tool_call": {
                    "tool": tool,
                    # Keep command context so canonicalization can use example text.
                    "arguments": {"command": command},
                },
                "query": query,
            }
        )

    for row in rows:
        if not isinstance(row, dict):
            continue

        # Nested format: {"tool": "...", "examples": [{"nl_query","command"}...]}
        if "examples" in row:
            tool = row.get("tool", "")
            examples = row.get("examples", [])
            if not isinstance(examples, list):
                continue
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                add_item(
                    tool=tool,
                    query=ex.get("nl_query", ex.get("query", "")),
                    command=ex.get("command", ""),
                )
            continue

        # Flat format: {"tool","nl_query","command",...}
        add_item(
            tool=row.get("tool", ""),
            query=row.get("nl_query", row.get("query", "")),
            command=row.get("command", ""),
        )

    return items


def load_training_items_from_path(
    data_path: str,
    augmentation_prob: float,
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = _load_json_or_jsonl(path)
    items = _rows_to_training_items(rows, augmentation_prob=augmentation_prob)
    if max_samples and max_samples > 0:
        items = items[:max_samples]
    return items


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
    scheduler=None,
    distributed: bool = False,
    is_main_process: bool = True,
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
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch+1}",
        disable=not is_main_process,
    )
    
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(progress_bar):
        batch_start = time.perf_counter()
        tool_calls = [item["tool_call"] for item in batch]
        queries = [item["query"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], device=device)
        
        # Forward pass: intent embedding
        intent_embeddings = intent_embedder(
            tool_calls=tool_calls,
            queries=queries
        )  # (batch_size, 1024)
        
        # ADDED: Add noise to prevent collapse
        if intent_embedder.training and hasattr(config, 'embedding_noise_std'):
            noise = torch.randn_like(intent_embeddings) * config.embedding_noise_std
            intent_embeddings = intent_embeddings + noise
        
        # Project to 128-D
        projected_embeddings = projection_head(intent_embeddings)  # (batch_size, 128)
        
        # Compute losses
        circle_loss = circle_loss_fn(projected_embeddings, labels)
        
        if config.use_contrastive_loss:
            contrastive_loss = contrastive_loss_fn(projected_embeddings, labels)
        else:
            contrastive_loss = projected_embeddings.new_zeros(1)
        
        # Regularization losses
        reg_losses = reg_loss_fn(intent_embeddings)
        
        # Total loss
        total_loss = (
            config.circle_loss_weight * circle_loss +
            config.contrastive_loss_weight * contrastive_loss +
            reg_losses["total_reg_loss"]
        )
        
        # Backward pass
        loss = total_loss / config.gradient_accumulation_steps
        loss.backward()
        
        is_accum_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if is_accum_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(
                list(intent_embedder.parameters()) + list(projection_head.parameters()),
                config.gradient_clip
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Accumulate losses
        total_losses["total_loss"] += total_loss.item()
        total_losses["circle_loss"] += circle_loss.item()
        total_losses["contrastive_loss"] += contrastive_loss.item()
        total_losses["l2_loss"] += reg_losses["l2_loss"].item()
        total_losses["variance_loss"] += reg_losses["variance_loss"].item()
        num_batches += 1
        
        current_step = global_step + batch_idx
        batch_time_sec = time.perf_counter() - batch_start
        if batch_time_sec > 120 and is_main_process:
            print(
                f"[warn] Slow batch detected: epoch={epoch + 1} "
                f"batch={batch_idx + 1}/{len(dataloader)} time={batch_time_sec:.1f}s"
            )
        
        # Update progress bar and log
        if current_step % config.log_interval == 0:
            avg_loss = total_loss.item()
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "circle": f"{circle_loss.item():.4f}",
                "contrastive": f"{contrastive_loss.item():.4f}",
                "bt(s)": f"{batch_time_sec:.1f}",
            })
            
            if config.use_wandb and is_main_process:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/circle_loss": circle_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/l2_loss": reg_losses["l2_loss"].item(),
                    "train/variance_loss": reg_losses["variance_loss"].item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step": current_step
                })
    
    if distributed:
        reduced = torch.tensor(
            [
                total_losses["total_loss"],
                total_losses["circle_loss"],
                total_losses["contrastive_loss"],
                total_losses["l2_loss"],
                total_losses["variance_loss"],
                float(num_batches),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        num_batches = int(reduced[5].item())
        total_losses["total_loss"] = reduced[0].item()
        total_losses["circle_loss"] = reduced[1].item()
        total_losses["contrastive_loss"] = reduced[2].item()
        total_losses["l2_loss"] = reduced[3].item()
        total_losses["variance_loss"] = reduced[4].item()

    # Average losses
    denom = max(1, num_batches)
    for key in total_losses:
        total_losses[key] /= denom
    
    return total_losses, num_batches


def evaluate(
    intent_embedder: ToolIntentEmbedder,
    projection_head: ProjectionHead,
    dataloader: DataLoader,
    device: torch.device,
    config: IntentEmbeddingConfig,
    circle_loss_fn: CircleLoss,
    contrastive_loss_fn: ContrastiveLoss,
    reg_loss_fn: EmbeddingRegularizationLoss,
    tool_names: Optional[List[str]] = None,
    distributed: bool = False,
    world_size: int = 1,
    is_main_process: bool = True,
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
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process):
            tool_calls = [item["tool_call"] for item in batch]
            queries = [item["query"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], device=device)
            
            # Forward pass (no noise during eval)
            intent_embeddings = intent_embedder(
                tool_calls=tool_calls,
                queries=queries
            )
            projected_embeddings = projection_head(intent_embeddings)
            
            # Compute losses
            circle_loss = circle_loss_fn(projected_embeddings, labels)
            
            if config.use_contrastive_loss:
                contrastive_loss = contrastive_loss_fn(projected_embeddings, labels)
            else:
                contrastive_loss = projected_embeddings.new_zeros(1)
            
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
    
    local_embeddings = (
        torch.cat(all_embeddings, dim=0)
        if all_embeddings
        else torch.empty((0, config.projection_dim), dtype=torch.float32)
    )
    local_labels = (
        torch.cat(all_labels, dim=0)
        if all_labels
        else torch.empty((0,), dtype=torch.long)
    )

    if distributed:
        gathered_embeddings: List[Any] = [None for _ in range(world_size)]
        gathered_labels: List[Any] = [None for _ in range(world_size)]
        gathered_tool_calls: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]

        dist.all_gather_object(gathered_embeddings, local_embeddings)
        dist.all_gather_object(gathered_labels, local_labels)
        dist.all_gather_object(gathered_tool_calls, all_tool_calls)

        all_embeddings = [t for t in gathered_embeddings if isinstance(t, torch.Tensor) and t.numel() > 0]
        all_labels = [t for t in gathered_labels if isinstance(t, torch.Tensor) and t.numel() > 0]
        all_tool_calls = [tc for rank_calls in gathered_tool_calls for tc in rank_calls]

        stats = torch.tensor(
            [
                total_loss,
                circle_loss_total,
                contrastive_loss_total,
                l2_loss_total,
                variance_loss_total,
                float(num_batches),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        circle_loss_total = stats[1].item()
        contrastive_loss_total = stats[2].item()
        l2_loss_total = stats[3].item()
        variance_loss_total = stats[4].item()
        num_batches = int(stats[5].item())
    else:
        all_embeddings = [local_embeddings] if local_embeddings.numel() > 0 else []
        all_labels = [local_labels] if local_labels.numel() > 0 else []

    if not all_embeddings or not all_labels:
        return {
            "total_loss": 0.0,
            "circle_loss": 0.0,
            "contrastive_loss": 0.0,
            "l2_loss": 0.0,
            "variance_loss": 0.0,
            "cluster_accuracy": 0.0,
            "intra_cluster_similarity": 0.0,
            "inter_cluster_similarity": 0.0,
            "cluster_separation": 0.0,
            "silhouette_score": 0.0,
        }

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_cluster_metrics(
        all_embeddings_tensor,
        all_labels_tensor,
        all_tool_calls,
        tool_names=tool_names,
    )
    
    # Add loss metrics
    metrics["total_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
    metrics["circle_loss"] = circle_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["contrastive_loss"] = contrastive_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["l2_loss"] = l2_loss_total / num_batches if num_batches > 0 else 0.0
    metrics["variance_loss"] = variance_loss_total / num_batches if num_batches > 0 else 0.0
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    config = IntentEmbeddingConfig()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        config.warmup_ratio = args.warmup_ratio
    if args.freeze_encoder:
        config.freeze_encoder = True
    if args.freeze_encoder_layers is not None:
        config.freeze_encoder_layers = args.freeze_encoder_layers
    if args.embedding_noise_std is not None:
        config.embedding_noise_std = args.embedding_noise_std
    if args.pooling_strategy is not None:
        config.pooling_strategy = args.pooling_strategy
    if args.torch_dtype is not None:
        config.torch_dtype = args.torch_dtype
    if args.disable_wandb:
        config.use_wandb = False

    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    try:
        distributed, rank, world_size, local_rank, device = setup_distributed(args)
        is_main_process = rank == 0

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if is_main_process:
            print(f"Using device: {device} | distributed={distributed} | world_size={world_size}")
            os.makedirs(config.output_dir, exist_ok=True)

        # Initialize wandb only on rank 0.
        if config.use_wandb and is_main_process:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=getattr(config, "wandb_run_name", None) or "intent_embedder",
                config={**config.__dict__, **vars(args), "world_size": world_size},
            )

        # Load training/validation data from NL-command pair datasets only.
        if is_main_process:
            print(f"Loading dataset from: {args.data_path}")
        all_data = load_training_items_from_path(
            data_path=args.data_path,
            augmentation_prob=config.query_augmentation_prob,
            max_samples=args.max_samples,
        )
        if not all_data:
            raise ValueError(f"No usable samples loaded from {args.data_path}")

        if args.val_data_path:
            if is_main_process:
                print(f"Loading validation dataset from: {args.val_data_path}")
            val_data = load_training_items_from_path(
                data_path=args.val_data_path,
                augmentation_prob=0.0,
                max_samples=0,
            )
            train_data = all_data
        else:
            rng = random.Random(args.seed)
            rng.shuffle(all_data)
            n_val = max(1, int(len(all_data) * args.val_ratio))
            val_data = all_data[:n_val]
            train_data = all_data[n_val:]

        combined_data = train_data + val_data
        tool_counts: Dict[str, int] = {}
        for item in combined_data:
            tool = item["tool_call"].get("tool", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        tool_names = sorted(tool_counts.keys())

        if is_main_process:
            print(f"\nLoaded {len(combined_data)} total samples")
            print("Tool distribution:")
            for tool, count in sorted(tool_counts.items()):
                print(f"  {tool}: {count}")
            print(f"Unique tools: {len(tool_names)}")

        train_tool_calls = [item["tool_call"] for item in train_data]
        train_queries = [item["query"] for item in train_data]
        train_dataset = IntentDataset(train_tool_calls, queries=train_queries, tool_names=tool_names)

        val_tool_calls = [item["tool_call"] for item in val_data]
        val_queries = [item["query"] for item in val_data]
        val_dataset = IntentDataset(val_tool_calls, queries=val_queries, tool_names=tool_names)

        if is_main_process:
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val dataset size: {len(val_dataset)}")

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_intent_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_intent_batch,
        )

        if is_main_process:
            print("\nInitializing models...")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_torch_dtype = config.torch_dtype
        use_dtype = dtype_map.get(model_torch_dtype, torch.float32)

        intent_embedder = ToolIntentEmbedder(
            model_name=config.encoder_model,
            embedding_dim=config.intent_embedding_dim,
            pooling_strategy=config.pooling_strategy,
            dropout=config.dropout,
            freeze_base=config.freeze_encoder,
            freeze_layers=config.freeze_encoder_layers,
            torch_dtype=model_torch_dtype,
            max_length=config.max_length,
        ).to(device)

        projection_head = ProjectionHead(
            input_dim=config.intent_embedding_dim,
            output_dim=config.projection_dim,
            dropout=config.dropout,
        ).to(device).to(use_dtype)

        total_params = sum(p.numel() for p in intent_embedder.parameters())
        trainable_params = sum(p.numel() for p in intent_embedder.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in projection_head.parameters())
        if is_main_process:
            print(f"Intent Embedder - Total params: {total_params:,}, Trainable: {trainable_params:,}")
            print(f"Projection Head - Total params: {proj_params:,}")

        if distributed:
            intent_embedder = DDP(
                intent_embedder,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            projection_head = DDP(
                projection_head,
                device_ids=[local_rank],
                output_device=local_rank,
                # Loss fallbacks guarantee gradient flow even on pairless batches.
                find_unused_parameters=False,
            )

        circle_loss_fn = CircleLoss(
            margin=config.circle_loss_margin,
            gamma=config.circle_loss_gamma,
            similarity_type="cosine",
        )
        contrastive_loss_fn = ContrastiveLoss(
            temperature=config.contrastive_temperature,
            use_tool_labels=True,
        )
        reg_loss_fn = EmbeddingRegularizationLoss(
            l2_weight=config.embedding_l2_weight,
            variance_weight=config.embedding_variance_weight,
            target_variance=1.0 / config.intent_embedding_dim,
        )

        if is_main_process:
            print("\nLoss configuration:")
            print(f"  Circle Loss - margin: {config.circle_loss_margin}, gamma: {config.circle_loss_gamma}")
            print(f"  Contrastive Loss - temperature: {config.contrastive_temperature}, enabled: {config.use_contrastive_loss}")
            print(f"  Regularization - L2 weight: {config.embedding_l2_weight}, Variance weight: {config.embedding_variance_weight}")
            print(f"  Embedding noise std: {config.embedding_noise_std}")

        optimizer = torch.optim.AdamW(
            list(intent_embedder.parameters()) + list(projection_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        if config.warmup_steps and config.warmup_steps > 0:
            warmup_steps = min(config.warmup_steps, total_steps)
        else:
            warmup_steps = int(total_steps * config.warmup_ratio)
        if warmup_steps >= total_steps:
            # If explicit warmup is larger than the full run, fall back to ratio.
            warmup_steps = int(total_steps * config.warmup_ratio)
            if warmup_steps >= total_steps:
                warmup_steps = 0

        scheduler = None
        scheduler_name = "none"
        if config.use_lr_scheduler and total_steps > 0:
            if warmup_steps > 0 and warmup_steps < total_steps:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    total_iters=warmup_steps,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=1e-7,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
                scheduler_name = "linear_warmup+cosine"
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=1e-7,
                )
                scheduler_name = "cosine"

        if is_main_process:
            print("\nTraining configuration:")
            print(f"  Epochs: {config.num_epochs}")
            print(f"  Batch size (per GPU): {config.batch_size}")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Total optimizer steps: {total_steps}")
            print(f"  Warmup steps: {warmup_steps}")
            print(f"  Scheduler: {scheduler_name}")

        best_val_loss = float("inf")
        global_step = 0

        def unwrap_model(module):
            return module.module if isinstance(module, DDP) else module

        for epoch in range(config.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            if is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{config.num_epochs}")
                print(f"{'='*60}")

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
                scheduler,
                distributed=distributed,
                is_main_process=is_main_process,
            )
            global_step += num_batches

            val_metrics = evaluate(
                intent_embedder,
                projection_head,
                val_loader,
                device,
                config,
                circle_loss_fn,
                contrastive_loss_fn,
                reg_loss_fn,
                tool_names=train_dataset.tool_names,
                distributed=distributed,
                world_size=world_size,
                is_main_process=is_main_process,
            )

            if is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1} Summary")
                print(f"{'='*60}")
                print(f"Train Loss: {train_losses['total_loss']:.4f}")
                print(f"  - Circle: {train_losses['circle_loss']:.4f}")
                print(f"  - Contrastive: {train_losses['contrastive_loss']:.4f}")
                print(f"  - L2: {train_losses['l2_loss']:.6f}")
                print(f"  - Variance: {train_losses['variance_loss']:.6f}")
                print(f"\nVal Loss: {val_metrics.get('total_loss', 0.0):.4f}")
                print(f"  - Circle: {val_metrics.get('circle_loss', 0.0):.4f}")
                print(f"  - Contrastive: {val_metrics.get('contrastive_loss', 0.0):.4f}")
                print("\nClustering Metrics:")
                print(f"  - Accuracy: {val_metrics.get('cluster_accuracy', 0.0):.4f}")
                print(f"  - Intra-cluster sim: {val_metrics.get('intra_cluster_similarity', 0.0):.4f}")
                print(f"  - Inter-cluster sim: {val_metrics.get('inter_cluster_similarity', 0.0):.4f}")
                print(f"  - Separation: {val_metrics.get('cluster_separation', 0.0):.4f}")
                print(f"  - Silhouette: {val_metrics.get('silhouette_score', 0.0):.4f}")

                if config.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/epoch_loss": train_losses["total_loss"],
                            "train/epoch_circle_loss": train_losses["circle_loss"],
                            "train/epoch_contrastive_loss": train_losses["contrastive_loss"],
                            "val/total_loss": val_metrics.get("total_loss", 0.0),
                            "val/circle_loss": val_metrics.get("circle_loss", 0.0),
                            "val/contrastive_loss": val_metrics.get("contrastive_loss", 0.0),
                            "val/l2_loss": val_metrics.get("l2_loss", 0.0),
                            "val/variance_loss": val_metrics.get("variance_loss", 0.0),
                            "val/cluster_accuracy": val_metrics.get("cluster_accuracy", 0.0),
                            "val/intra_cluster_similarity": val_metrics.get("intra_cluster_similarity", 0.0),
                            "val/inter_cluster_similarity": val_metrics.get("inter_cluster_similarity", 0.0),
                            "val/cluster_separation": val_metrics.get("cluster_separation", 0.0),
                            "val/silhouette_score": val_metrics.get("silhouette_score", 0.0),
                        }
                    )

                val_loss = val_metrics.get("total_loss", train_losses["total_loss"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "epoch": epoch + 1,
                        "intent_embedder_state_dict": unwrap_model(intent_embedder).state_dict(),
                        "projection_head_state_dict": unwrap_model(projection_head).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "tool_names": train_dataset.tool_names,
                        "tool_to_idx": train_dataset.tool_to_idx,
                        "config": config.__dict__,
                        "val_metrics": val_metrics,
                        "train_losses": train_losses,
                    }
                    output_dir = os.path.join(config.output_dir, "intent_embedder")
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
                    print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")

            if distributed:
                dist.barrier()

        if is_main_process:
            print(f"\n{'='*60}")
            print("Training complete!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*60}")

            if config.use_wandb:
                wandb.finish()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
