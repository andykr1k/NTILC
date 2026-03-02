#!/usr/bin/env python3
"""
Training script for NTILC cluster retrieval (Phase 2).

This script trains on NL-command pair datasets (json/jsonl).
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.cluster_retrieval import ClusterRetrieval
from models.intent_embedder import ToolIntentEmbedder
from models.projection_head import ProjectionHead
from models.query_encoder import QueryEncoder
from training.config import IntentEmbeddingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train query encoder for cluster retrieval using NL-command pairs."
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default="checkpoints/intent_embedder/best_model.pt",
        help="Path to Phase 1 checkpoint (intent embedder + projection head).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/man/nl_command_pairs_flat_train_v2.jsonl",
        help="Path to train dataset (json/jsonl). Supports flat or nested records.",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Optional separate validation dataset (json/jsonl).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio if val-data-path is not provided.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on loaded training samples (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers per process.",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=60,
        help="DDP process-group timeout in minutes.",
    )
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
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


def _rows_to_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts flat rows or nested records to:
      {"query": ..., "tool_call": {"tool": ..., "arguments": {"command": ...}}}
    """
    items: List[Dict[str, Any]] = []

    def add_item(tool: str, query: str, command: str) -> None:
        tool = str(tool).strip()
        query = str(query).strip()
        command = str(command).strip()
        if not tool or not query or not command:
            return
        items.append(
            {
                "query": query,
                "tool_call": {
                    "tool": tool,
                    "arguments": {"command": command},
                },
            }
        )

    for row in rows:
        if not isinstance(row, dict):
            continue

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

        add_item(
            tool=row.get("tool", ""),
            query=row.get("nl_query", row.get("query", "")),
            command=row.get("command", ""),
        )

    return items


def load_items_from_path(path_str: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = _load_json_or_jsonl(path)
    items = _rows_to_items(rows)
    if max_samples and max_samples > 0:
        items = items[:max_samples]
    return items


class QueryClusterDataset(Dataset):
    """Dataset for (query, target_cluster_embedding, label)."""

    def __init__(
        self,
        queries: List[str],
        tool_calls: List[Dict[str, Any]],
        tool_labels: List[int],
        intent_embedder: ToolIntentEmbedder,
        projection_head: ProjectionHead,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.queries = queries
        self.tool_calls = tool_calls
        self.labels = tool_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Precompute frozen target embeddings.
        print("Precomputing target embeddings...")
        intent_embedder.eval()
        projection_head.eval()
        target_embeddings: List[torch.Tensor] = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(tool_calls), batch_size):
                batch_tool_calls = tool_calls[i : i + batch_size]
                intent_embeds = intent_embedder(tool_calls=batch_tool_calls)
                projected = projection_head(intent_embeds)
                target_embeddings.append(projected.cpu())

        self.target_embeddings = torch.cat(target_embeddings, dim=0)
        print(f"Precomputed {len(self.target_embeddings)} target embeddings")

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoded = self.tokenizer(
            self.queries[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_embedding": self.target_embeddings[idx],
            "label": self.labels[idx],
        }


def train_epoch(
    query_encoder: QueryEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: IntentEmbeddingConfig,
    epoch: int,
    global_step: int,
    scheduler=None,
    distributed: bool = False,
    is_main_process: bool = True,
) -> Tuple[Dict[str, float], int]:
    query_encoder.train()
    total_losses = {"total_loss": 0.0, "mse_loss": 0.0, "cosine_loss": 0.0}
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch + 1}",
        disable=not is_main_process,
    )
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_embeddings = batch["target_embedding"].to(device)

        query_embeddings = query_encoder(input_ids, attention_mask)
        if target_embeddings.dtype != query_embeddings.dtype:
            target_embeddings = target_embeddings.to(query_embeddings.dtype)

        mse_loss = F.mse_loss(query_embeddings, target_embeddings)
        cosine_sim = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
        cosine_loss = (1 - cosine_sim).mean()
        total_loss = mse_loss + cosine_loss

        loss = total_loss / config.gradient_accumulation_steps
        loss.backward()

        is_accum_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if is_accum_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_losses["total_loss"] += total_loss.item()
        total_losses["mse_loss"] += mse_loss.item()
        total_losses["cosine_loss"] += cosine_loss.item()
        num_batches += 1

        current_step = global_step + batch_idx
        if current_step % config.log_interval == 0:
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "mse": f"{mse_loss.item():.4f}",
                    "cosine": f"{cosine_loss.item():.4f}",
                }
            )
            if config.use_wandb and is_main_process:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/cosine_loss": cosine_loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step": current_step,
                    }
                )

    if distributed:
        reduced = torch.tensor(
            [
                total_losses["total_loss"],
                total_losses["mse_loss"],
                total_losses["cosine_loss"],
                float(num_batches),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_losses["total_loss"] = reduced[0].item()
        total_losses["mse_loss"] = reduced[1].item()
        total_losses["cosine_loss"] = reduced[2].item()
        num_batches = int(reduced[3].item())

    denom = max(1, num_batches)
    for key in total_losses:
        total_losses[key] /= denom
    return total_losses, num_batches


def evaluate(
    query_encoder: QueryEncoder,
    cluster_retrieval: ClusterRetrieval,
    dataloader: DataLoader,
    device: torch.device,
    cluster_centroids: torch.Tensor,
    distributed: bool = False,
    is_main_process: bool = True,
) -> Dict[str, float]:
    query_encoder.eval()
    correct_clusters = 0
    total_samples = 0
    similarity_sum = 0.0
    similarity_count = 0.0
    total_loss = 0.0
    mse_loss_total = 0.0
    cosine_loss_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_embeddings = batch["target_embedding"].to(device)
            labels = batch["label"].to(device)

            query_embeddings = query_encoder(input_ids, attention_mask)
            if target_embeddings.dtype != query_embeddings.dtype:
                target_embeddings = target_embeddings.to(query_embeddings.dtype)

            mse_loss = F.mse_loss(query_embeddings, target_embeddings)
            cosine_sim = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
            cosine_loss = (1 - cosine_sim).mean()
            val_loss = mse_loss + cosine_loss

            total_loss += val_loss.item()
            mse_loss_total += mse_loss.item()
            cosine_loss_total += cosine_loss.item()
            num_batches += 1

            retrieval = cluster_retrieval(
                query_embeddings=query_embeddings,
                cluster_embeddings=cluster_centroids.to(device, dtype=query_embeddings.dtype),
                top_k=1,
            )
            pred_ids = retrieval["cluster_ids"].squeeze(1)
            valid_mask = pred_ids >= 0
            correct_clusters += (pred_ids[valid_mask] == labels[valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()

            similarities = F.cosine_similarity(query_embeddings, target_embeddings, dim=1)
            similarity_sum += similarities.sum().item()
            similarity_count += float(similarities.numel())

    if distributed:
        stats = torch.tensor(
            [
                total_loss,
                mse_loss_total,
                cosine_loss_total,
                float(correct_clusters),
                float(total_samples),
                similarity_sum,
                similarity_count,
                float(num_batches),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        mse_loss_total = stats[1].item()
        cosine_loss_total = stats[2].item()
        correct_clusters = int(stats[3].item())
        total_samples = int(stats[4].item())
        similarity_sum = stats[5].item()
        similarity_count = stats[6].item()
        num_batches = int(stats[7].item())

    avg_similarity = similarity_sum / max(1.0, similarity_count)
    return {
        "total_loss": total_loss / max(1, num_batches),
        "mse_loss": mse_loss_total / max(1, num_batches),
        "cosine_loss": cosine_loss_total / max(1, num_batches),
        "avg_similarity": avg_similarity,
        "cluster_accuracy": correct_clusters / total_samples if total_samples > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    config = IntentEmbeddingConfig()
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

        checkpoint_path = Path(args.phase1_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Phase 1 checkpoint not found: {checkpoint_path}")

        if is_main_process:
            print(f"Loading Phase 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_config = checkpoint.get("config", {})

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_torch_dtype = ckpt_config.get("torch_dtype", config.torch_dtype)
        use_dtype = dtype_map.get(model_torch_dtype, torch.float32)

        intent_embedder = ToolIntentEmbedder(
            model_name=ckpt_config.get("encoder_model", config.encoder_model),
            embedding_dim=ckpt_config.get("intent_embedding_dim", config.intent_embedding_dim),
            pooling_strategy=ckpt_config.get("pooling_strategy", config.pooling_strategy),
            dropout=ckpt_config.get("dropout", config.dropout),
            torch_dtype=model_torch_dtype,
            max_length=ckpt_config.get("max_length", config.max_length),
        ).to(device)
        intent_embedder.load_state_dict(checkpoint["intent_embedder_state_dict"])
        for param in intent_embedder.parameters():
            param.requires_grad = False
        intent_embedder.eval()

        projection_head = ProjectionHead(
            input_dim=ckpt_config.get("intent_embedding_dim", config.intent_embedding_dim),
            output_dim=ckpt_config.get("projection_dim", config.projection_dim),
            dropout=ckpt_config.get("dropout", config.dropout),
        ).to(device).to(use_dtype)
        projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
        for param in projection_head.parameters():
            param.requires_grad = False
        projection_head.eval()

        if config.use_wandb and is_main_process:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=getattr(config, "wandb_run_name", None) or "cluster_retrieval",
                config={**config.__dict__, **vars(args), "world_size": world_size},
            )

        if is_main_process:
            print(f"Loading training data: {args.data_path}")
        all_train_items = load_items_from_path(args.data_path, max_samples=args.max_samples)
        if not all_train_items:
            raise ValueError(f"No usable samples loaded from {args.data_path}")

        if args.val_data_path:
            if is_main_process:
                print(f"Loading validation data: {args.val_data_path}")
            val_items = load_items_from_path(args.val_data_path, max_samples=0)
            train_items = all_train_items
        else:
            rng = random.Random(args.seed)
            rng.shuffle(all_train_items)
            n_val = max(1, int(len(all_train_items) * args.val_ratio))
            val_items = all_train_items[:n_val]
            train_items = all_train_items[n_val:]

        phase1_tool_names = checkpoint.get("tool_names", [])
        tool_names = list(phase1_tool_names) if isinstance(phase1_tool_names, list) else []
        if not tool_names:
            tool_names = sorted(
                {
                    item["tool_call"]["tool"]
                    for item in (train_items + val_items)
                    if item.get("tool_call") and item["tool_call"].get("tool")
                }
            )

        tool_to_idx = {tool: idx for idx, tool in enumerate(tool_names)}
        for item in (train_items + val_items):
            tool = item["tool_call"]["tool"]
            if tool not in tool_to_idx:
                tool_to_idx[tool] = len(tool_names)
                tool_names.append(tool)

        if is_main_process:
            print(f"Train samples: {len(train_items)}")
            print(f"Val samples:   {len(val_items)}")
            print(f"Unique tools:  {len(tool_names)}")

        train_queries = [item["query"] for item in train_items]
        train_tool_calls = [item["tool_call"] for item in train_items]
        train_labels = [tool_to_idx[item["tool_call"]["tool"]] for item in train_items]

        val_queries = [item["query"] for item in val_items]
        val_tool_calls = [item["tool_call"] for item in val_items]
        val_labels = [tool_to_idx[item["tool_call"]["tool"]] for item in val_items]

        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_config.get("encoder_model", config.encoder_model),
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = QueryClusterDataset(
            queries=train_queries,
            tool_calls=train_tool_calls,
            tool_labels=train_labels,
            intent_embedder=intent_embedder,
            projection_head=projection_head,
            tokenizer=tokenizer,
            max_length=256,
        )
        val_dataset = QueryClusterDataset(
            queries=val_queries,
            tool_calls=val_tool_calls,
            tool_labels=val_labels,
            intent_embedder=intent_embedder,
            projection_head=projection_head,
            tokenizer=tokenizer,
            max_length=256,
        )

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
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
        )

        query_encoder = QueryEncoder(
            base_model=ckpt_config.get("encoder_model", config.encoder_model),
            output_dim=ckpt_config.get("projection_dim", config.projection_dim),
            dropout=ckpt_config.get("dropout", config.dropout),
            torch_dtype=model_torch_dtype,
        ).to(device).to(use_dtype)

        if distributed:
            query_encoder = DDP(
                query_encoder,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        cluster_retrieval = ClusterRetrieval(
            embedding_dim=ckpt_config.get("projection_dim", config.projection_dim),
            num_clusters=len(tool_names),
            similarity_type="cosine",
        )
        cluster_assignments = torch.tensor(train_labels, dtype=torch.long)
        cluster_retrieval.update_cluster_centroids(train_dataset.target_embeddings, cluster_assignments)
        cluster_centroids = cluster_retrieval.get_cluster_centroids()

        optimizer = torch.optim.AdamW(
            query_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
        total_steps = max(1, steps_per_epoch * config.num_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-7,
        )

        def unwrap_model(module):
            return module.module if isinstance(module, DDP) else module

        best_val_loss = float("inf")
        global_step = 0
        for epoch in range(config.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            if is_main_process:
                print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            train_losses, num_batches = train_epoch(
                query_encoder=query_encoder,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                config=config,
                epoch=epoch,
                global_step=global_step,
                scheduler=scheduler,
                distributed=distributed,
                is_main_process=is_main_process,
            )
            global_step += num_batches

            val_metrics = evaluate(
                query_encoder=query_encoder,
                cluster_retrieval=cluster_retrieval,
                dataloader=val_loader,
                device=device,
                cluster_centroids=cluster_centroids,
                distributed=distributed,
                is_main_process=is_main_process,
            )
            if is_main_process:
                print(f"Train Loss: {train_losses['total_loss']:.4f}")
                print(f"Val Loss:   {val_metrics['total_loss']:.4f}")
                print(f"Val Acc:    {val_metrics['cluster_accuracy']:.4f}")
                print(f"Val Sim:    {val_metrics['avg_similarity']:.4f}")

                if config.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/epoch_total_loss": train_losses["total_loss"],
                            "train/epoch_mse_loss": train_losses["mse_loss"],
                            "train/epoch_cosine_loss": train_losses["cosine_loss"],
                            "val/total_loss": val_metrics["total_loss"],
                            "val/mse_loss": val_metrics["mse_loss"],
                            "val/cosine_loss": val_metrics["cosine_loss"],
                            "val/avg_similarity": val_metrics["avg_similarity"],
                            "val/cluster_accuracy": val_metrics["cluster_accuracy"],
                        }
                    )

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    checkpoint_out = {
                        "epoch": epoch + 1,
                        "query_encoder_state_dict": unwrap_model(query_encoder).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "cluster_centroids": cluster_centroids,
                        "tool_names": tool_names,
                        "tool_to_idx": tool_to_idx,
                        "phase1_checkpoint": str(checkpoint_path),
                        "config": {**config.__dict__, **vars(args)},
                        "val_metrics": val_metrics,
                    }
                    output_dir = Path(config.output_dir) / "cluster_retrieval"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / "best_model.pt"
                    torch.save(checkpoint_out, output_path)
                    print(f"Saved best model: {output_path} (val_loss={best_val_loss:.4f})")

            if distributed:
                dist.barrier()

        if is_main_process:
            print("\nTraining complete!")
            if config.use_wandb:
                wandb.finish()
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
