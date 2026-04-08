from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, get_linear_schedule_with_warmup

from training.train_embedding_space import (
    LOSS_TYPE_ALIASES,
    build_tokenizer,
    choose_device,
    clean_rows,
    compute_tool_centroids,
    embed_texts,
    infer_hidden_size,
    load_jsonl,
    mean_pool,
    normalize_loss_type,
    resolve_output_dir,
    retrieval_accuracy,
    stratified_split,
)
from training.wandb_diagnostics import (
    add_diagnostic_arguments,
    add_wandb_arguments,
    build_wandb_log_payload,
    compute_embedding_diagnostics,
    init_wandb_run,
    log_output_artifact,
    select_diagnostic_indices,
)


DATA_DIR = Path("data/ToolVerifier")
DEFAULT_DATASET_PATH = DATA_DIR / "tool_embedding_dataset.jsonl"
DEFAULT_OUTPUT_DIR = DATA_DIR / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hierarchical tool embedding model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the synthetic tool dataset.",
    )
    parser.add_argument(
        "--hierarchy-path",
        type=str,
        required=True,
        help="Path to a JSON mapping of tool_name -> parent_name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base directory where checkpoints and metrics are saved under architecture/loss subdirectories.",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base text encoder used for the embedding model.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Output embedding size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout used in the projection head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup steps as a fraction of total steps.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.2,
        help="Extra weight that pulls samples toward their matched prototypes in the prototype_ce loss.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="contrastive",
        choices=tuple(LOSS_TYPE_ALIASES),
        help="Training loss to use: prototype_ce (deprecated alias: current), contrastive, or circle.",
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=0.5,
        help="Margin used by the contrastive loss on cosine distance.",
    )
    parser.add_argument(
        "--circle-margin",
        type=float,
        default=0.25,
        help="Margin used by the circle loss.",
    )
    parser.add_argument(
        "--circle-gamma",
        type=float,
        default=32.0,
        help="Scale factor used by the circle loss.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the base encoder and train only the projection head and prototypes.",
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
        default="cuda:4",
        help="Use auto, cuda, cuda:0, or cpu.",
    )
    add_wandb_arguments(parser)
    add_diagnostic_arguments(parser)
    return parser.parse_args()


class HierarchicalToolDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        tool_to_index: Dict[str, int],
        parent_to_index: Dict[str, int],
        tool_to_parent: Dict[str, str],
    ):
        self.rows = list(rows)
        self.tool_to_index = dict(tool_to_index)
        self.parent_to_index = dict(parent_to_index)
        self.tool_to_parent = dict(tool_to_parent)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        tool = row["tool"]
        parent = self.tool_to_parent[tool]
        return {
            "text": row["query"],
            "tool": tool,
            "parent": parent,
            "label": self.tool_to_index[tool],
            "parent_label": self.parent_to_index[parent],
        }


def build_collate_fn(tokenizer, max_length: int):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        parent_labels = torch.tensor([item["parent_label"] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["parent_labels"] = parent_labels
        encoded["texts"] = texts
        return encoded

    return collate


def load_hierarchy_mapping(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")

    mapping: Dict[str, str] = {}
    for raw_tool, raw_parent in payload.items():
        tool = str(raw_tool).strip()
        parent = str(raw_parent).strip()
        if not tool or not parent:
            raise ValueError(f"Hierarchy entries must have non-empty tool and parent names in {path}.")
        mapping[tool] = parent
    return mapping


def validate_hierarchy_mapping(
    tool_names: Sequence[str],
    tool_to_parent: Dict[str, str],
) -> Dict[str, str]:
    expected_tools = set(tool_names)
    provided_tools = set(tool_to_parent)
    missing = sorted(expected_tools - provided_tools)
    extra = sorted(provided_tools - expected_tools)

    if missing or extra:
        message_parts: List[str] = []
        if missing:
            message_parts.append(f"missing mappings for: {missing}")
        if extra:
            message_parts.append(f"extra mappings for: {extra}")
        raise ValueError("Invalid hierarchy mapping: " + "; ".join(message_parts))

    return {tool: tool_to_parent[tool] for tool in tool_names}


class HierarchicalToolEmbeddingModel(nn.Module):
    def __init__(
        self,
        encoder_model: str,
        tool_names: Sequence[str],
        parent_names: Sequence[str],
        tool_to_parent: Sequence[str],
        embedding_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if len(tool_names) != len(tool_to_parent):
            raise ValueError("tool_names and tool_to_parent must have the same length.")

        self.encoder_model = encoder_model
        self.embedding_dim = embedding_dim
        self.dropout_value = dropout
        self.tool_names = list(tool_names)
        self.parent_names = list(parent_names)
        self.tool_to_parent = list(tool_to_parent)

        self.encoder = AutoModel.from_pretrained(encoder_model, trust_remote_code=True)
        hidden_size = infer_hidden_size(self.encoder.config)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
        )

        self.parent_prototypes = nn.Parameter(torch.randn(len(self.parent_names), embedding_dim))
        self.tool_prototypes = nn.Parameter(torch.randn(len(self.tool_names), embedding_dim))
        self.parent_logit_scale = nn.Parameter(torch.tensor(math.log(20.0), dtype=torch.float32))
        self.tool_logit_scale = nn.Parameter(torch.tensor(math.log(20.0), dtype=torch.float32))

        parent_to_index = {parent: index for index, parent in enumerate(self.parent_names)}
        tool_to_parent_index = torch.tensor(
            [parent_to_index[parent] for parent in self.tool_to_parent],
            dtype=torch.long,
        )
        parent_to_tool_mask = torch.zeros(
            len(self.parent_names),
            len(self.tool_names),
            dtype=torch.bool,
        )
        for tool_index, parent_index in enumerate(tool_to_parent_index.tolist()):
            parent_to_tool_mask[parent_index, tool_index] = True

        self.register_buffer("tool_to_parent_index", tool_to_parent_index)
        self.register_buffer("parent_to_tool_mask", parent_to_tool_mask)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        embeddings = self.projection(pooled)
        return F.normalize(embeddings, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embeddings = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        parent_prototypes = F.normalize(self.parent_prototypes, dim=-1)
        tool_prototypes = F.normalize(self.tool_prototypes, dim=-1)
        parent_scale = self.parent_logit_scale.exp().clamp(max=100.0)
        tool_scale = self.tool_logit_scale.exp().clamp(max=100.0)
        parent_logits = embeddings @ parent_prototypes.T * parent_scale
        tool_logits = embeddings @ tool_prototypes.T * tool_scale
        return embeddings, parent_logits, tool_logits

    def mask_tool_logits(self, tool_logits: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
        mask = self.parent_to_tool_mask[parent_indices]
        fill_value = torch.finfo(tool_logits.dtype).min
        return tool_logits.masked_fill(~mask, fill_value)


def compute_similarity_sets(
    model: HierarchicalToolEmbeddingModel,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    parent_labels: torch.Tensor,
) -> Dict[str, Any]:
    parent_prototypes = F.normalize(model.parent_prototypes, dim=-1)
    tool_prototypes = F.normalize(model.tool_prototypes, dim=-1)

    parent_similarities = embeddings @ parent_prototypes.T
    tool_similarities = embeddings @ tool_prototypes.T

    parent_positive = parent_similarities.gather(1, parent_labels.unsqueeze(1)).squeeze(1)
    tool_positive = tool_similarities.gather(1, labels.unsqueeze(1)).squeeze(1)

    parent_negative_mask = torch.ones_like(parent_similarities, dtype=torch.bool)
    parent_negative_mask.scatter_(1, parent_labels.unsqueeze(1), False)

    tool_negative_mask = model.parent_to_tool_mask[parent_labels].clone()
    tool_negative_mask.scatter_(1, labels.unsqueeze(1), False)

    parent_negative_sets = [
        parent_similarities[index][parent_negative_mask[index]]
        for index in range(parent_similarities.size(0))
    ]
    tool_negative_sets = [
        tool_similarities[index][tool_negative_mask[index]]
        for index in range(tool_similarities.size(0))
    ]

    return {
        "parent_positive": parent_positive,
        "parent_negative_sets": parent_negative_sets,
        "parent_matched": parent_prototypes[parent_labels],
        "tool_positive": tool_positive,
        "tool_negative_sets": tool_negative_sets,
        "tool_matched": tool_prototypes[labels],
    }


def contrastive_loss_from_sets(
    positive_similarities: torch.Tensor,
    negative_sets: Sequence[torch.Tensor],
    margin: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    positive_distances = 1.0 - positive_similarities
    positive_loss = positive_distances.pow(2).mean()

    negative_losses: List[torch.Tensor] = []
    active_margins: List[torch.Tensor] = []
    for negatives in negative_sets:
        if negatives.numel() == 0:
            continue
        negative_distances = 1.0 - negatives
        negative_margin = F.relu(margin - negative_distances)
        negative_losses.append(negative_margin.pow(2).mean())
        active_margins.append(negative_margin.mean())

    if negative_losses:
        negative_loss = torch.stack(negative_losses).mean()
        active_negative_margin = torch.stack(active_margins).mean()
    else:
        negative_loss = positive_similarities.new_zeros(())
        active_negative_margin = positive_similarities.new_zeros(())

    loss = positive_loss + negative_loss
    return loss, {
        "positive_distance": float(positive_distances.mean().detach().cpu()),
        "active_negative_margin": float(active_negative_margin.detach().cpu()),
    }


def circle_loss_from_sets(
    positive_similarities: torch.Tensor,
    negative_sets: Sequence[torch.Tensor],
    margin: float,
    gamma: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    op = 1.0 + margin
    on = -margin
    delta_p = 1.0 - margin
    delta_n = margin

    losses: List[torch.Tensor] = []
    hardest_negatives: List[torch.Tensor] = []
    for positive_similarity, negatives in zip(positive_similarities, negative_sets, strict=True):
        alpha_p = F.relu(op - positive_similarity.detach())
        positive_logit = -gamma * alpha_p * (positive_similarity - delta_p)
        if negatives.numel() == 0:
            losses.append(F.softplus(positive_logit))
            continue

        alpha_n = F.relu(negatives.detach() - on)
        negative_logits = gamma * alpha_n * (negatives - delta_n)
        negative_term = torch.logsumexp(negative_logits, dim=0)
        losses.append(F.softplus(positive_logit + negative_term))
        hardest_negatives.append(negatives.max())

    loss = torch.stack(losses).mean()
    hardest_negative = (
        torch.stack(hardest_negatives).mean()
        if hardest_negatives
        else positive_similarities.new_zeros(())
    )
    return loss, {
        "positive_similarity": float(positive_similarities.mean().detach().cpu()),
        "hardest_negative_similarity": float(hardest_negative.detach().cpu()),
    }


def compute_loss(
    model: HierarchicalToolEmbeddingModel,
    embeddings: torch.Tensor,
    parent_logits: torch.Tensor,
    tool_logits: torch.Tensor,
    labels: torch.Tensor,
    parent_labels: torch.Tensor,
    loss_type: str,
    alignment_weight: float,
    contrastive_margin: float,
    circle_margin: float,
    circle_gamma: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    similarity_sets = compute_similarity_sets(
        model=model,
        embeddings=embeddings,
        labels=labels,
        parent_labels=parent_labels,
    )

    if loss_type == "prototype_ce":
        masked_child_logits = model.mask_tool_logits(tool_logits, parent_labels)
        parent_ce = F.cross_entropy(parent_logits, parent_labels)
        child_ce = F.cross_entropy(masked_child_logits, labels)
        parent_alignment = (1.0 - (embeddings * similarity_sets["parent_matched"]).sum(dim=-1)).mean()
        child_alignment = (1.0 - (embeddings * similarity_sets["tool_matched"]).sum(dim=-1)).mean()
        alignment = 0.5 * (parent_alignment + child_alignment)
        loss = parent_ce + child_ce + alignment_weight * alignment
        return loss, {
            "parent_cross_entropy": float(parent_ce.detach().cpu()),
            "child_cross_entropy": float(child_ce.detach().cpu()),
            "parent_alignment": float(parent_alignment.detach().cpu()),
            "child_alignment": float(child_alignment.detach().cpu()),
        }

    if loss_type == "contrastive":
        parent_loss, parent_metrics = contrastive_loss_from_sets(
            positive_similarities=similarity_sets["parent_positive"],
            negative_sets=similarity_sets["parent_negative_sets"],
            margin=contrastive_margin,
        )
        child_loss, child_metrics = contrastive_loss_from_sets(
            positive_similarities=similarity_sets["tool_positive"],
            negative_sets=similarity_sets["tool_negative_sets"],
            margin=contrastive_margin,
        )
        loss = parent_loss + child_loss
        return loss, {
            "parent_positive_distance": parent_metrics["positive_distance"],
            "parent_active_negative_margin": parent_metrics["active_negative_margin"],
            "child_positive_distance": child_metrics["positive_distance"],
            "child_active_negative_margin": child_metrics["active_negative_margin"],
        }

    if loss_type == "circle":
        parent_loss, parent_metrics = circle_loss_from_sets(
            positive_similarities=similarity_sets["parent_positive"],
            negative_sets=similarity_sets["parent_negative_sets"],
            margin=circle_margin,
            gamma=circle_gamma,
        )
        child_loss, child_metrics = circle_loss_from_sets(
            positive_similarities=similarity_sets["tool_positive"],
            negative_sets=similarity_sets["tool_negative_sets"],
            margin=circle_margin,
            gamma=circle_gamma,
        )
        loss = parent_loss + child_loss
        return loss, {
            "parent_positive_similarity": parent_metrics["positive_similarity"],
            "parent_hardest_negative_similarity": parent_metrics["hardest_negative_similarity"],
            "child_positive_similarity": child_metrics["positive_similarity"],
            "child_hardest_negative_similarity": child_metrics["hardest_negative_similarity"],
        }

    raise ValueError(f"Unsupported loss type: {loss_type}")


@torch.inference_mode()
def compute_parent_centroids(
    model: HierarchicalToolEmbeddingModel,
    tokenizer,
    rows: Sequence[Dict[str, str]],
    parent_names: Sequence[str],
    tool_to_parent: Dict[str, str],
    device: torch.device | str,
    max_length: int,
    batch_size: int = 32,
) -> torch.Tensor:
    texts = [row["query"] for row in rows]
    embeddings = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        progress_desc="Embedding for parent centroids",
    )
    parent_to_index = {parent: index for index, parent in enumerate(parent_names)}
    labels = torch.tensor(
        [parent_to_index[tool_to_parent[row["tool"]]] for row in rows],
        dtype=torch.long,
    )

    centroids: List[torch.Tensor] = []
    for index in tqdm(
        range(len(parent_names)),
        desc="Averaging parent centroids",
        unit="parent",
        leave=False,
    ):
        mask = labels == index
        if not mask.any():
            centroids.append(torch.zeros(model.embedding_dim))
            continue
        centroid = embeddings[mask].mean(dim=0)
        centroids.append(F.normalize(centroid, dim=-1))
    return torch.stack(centroids, dim=0)


@torch.inference_mode()
def parent_retrieval_accuracy(
    model: HierarchicalToolEmbeddingModel,
    tokenizer,
    rows: Sequence[Dict[str, str]],
    parent_centroids: torch.Tensor,
    parent_names: Sequence[str],
    tool_to_parent: Dict[str, str],
    device: torch.device | str,
    max_length: int,
    batch_size: int = 32,
) -> float:
    if not rows:
        return float("nan")

    embeddings = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=[row["query"] for row in rows],
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        progress_desc="Embedding for parent retrieval",
    )
    scores = embeddings @ parent_centroids.T
    predictions = scores.argmax(dim=-1)
    parent_to_index = {parent: index for index, parent in enumerate(parent_names)}
    labels = torch.tensor(
        [parent_to_index[tool_to_parent[row["tool"]]] for row in rows],
        dtype=torch.long,
    )
    return float((predictions == labels).float().mean().item())


@torch.inference_mode()
def classification_accuracies(
    model: HierarchicalToolEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    total = 0
    correct_parent = 0
    correct_tool = 0
    model.eval()
    for batch in tqdm(
        loader,
        desc="Validation classification",
        unit="batch",
        leave=False,
    ):
        labels = batch["labels"].to(device)
        parent_labels = batch["parent_labels"].to(device)
        encoded = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask"}
        }
        _, parent_logits, tool_logits = model(**encoded)
        parent_predictions = parent_logits.argmax(dim=-1)
        masked_tool_logits = model.mask_tool_logits(tool_logits, parent_predictions)
        tool_predictions = masked_tool_logits.argmax(dim=-1)
        total += labels.numel()
        correct_parent += int((parent_predictions == parent_labels).sum().item())
        correct_tool += int((tool_predictions == labels).sum().item())
    if total == 0:
        return float("nan"), float("nan")
    return correct_tool / total, correct_parent / total


def save_checkpoint(
    path: Path,
    model: HierarchicalToolEmbeddingModel,
    tool_names: Sequence[str],
    tool_centroids: torch.Tensor,
    parent_names: Sequence[str],
    parent_centroids: torch.Tensor,
    tool_to_parent: Dict[str, str],
    args: argparse.Namespace,
    history: Sequence[Dict[str, Any]],
) -> None:
    payload = {
        "architecture": "hierarchical",
        "loss_name": normalize_loss_type(args.loss_type),
        "encoder_model": args.encoder_model,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "tool_names": list(tool_names),
        "tool_centroids": tool_centroids.cpu(),
        "parent_names": list(parent_names),
        "parent_centroids": parent_centroids.cpu(),
        "tool_to_parent": dict(tool_to_parent),
        "model_state_dict": model.state_dict(),
        "history": list(history),
        "training_args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.loss_type = normalize_loss_type(args.loss_type)

    dataset_path = Path(args.dataset_path)
    hierarchy_path = Path(args.hierarchy_path)
    output_dir = resolve_output_dir(args.output_dir, architecture="hierarchical", loss_type=args.loss_type)
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = clean_rows(load_jsonl(dataset_path))
    if len(rows) < 2:
        raise ValueError("Need at least two rows to train.")

    tool_names = sorted({row["tool"] for row in rows})
    tool_to_parent = validate_hierarchy_mapping(tool_names, load_hierarchy_mapping(hierarchy_path))
    parent_names = sorted({tool_to_parent[tool] for tool in tool_names})
    tool_to_index = {tool: index for index, tool in enumerate(tool_names)}
    parent_to_index = {parent: index for index, parent in enumerate(parent_names)}

    if args.loss_type in {"contrastive", "circle"} and len(tool_names) < 2:
        raise ValueError(f"{args.loss_type} loss requires at least two distinct tools.")

    train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed)
    diagnostic_indices = select_diagnostic_indices(
        row_count=len(val_rows),
        max_samples=args.diagnostic_max_samples,
        seed=args.seed,
    )
    diagnostic_rows = [val_rows[index] for index in diagnostic_indices]

    print(f"Loaded {len(rows)} rows from {dataset_path}")
    print(f"Tools: {len(tool_names)}")
    print(f"Parents: {len(parent_names)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Loss: {args.loss_type}")
    print(f"Output dir: {output_dir}")

    device = choose_device(args.device)
    tokenizer = build_tokenizer(args.encoder_model)
    collate_fn = build_collate_fn(tokenizer, args.max_length)

    wandb_module, wandb_run = init_wandb_run(
        enabled=args.wandb,
        args=args,
        embedding_type="hierarchical",
        loss_type=args.loss_type,
        config={
            **vars(args),
            "dataset_path": str(dataset_path),
            "hierarchy_path": str(hierarchy_path),
            "output_dir": str(output_dir),
            "tool_count": len(tool_names),
            "parent_count": len(parent_names),
            "train_row_count": len(train_rows),
            "val_row_count": len(val_rows),
            "diagnostic_sample_size": len(diagnostic_rows),
        },
    )

    train_loader = DataLoader(
        HierarchicalToolDataset(train_rows, tool_to_index, parent_to_index, tool_to_parent),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        HierarchicalToolDataset(val_rows, tool_to_index, parent_to_index, tool_to_parent),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = HierarchicalToolEmbeddingModel(
        encoder_model=args.encoder_model,
        tool_names=tool_names,
        parent_names=parent_names,
        tool_to_parent=[tool_to_parent[tool] for tool in tool_names],
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_retrieval = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    try:
        epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_iterator:
            epoch_start_time = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_batches = 0
            running_loss_metric_sums: Dict[str, float] = {}

            batch_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch} train",
                unit="batch",
                leave=False,
            )
            for batch in batch_iterator:
                labels = batch["labels"].to(device)
                parent_labels = batch["parent_labels"].to(device)
                encoded = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"}
                }

                optimizer.zero_grad()
                embeddings, parent_logits, tool_logits = model(**encoded)
                loss, loss_metrics = compute_loss(
                    model=model,
                    embeddings=embeddings,
                    parent_logits=parent_logits,
                    tool_logits=tool_logits,
                    labels=labels,
                    parent_labels=parent_labels,
                    loss_type=args.loss_type,
                    alignment_weight=args.alignment_weight,
                    contrastive_margin=args.contrastive_margin,
                    circle_margin=args.circle_margin,
                    circle_gamma=args.circle_gamma,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += float(loss.detach().cpu())
                running_batches += 1
                for name, value in loss_metrics.items():
                    running_loss_metric_sums[name] = running_loss_metric_sums.get(name, 0.0) + float(value)
                batch_iterator.set_postfix(loss=running_loss / running_batches)

            averaged_loss_metrics = {
                f"train_{name}": value / max(1, running_batches)
                for name, value in running_loss_metric_sums.items()
            }

            train_tool_centroids = compute_tool_centroids(
                model=model,
                tokenizer=tokenizer,
                rows=train_rows,
                tool_names=tool_names,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            train_parent_centroids = compute_parent_centroids(
                model=model,
                tokenizer=tokenizer,
                rows=train_rows,
                parent_names=parent_names,
                tool_to_parent=tool_to_parent,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            train_tool_retrieval = retrieval_accuracy(
                model=model,
                tokenizer=tokenizer,
                rows=train_rows,
                centroids=train_tool_centroids,
                tool_names=tool_names,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            train_parent_retrieval = parent_retrieval_accuracy(
                model=model,
                tokenizer=tokenizer,
                rows=train_rows,
                parent_centroids=train_parent_centroids,
                parent_names=parent_names,
                tool_to_parent=tool_to_parent,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            if args.wandb:
                if val_rows:
                    val_embeddings = embed_texts(
                        model=model,
                        tokenizer=tokenizer,
                        texts=[row["query"] for row in val_rows],
                        device=device,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                        progress_desc="Embedding for hierarchical validation diagnostics",
                    )
                    val_tool_labels = torch.tensor(
                        [tool_to_index[row["tool"]] for row in val_rows],
                        dtype=torch.long,
                    )
                    val_parent_labels = torch.tensor(
                        [parent_to_index[tool_to_parent[row["tool"]]] for row in val_rows],
                        dtype=torch.long,
                    )
                    val_tool_scores = val_embeddings @ train_tool_centroids.T
                    val_parent_scores = val_embeddings @ train_parent_centroids.T
                    val_tool_retrieval = float(
                        (val_tool_scores.argmax(dim=-1) == val_tool_labels).float().mean().item()
                    )
                    val_parent_retrieval = float(
                        (val_parent_scores.argmax(dim=-1) == val_parent_labels).float().mean().item()
                    )
                else:
                    val_embeddings = torch.empty((0, train_tool_centroids.size(1)))
                    val_tool_retrieval = float("nan")
                    val_parent_retrieval = float("nan")
            else:
                val_embeddings = torch.empty((0, train_tool_centroids.size(1)))
                val_tool_retrieval = retrieval_accuracy(
                    model=model,
                    tokenizer=tokenizer,
                    rows=val_rows,
                    centroids=train_tool_centroids,
                    tool_names=tool_names,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )
                val_parent_retrieval = parent_retrieval_accuracy(
                    model=model,
                    tokenizer=tokenizer,
                    rows=val_rows,
                    parent_centroids=train_parent_centroids,
                    parent_names=parent_names,
                    tool_to_parent=tool_to_parent,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )

            val_tool_classification, val_parent_classification = classification_accuracies(
                model,
                val_loader,
                device,
            )

            diagnostic_bundles: list[Dict[str, Any]] = []
            if args.wandb:
                diagnostic_embeddings = (
                    val_embeddings[diagnostic_indices]
                    if diagnostic_indices
                    else val_embeddings[:0]
                )
                tool_diagnostics = compute_embedding_diagnostics(
                    rows=diagnostic_rows,
                    embeddings=diagnostic_embeddings,
                    label_names=tool_names,
                    row_label_names=[row["tool"] for row in diagnostic_rows],
                    centroids=train_tool_centroids,
                    seed=args.seed,
                    top_k_labels=args.diagnostic_top_k_tools,
                    overlap_margin=args.diagnostic_overlap_margin,
                    min_label_samples=args.diagnostic_min_tool_samples,
                    heatmap_labels=args.diagnostic_heatmap_tools,
                    label_column="tool",
                    label_namespace="validation/tool",
                )
                parent_diagnostics = compute_embedding_diagnostics(
                    rows=diagnostic_rows,
                    embeddings=diagnostic_embeddings,
                    label_names=parent_names,
                    row_label_names=[tool_to_parent[row["tool"]] for row in diagnostic_rows],
                    centroids=train_parent_centroids,
                    seed=args.seed,
                    top_k_labels=args.diagnostic_top_k_tools,
                    overlap_margin=args.diagnostic_overlap_margin,
                    min_label_samples=args.diagnostic_min_tool_samples,
                    heatmap_labels=args.diagnostic_heatmap_tools,
                    label_column="parent",
                    label_namespace="validation/parent",
                )
                diagnostic_bundles.extend([tool_diagnostics, parent_diagnostics])
            else:
                tool_diagnostics = None
                parent_diagnostics = None

            score = val_tool_retrieval if not math.isnan(val_tool_retrieval) else train_tool_retrieval
            improved = score > best_val_retrieval
            if improved:
                best_val_retrieval = score
                best_epoch = epoch

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, running_batches),
                "train_tool_retrieval_accuracy": train_tool_retrieval,
                "train_parent_retrieval_accuracy": train_parent_retrieval,
                "val_tool_retrieval_accuracy": val_tool_retrieval,
                "val_parent_retrieval_accuracy": val_parent_retrieval,
                "val_tool_classification_accuracy": val_tool_classification,
                "val_parent_classification_accuracy": val_parent_classification,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "epoch_duration_seconds": time.perf_counter() - epoch_start_time,
                "best_val_tool_retrieval_accuracy": best_val_retrieval,
                "best_epoch": best_epoch,
            }
            epoch_metrics.update(averaged_loss_metrics)
            if tool_diagnostics is not None:
                epoch_metrics.update(tool_diagnostics["scalars"])
            if parent_diagnostics is not None:
                epoch_metrics.update(parent_diagnostics["scalars"])

            history.append(epoch_metrics)
            epoch_iterator.set_postfix(
                train_loss=f"{epoch_metrics['train_loss']:.4f}",
                val_tool_retrieval=(
                    "nan"
                    if math.isnan(epoch_metrics["val_tool_retrieval_accuracy"])
                    else f"{epoch_metrics['val_tool_retrieval_accuracy']:.4f}"
                ),
            )
            tqdm.write(json.dumps(epoch_metrics, indent=2))

            save_checkpoint(
                path=output_dir / "last.pt",
                model=model,
                tool_names=tool_names,
                tool_centroids=train_tool_centroids,
                parent_names=parent_names,
                parent_centroids=train_parent_centroids,
                tool_to_parent=tool_to_parent,
                args=args,
                history=history,
            )

            if improved:
                save_checkpoint(
                    path=output_dir / "best.pt",
                    model=model,
                    tool_names=tool_names,
                    tool_centroids=train_tool_centroids,
                    parent_names=parent_names,
                    parent_centroids=train_parent_centroids,
                    tool_to_parent=tool_to_parent,
                    args=args,
                    history=history,
                )

            if wandb_run is not None and wandb_module is not None:
                wandb_payload = build_wandb_log_payload(
                    wandb_module=wandb_module,
                    scalars=epoch_metrics,
                    diagnostics=diagnostic_bundles,
                )
                wandb_run.log(wandb_payload, step=epoch)
                wandb_run.summary["best_val_tool_retrieval_accuracy"] = best_val_retrieval
                wandb_run.summary["best_epoch"] = best_epoch
                if tool_diagnostics is not None:
                    for key, value in tool_diagnostics["summary"].items():
                        wandb_run.summary[key] = value
                if parent_diagnostics is not None:
                    for key, value in parent_diagnostics["summary"].items():
                        wandb_run.summary[key] = value

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if wandb_run is not None and wandb_module is not None:
            log_output_artifact(
                wandb_module=wandb_module,
                run=wandb_run,
                output_dir=output_dir,
            )

        print(f"Saved checkpoints to {output_dir}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
