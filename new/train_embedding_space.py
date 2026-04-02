from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple tool embedding model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolCall15/tool_embedding_dataset.jsonl",
        help="Path to the synthetic tool dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolCall15/output",
        help="Where to save checkpoints and metrics.",
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
        help="Extra weight that pulls samples toward their tool prototype.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the base encoder and train only the projection head and tool prototypes.",
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
        default="cuda:2",
        help="Use auto, cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Loading dataset", unit="line"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for row in tqdm(rows, desc="Cleaning rows", unit="row"):
        tool = str(row.get("tool", "")).strip()
        query = str(row.get("query", row.get("text", ""))).strip()
        if tool and query:
            cleaned.append({"tool": tool, "query": query})
    return cleaned


def stratified_split(
    rows: Sequence[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if val_ratio <= 0:
        return list(rows), []

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["tool"]].append(dict(row))

    rng = random.Random(seed)
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []

    for tool_rows in tqdm(
        grouped.values(),
        total=len(grouped),
        desc="Splitting tools",
        unit="tool",
        leave=False,
    ):
        rng.shuffle(tool_rows)
        if len(tool_rows) < 2:
            train_rows.extend(tool_rows)
            continue
        val_count = max(1, int(round(len(tool_rows) * val_ratio)))
        val_count = min(val_count, len(tool_rows) - 1)
        val_rows.extend(tool_rows[:val_count])
        train_rows.extend(tool_rows[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


class ToolDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], tool_to_index: Dict[str, int]):
        self.rows = list(rows)
        self.tool_to_index = dict(tool_to_index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        return {
            "text": row["query"],
            "tool": row["tool"],
            "label": self.tool_to_index[row["tool"]],
        }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine encoder hidden size.")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class SimpleToolEmbeddingModel(nn.Module):
    def __init__(
        self,
        encoder_model: str,
        num_tools: int,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.embedding_dim = embedding_dim
        self.dropout_value = dropout
        self.encoder = AutoModel.from_pretrained(encoder_model, trust_remote_code=True)
        hidden_size = infer_hidden_size(self.encoder.config)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
        )
        self.tool_prototypes = nn.Parameter(torch.randn(num_tools, embedding_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0), dtype=torch.float32))

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
        prototypes = F.normalize(self.tool_prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = embeddings @ prototypes.T * scale
        return embeddings, logits


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer needs a pad_token, eos_token, or unk_token.")
    return tokenizer


def build_collate_fn(tokenizer, max_length: int):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["texts"] = texts
        return encoded

    return collate


def compute_loss(
    model: SimpleToolEmbeddingModel,
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    alignment_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    cross_entropy = F.cross_entropy(logits, labels)
    prototypes = F.normalize(model.tool_prototypes, dim=-1)
    matched = prototypes[labels]
    alignment = (1.0 - (embeddings * matched).sum(dim=-1)).mean()
    loss = cross_entropy + alignment_weight * alignment
    return loss, {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "alignment": float(alignment.detach().cpu()),
    }


@torch.inference_mode()
def embed_texts(
    model: SimpleToolEmbeddingModel,
    tokenizer,
    texts: Sequence[str],
    device: torch.device | str,
    max_length: int,
    batch_size: int = 32,
    progress_desc: str = "Embedding texts",
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, model.embedding_dim))

    model.eval()
    device = torch.device(device)
    chunks: List[torch.Tensor] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=progress_desc,
        unit="batch",
        leave=False,
    ):
        batch_texts = list(texts[start:start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        embeddings = model.encode(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        chunks.append(embeddings.cpu())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def compute_tool_centroids(
    model: SimpleToolEmbeddingModel,
    tokenizer,
    rows: Sequence[Dict[str, str]],
    tool_names: Sequence[str],
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
        progress_desc="Embedding for centroids",
    )
    tool_to_index = {tool: index for index, tool in enumerate(tool_names)}
    labels = torch.tensor([tool_to_index[row["tool"]] for row in rows], dtype=torch.long)

    centroids: List[torch.Tensor] = []
    for index in tqdm(
        range(len(tool_names)),
        desc="Averaging centroids",
        unit="tool",
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
def retrieval_accuracy(
    model: SimpleToolEmbeddingModel,
    tokenizer,
    rows: Sequence[Dict[str, str]],
    centroids: torch.Tensor,
    tool_names: Sequence[str],
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
        progress_desc="Embedding for retrieval",
    )
    scores = embeddings @ centroids.T
    predictions = scores.argmax(dim=-1)
    tool_to_index = {tool: index for index, tool in enumerate(tool_names)}
    labels = torch.tensor([tool_to_index[row["tool"]] for row in rows], dtype=torch.long)
    return float((predictions == labels).float().mean().item())


@torch.inference_mode()
def classification_accuracy(
    model: SimpleToolEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    total = 0
    correct = 0
    model.eval()
    for batch in tqdm(
        loader,
        desc="Validation classification",
        unit="batch",
        leave=False,
    ):
        labels = batch["labels"].to(device)
        encoded = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask"}
        }
        _, logits = model(**encoded)
        predictions = logits.argmax(dim=-1)
        total += labels.numel()
        correct += int((predictions == labels).sum().item())
    return correct / total if total else float("nan")


def save_checkpoint(
    path: Path,
    model: SimpleToolEmbeddingModel,
    tool_names: Sequence[str],
    centroids: torch.Tensor,
    args: argparse.Namespace,
    history: Sequence[Dict[str, Any]],
) -> None:
    payload = {
        "encoder_model": args.encoder_model,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "tool_names": list(tool_names),
        "tool_centroids": centroids.cpu(),
        "model_state_dict": model.state_dict(),
        "history": list(history),
        "training_args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint_bundle(checkpoint_path: str | Path, device: str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SimpleToolEmbeddingModel(
        encoder_model=checkpoint["encoder_model"],
        num_tools=len(checkpoint["tool_names"]),
        embedding_dim=checkpoint["embedding_dim"],
        dropout=checkpoint["dropout"],
        freeze_encoder=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = build_tokenizer(checkpoint["encoder_model"])
    return {
        "model": model,
        "tokenizer": tokenizer,
        "tool_names": checkpoint["tool_names"],
        "centroids": checkpoint["tool_centroids"].to(device),
        "max_length": checkpoint["max_length"],
        "history": checkpoint.get("history", []),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = clean_rows(load_jsonl(dataset_path))
    if len(rows) < 2:
        raise ValueError("Need at least two rows to train.")

    tool_names = sorted({row["tool"] for row in rows})
    tool_to_index = {tool: index for index, tool in enumerate(tool_names)}
    train_rows, val_rows = stratified_split(rows, args.val_ratio, args.seed)

    print(f"Loaded {len(rows)} rows from {dataset_path}")
    print(f"Tools: {len(tool_names)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")

    device = choose_device(args.device)
    tokenizer = build_tokenizer(args.encoder_model)
    collate_fn = build_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(
        ToolDataset(train_rows, tool_to_index),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ToolDataset(val_rows, tool_to_index),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SimpleToolEmbeddingModel(
        encoder_model=args.encoder_model,
        num_tools=len(tool_names),
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
    history: List[Dict[str, Any]] = []

    epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        running_batches = 0

        batch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch} train",
            unit="batch",
            leave=False,
        )
        for batch in batch_iterator:
            labels = batch["labels"].to(device)
            encoded = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }

            optimizer.zero_grad()
            embeddings, logits = model(**encoded)
            loss, _ = compute_loss(
                model=model,
                embeddings=embeddings,
                logits=logits,
                labels=labels,
                alignment_weight=args.alignment_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.detach().cpu())
            running_batches += 1
            batch_iterator.set_postfix(loss=running_loss / running_batches)

        train_centroids = compute_tool_centroids(
            model=model,
            tokenizer=tokenizer,
            rows=train_rows,
            tool_names=tool_names,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        train_retrieval = retrieval_accuracy(
            model=model,
            tokenizer=tokenizer,
            rows=train_rows,
            centroids=train_centroids,
            tool_names=tool_names,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        val_retrieval = retrieval_accuracy(
            model=model,
            tokenizer=tokenizer,
            rows=val_rows,
            centroids=train_centroids,
            tool_names=tool_names,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        val_classification = classification_accuracy(model, val_loader, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, running_batches),
            "train_retrieval_accuracy": train_retrieval,
            "val_retrieval_accuracy": val_retrieval,
            "val_classification_accuracy": val_classification,
        }
        history.append(epoch_metrics)
        epoch_iterator.set_postfix(
            train_loss=f"{epoch_metrics['train_loss']:.4f}",
            val_retrieval=(
                "nan"
                if math.isnan(epoch_metrics["val_retrieval_accuracy"])
                else f"{epoch_metrics['val_retrieval_accuracy']:.4f}"
            ),
        )
        tqdm.write(json.dumps(epoch_metrics, indent=2))

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            tool_names=tool_names,
            centroids=train_centroids,
            args=args,
            history=history,
        )

        score = val_retrieval if not math.isnan(val_retrieval) else train_retrieval
        if score > best_val_retrieval:
            best_val_retrieval = score
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                tool_names=tool_names,
                centroids=train_centroids,
                args=args,
                history=history,
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
