#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset_utils import clean_rows, load_dataset_rows, write_json
from training.train_embedding_space import embed_texts, load_checkpoint_bundle, normalize_loss_type
from training.wandb_diagnostics import compute_overlap_tables


DEFAULT_DATASET_GLOB = "data/*/tool_embedding_dataset_train.jsonl"
DEFAULT_OUTPUT_PATH = "analysis/tool_blur_summary.json"
LOSS_ALIASES = {
    "ce": "prototype_ce",
    "crossentropy": "prototype_ce",
    "cross_entropy": "prototype_ce",
    "cross-entropy": "prototype_ce",
    "functional margin": "functional_margin",
    "functional-margin": "functional_margin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic blur pairs from trained normal embedding spaces. "
            "By default this scans data/*/tool_embedding_dataset_train.jsonl, loads each "
            "dataset's normal/prototype_ce/best.pt and normal/functional_margin/best.pt, "
            "then writes one JSON summary with overlap deltas."
        )
    )
    parser.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help=f"Glob used to discover train JSONL files. Default: {DEFAULT_DATASET_GLOB}",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit train JSONL path to include. Can be passed multiple times. "
            "When omitted, --dataset-glob is used."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=f"Where to write the combined summary JSON. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--table-output-path",
        type=Path,
        default=None,
        help=(
            "Where to write the compact semantic blur table CSV. "
            "Defaults to <output-path stem>_table.csv."
        ),
    )
    parser.add_argument(
        "--checkpoint-root-name",
        default="output",
        help="Directory under each dataset folder that contains embedding checkpoints.",
    )
    parser.add_argument(
        "--architecture",
        default="normal",
        choices=("normal",),
        help="Embedding architecture to compare. Currently this script targets normal checkpoints.",
    )
    parser.add_argument(
        "--baseline-loss",
        default="prototype_ce",
        help="Baseline loss directory. Use prototype_ce for the normal cross-entropy model.",
    )
    parser.add_argument(
        "--comparison-loss",
        default="functional_margin",
        help="Comparison loss directory.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default="best.pt",
        help="Checkpoint filename loaded from each loss directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for embedding train rows: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding train queries.",
    )
    parser.add_argument(
        "--overlap-margin",
        type=float,
        default=0.03,
        help=(
            "A query from tool A overlaps tool B when score(B) is within this margin of "
            "score(A). This is the same cluster-overlap signal used by training diagnostics."
        ),
    )
    parser.add_argument(
        "--min-tool-samples",
        type=int,
        default=2,
        help="Minimum train rows per tool needed to include a pair in pair summaries.",
    )
    parser.add_argument(
        "--top-pairs-per-model",
        type=int,
        default=50,
        help="Number of highest-overlap pairs retained for each model in each dataset.",
    )
    parser.add_argument(
        "--top-deltas-per-dataset",
        type=int,
        default=75,
        help="Number of largest pair changes retained for each dataset.",
    )
    parser.add_argument(
        "--top-global-deltas",
        type=int,
        default=150,
        help="Number of largest pair changes retained across datasets.",
    )
    parser.add_argument(
        "--cluster-min-overlap",
        type=float,
        default=0.20,
        help="Minimum mutual overlap needed to connect two tools into a blur cluster.",
    )
    parser.add_argument(
        "--cluster-min-cosine",
        type=float,
        default=0.0,
        help="Optional minimum centroid cosine for cluster edges.",
    )
    parser.add_argument(
        "--max-clusters-per-model",
        type=int,
        default=25,
        help="Maximum connected blur clusters retained per model per dataset.",
    )
    parser.add_argument(
        "--examples-per-direction",
        type=int,
        default=3,
        help="Number of near-overlap train queries kept for each pair direction.",
    )
    parser.add_argument(
        "--include-full-pair-table",
        action="store_true",
        help="Keep all pair rows for both models. This can make the JSON much larger.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested ({device_arg}) but unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_dataset_paths(args: argparse.Namespace) -> List[Path]:
    paths = [path.expanduser().resolve() for path in args.dataset_path]
    if not paths:
        paths = [
            Path(path).resolve()
            for path in sorted(glob.glob(args.dataset_glob))
            if Path(path).is_file()
        ]

    seen: set[Path] = set()
    unique_paths: List[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def dataset_name_from_path(path: Path) -> str:
    return path.parent.name or path.stem


def checkpoint_path_for_dataset(
    dataset_path: Path,
    *,
    checkpoint_root_name: str,
    architecture: str,
    loss_name: str,
    checkpoint_filename: str,
) -> Path:
    return (
        dataset_path.parent
        / checkpoint_root_name
        / architecture
        / normalize_analysis_loss(loss_name)
        / checkpoint_filename
    )


def normalize_analysis_loss(loss_name: str) -> str:
    cleaned = str(loss_name).strip().lower()
    return normalize_loss_type(LOSS_ALIASES.get(cleaned, cleaned))


def safe_float(value: Any) -> float | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return as_float


def compact_text(value: Any, *, max_chars: int = 260) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def dataframe_records(dataframe: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_record in dataframe.to_dict(orient="records"):
        record: Dict[str, Any] = {}
        for key, value in raw_record.items():
            if isinstance(value, (np.floating, float)):
                record[key] = safe_float(value)
            elif isinstance(value, (np.bool_, bool)):
                record[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                record[key] = int(value)
            else:
                record[key] = value
        records.append(record)
    return records


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_rows_for_tool_names(
    rows: Sequence[Dict[str, Any]],
    tool_names: Sequence[str],
) -> List[Dict[str, Any]]:
    tool_set = set(tool_names)
    return [row for row in rows if row["tool"] in tool_set]


def validate_tool_alignment(
    rows: Sequence[Dict[str, Any]],
    tool_names: Sequence[str],
) -> Dict[str, Any]:
    row_tools = {row["tool"] for row in rows}
    checkpoint_tools = set(tool_names)
    missing_from_checkpoint = sorted(row_tools - checkpoint_tools)
    missing_from_rows = sorted(checkpoint_tools - row_tools)
    return {
        "missing_from_checkpoint": missing_from_checkpoint,
        "missing_from_rows": missing_from_rows,
    }


def near_examples(
    *,
    sample_records: Sequence[Dict[str, Any]],
    source_tool: str,
    rival_tool: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    candidates = [
        record
        for record in sample_records
        if record["tool"] == source_tool and record["closest_rival_label"] == rival_tool
    ]
    if not candidates:
        candidates = [
            record
            for record in sample_records
            if record["tool"] == source_tool
        ]
    candidates = sorted(candidates, key=lambda record: float(record["margin"]))
    examples: List[Dict[str, Any]] = []
    for record in candidates[:limit]:
        examples.append(
            {
                "query": compact_text(record["query"]),
                "predicted_tool": record["predicted_label"],
                "closest_rival_tool": record["closest_rival_label"],
                "true_score": safe_float(record["true_score"]),
                "closest_rival_score": safe_float(record["closest_rival_score"]),
                "margin": safe_float(record["margin"]),
                "misclassified": bool(record["misclassified"]),
                "near_overlap": bool(record["near_overlap"]),
            }
        )
    return examples


def enrich_pair_examples(
    pair: Dict[str, Any],
    *,
    sample_records: Sequence[Dict[str, Any]],
    examples_per_direction: int,
) -> Dict[str, Any]:
    enriched = dict(pair)
    label_a = str(pair["label_a"])
    label_b = str(pair["label_b"])
    enriched["a_near_b_examples"] = near_examples(
        sample_records=sample_records,
        source_tool=label_a,
        rival_tool=label_b,
        limit=examples_per_direction,
    )
    enriched["b_near_a_examples"] = near_examples(
        sample_records=sample_records,
        source_tool=label_b,
        rival_tool=label_a,
        limit=examples_per_direction,
    )
    return enriched


def compute_clusters(
    pair_records: Sequence[Dict[str, Any]],
    *,
    cluster_min_overlap: float,
    cluster_min_cosine: float,
    max_clusters: int,
) -> List[Dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_rows: List[Dict[str, Any]] = []
    for pair in pair_records:
        mutual_overlap = pair.get("mutual_overlap")
        centroid_cosine = pair.get("centroid_cosine")
        if mutual_overlap is None or centroid_cosine is None:
            continue
        if float(mutual_overlap) < cluster_min_overlap:
            continue
        if float(centroid_cosine) < cluster_min_cosine:
            continue
        first = str(pair["label_a"])
        second = str(pair["label_b"])
        adjacency[first].add(second)
        adjacency[second].add(first)
        edge_rows.append(pair)

    visited: set[str] = set()
    clusters: List[Dict[str, Any]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue: deque[str] = deque([start])
        visited.add(start)
        tools: List[str] = []
        while queue:
            node = queue.popleft()
            tools.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        if len(tools) < 2:
            continue

        tool_set = set(tools)
        cluster_pairs = [
            pair
            for pair in edge_rows
            if pair["label_a"] in tool_set and pair["label_b"] in tool_set
        ]
        cluster_pairs.sort(
            key=lambda pair: (
                safe_float(pair.get("mutual_overlap")) or -1.0,
                safe_float(pair.get("centroid_cosine")) or -1.0,
            ),
            reverse=True,
        )
        clusters.append(
            {
                "tools": sorted(tools),
                "tool_count": len(tools),
                "pair_count": len(cluster_pairs),
                "max_mutual_overlap": cluster_pairs[0]["mutual_overlap"] if cluster_pairs else None,
                "top_pairs": [
                    {
                        "label_a": pair["label_a"],
                        "label_b": pair["label_b"],
                        "mutual_overlap": pair["mutual_overlap"],
                        "centroid_cosine": pair["centroid_cosine"],
                        "a_to_b_overlap": pair["a_to_b_overlap"],
                        "b_to_a_overlap": pair["b_to_a_overlap"],
                    }
                    for pair in cluster_pairs[:10]
                ],
            }
        )

    clusters.sort(
        key=lambda cluster: (
            safe_float(cluster.get("max_mutual_overlap")) or -1.0,
            cluster["pair_count"],
            cluster["tool_count"],
        ),
        reverse=True,
    )
    return clusters[:max(max_clusters, 0)]


def summarize_overlap_model(
    *,
    checkpoint_path: Path,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    overlap_margin: float,
    min_tool_samples: int,
    top_pairs: int,
    cluster_min_overlap: float,
    cluster_min_cosine: float,
    max_clusters: int,
    examples_per_direction: int,
    include_full_pair_table: bool,
) -> Dict[str, Any]:
    bundle = load_checkpoint_bundle(checkpoint_path, device=str(device))
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    tool_names = list(bundle["tool_names"])
    centroids = F.normalize(bundle["centroids"].detach().to(device), dim=-1).cpu().numpy()
    usable_rows = normalize_rows_for_tool_names(rows, tool_names)
    tool_alignment = validate_tool_alignment(rows, tool_names)

    embeddings = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=[row["query"] for row in usable_rows],
        device=device,
        max_length=int(bundle["max_length"]),
        batch_size=batch_size,
        progress_desc=f"Embedding {checkpoint_path.parent.parent.name}/{checkpoint_path.parent.name}",
    )
    normalized_embeddings = F.normalize(embeddings, dim=-1).cpu().numpy()

    label_to_index = {tool_name: index for index, tool_name in enumerate(tool_names)}
    labels = np.asarray([label_to_index[row["tool"]] for row in usable_rows], dtype=np.int64)
    scores = normalized_embeddings @ centroids.T
    predictions = scores.argmax(axis=1)

    sample_df, label_df, pair_df, _, _ = compute_overlap_tables(
        rows=usable_rows,
        row_label_names=[row["tool"] for row in usable_rows],
        scores=scores,
        label_indices=labels,
        predictions=predictions,
        label_names=tool_names,
        normalized_centroids=centroids,
        overlap_margin=overlap_margin,
        min_label_samples=min_tool_samples,
        label_column="tool",
    )

    sample_records = dataframe_records(sample_df)
    label_records = dataframe_records(label_df)
    pair_records = dataframe_records(pair_df)
    retained_pair_records = pair_records if include_full_pair_table else pair_records[: max(top_pairs, 0)]
    retained_pair_records = [
        enrich_pair_examples(
            pair,
            sample_records=sample_records,
            examples_per_direction=examples_per_direction,
        )
        for pair in retained_pair_records
    ]
    clusters = compute_clusters(
        pair_records,
        cluster_min_overlap=cluster_min_overlap,
        cluster_min_cosine=cluster_min_cosine,
        max_clusters=max_clusters,
    )

    accuracy = float((predictions == labels).mean()) if len(labels) else float("nan")
    semantic_blur_rate = (
        float(sample_df["near_overlap"].mean())
        if len(sample_df)
        else float("nan")
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "architecture": bundle.get("architecture"),
        "loss_name": bundle.get("loss_name"),
        "encoder_model": getattr(model, "encoder_model", None),
        "row_count": len(rows),
        "usable_row_count": len(usable_rows),
        "tool_count": len(tool_names),
        "accuracy_on_train_rows": safe_float(accuracy),
        "semantic_blur_rate": safe_float(semantic_blur_rate),
        "semantic_blur_percent": safe_float(semantic_blur_rate * 100.0),
        "tool_alignment": tool_alignment,
        "top_tools_by_overlap": label_records[:25],
        "top_pairs": retained_pair_records,
        "clusters": clusters,
        "_all_pair_records": pair_records,
    }


def index_pairs(pair_records: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    indexed: Dict[tuple[str, str], Dict[str, Any]] = {}
    for pair in pair_records:
        key = (str(pair["label_a"]), str(pair["label_b"]))
        indexed[key] = pair
    return indexed


def compute_pair_deltas(
    *,
    baseline_pairs: Sequence[Dict[str, Any]],
    comparison_pairs: Sequence[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    baseline_by_pair = index_pairs(baseline_pairs)
    comparison_by_pair = index_pairs(comparison_pairs)
    all_keys = sorted(set(baseline_by_pair) | set(comparison_by_pair))

    deltas: List[Dict[str, Any]] = []
    for key in all_keys:
        baseline = baseline_by_pair.get(key, {})
        comparison = comparison_by_pair.get(key, {})
        base_overlap = safe_float(baseline.get("mutual_overlap")) or 0.0
        comp_overlap = safe_float(comparison.get("mutual_overlap")) or 0.0
        base_cosine = safe_float(baseline.get("centroid_cosine")) or 0.0
        comp_cosine = safe_float(comparison.get("centroid_cosine")) or 0.0
        base_confusion = max(
            safe_float(baseline.get("a_to_b_confusion")) or 0.0,
            safe_float(baseline.get("b_to_a_confusion")) or 0.0,
        )
        comp_confusion = max(
            safe_float(comparison.get("a_to_b_confusion")) or 0.0,
            safe_float(comparison.get("b_to_a_confusion")) or 0.0,
        )
        deltas.append(
            {
                "label_a": key[0],
                "label_b": key[1],
                "baseline_mutual_overlap": safe_float(base_overlap),
                "comparison_mutual_overlap": safe_float(comp_overlap),
                "delta_mutual_overlap": safe_float(comp_overlap - base_overlap),
                "baseline_centroid_cosine": safe_float(base_cosine),
                "comparison_centroid_cosine": safe_float(comp_cosine),
                "delta_centroid_cosine": safe_float(comp_cosine - base_cosine),
                "baseline_max_confusion": safe_float(base_confusion),
                "comparison_max_confusion": safe_float(comp_confusion),
                "delta_max_confusion": safe_float(comp_confusion - base_confusion),
            }
        )

    deltas.sort(
        key=lambda row: (
            abs(row["delta_mutual_overlap"] or 0.0),
            abs(row["delta_centroid_cosine"] or 0.0),
            abs(row["delta_max_confusion"] or 0.0),
        ),
        reverse=True,
    )
    return deltas[: max(top_n, 0)]


def split_delta_directions(delta_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    increased = sorted(
        [row for row in delta_rows if (row["delta_mutual_overlap"] or 0.0) > 0],
        key=lambda row: row["delta_mutual_overlap"] or 0.0,
        reverse=True,
    )
    decreased = sorted(
        [row for row in delta_rows if (row["delta_mutual_overlap"] or 0.0) < 0],
        key=lambda row: row["delta_mutual_overlap"] or 0.0,
    )
    return {
        "largest_overlap_increases": increased,
        "largest_overlap_decreases": decreased,
    }


def remove_private_tables(summary: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(summary)
    public.pop("_all_pair_records", None)
    return public


def build_semantic_blur_table(dataset_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []
    for dataset_summary in dataset_summaries:
        if dataset_summary.get("status") != "ok":
            continue
        for section_name in ("baseline", "comparison"):
            model_summary = dataset_summary[section_name]
            table_rows.append(
                {
                    "dataset": dataset_summary["dataset"],
                    "loss_type": model_summary["loss_name"],
                    "total_tools": model_summary["tool_count"],
                    "semantic_blur_percent": round(
                        float(model_summary["semantic_blur_percent"]),
                        4,
                    )
                    if model_summary.get("semantic_blur_percent") is not None
                    else None,
                }
            )
    return table_rows


def analyze_dataset(path: Path, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    dataset_name = dataset_name_from_path(path)
    baseline_loss = normalize_analysis_loss(args.baseline_loss)
    comparison_loss = normalize_analysis_loss(args.comparison_loss)
    baseline_checkpoint = checkpoint_path_for_dataset(
        path,
        checkpoint_root_name=args.checkpoint_root_name,
        architecture=args.architecture,
        loss_name=baseline_loss,
        checkpoint_filename=args.checkpoint_filename,
    )
    comparison_checkpoint = checkpoint_path_for_dataset(
        path,
        checkpoint_root_name=args.checkpoint_root_name,
        architecture=args.architecture,
        loss_name=comparison_loss,
        checkpoint_filename=args.checkpoint_filename,
    )

    missing_checkpoints = [
        str(checkpoint)
        for checkpoint in (baseline_checkpoint, comparison_checkpoint)
        if not checkpoint.exists()
    ]
    if missing_checkpoints:
        return {
            "dataset": dataset_name,
            "dataset_path": str(path),
            "status": "skipped",
            "reason": "missing_checkpoints",
            "missing_checkpoints": missing_checkpoints,
        }

    rows = clean_rows(load_dataset_rows(path))
    if not rows:
        return {
            "dataset": dataset_name,
            "dataset_path": str(path),
            "status": "skipped",
            "reason": "no_usable_rows",
        }

    print(f"[{dataset_name}] loading {baseline_loss}: {baseline_checkpoint}")
    baseline_summary = summarize_overlap_model(
        checkpoint_path=baseline_checkpoint,
        rows=rows,
        device=device,
        batch_size=args.batch_size,
        overlap_margin=args.overlap_margin,
        min_tool_samples=args.min_tool_samples,
        top_pairs=args.top_pairs_per_model,
        cluster_min_overlap=args.cluster_min_overlap,
        cluster_min_cosine=args.cluster_min_cosine,
        max_clusters=args.max_clusters_per_model,
        examples_per_direction=args.examples_per_direction,
        include_full_pair_table=args.include_full_pair_table,
    )

    print(f"[{dataset_name}] loading {comparison_loss}: {comparison_checkpoint}")
    comparison_summary = summarize_overlap_model(
        checkpoint_path=comparison_checkpoint,
        rows=rows,
        device=device,
        batch_size=args.batch_size,
        overlap_margin=args.overlap_margin,
        min_tool_samples=args.min_tool_samples,
        top_pairs=args.top_pairs_per_model,
        cluster_min_overlap=args.cluster_min_overlap,
        cluster_min_cosine=args.cluster_min_cosine,
        max_clusters=args.max_clusters_per_model,
        examples_per_direction=args.examples_per_direction,
        include_full_pair_table=args.include_full_pair_table,
    )

    delta_rows = compute_pair_deltas(
        baseline_pairs=baseline_summary["_all_pair_records"],
        comparison_pairs=comparison_summary["_all_pair_records"],
        top_n=args.top_deltas_per_dataset,
    )
    delta_directions = split_delta_directions(delta_rows)

    return {
        "dataset": dataset_name,
        "dataset_path": str(path),
        "status": "ok",
        "row_count": len(rows),
        "baseline_loss": baseline_loss,
        "comparison_loss": comparison_loss,
        "baseline": remove_private_tables(baseline_summary),
        "comparison": remove_private_tables(comparison_summary),
        "pair_deltas": delta_rows,
        **delta_directions,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset_paths = resolve_dataset_paths(args)
    if not dataset_paths:
        raise SystemExit(f"No train files found for glob: {args.dataset_glob}")

    dataset_summaries: List[Dict[str, Any]] = []
    global_deltas: List[Dict[str, Any]] = []
    for dataset_path in dataset_paths:
        print(f"Analyzing dataset train split: {dataset_path}")
        dataset_summary = analyze_dataset(dataset_path, args, device)
        dataset_summaries.append(dataset_summary)
        if dataset_summary.get("status") == "ok":
            for delta in dataset_summary.get("pair_deltas", []):
                global_deltas.append(
                    {
                        "dataset": dataset_summary["dataset"],
                        "dataset_path": dataset_summary["dataset_path"],
                        **delta,
                    }
                )

    global_deltas.sort(
        key=lambda row: (
            abs(row["delta_mutual_overlap"] or 0.0),
            abs(row["delta_centroid_cosine"] or 0.0),
            abs(row["delta_max_confusion"] or 0.0),
        ),
        reverse=True,
    )
    global_direction_rows = split_delta_directions(global_deltas)
    semantic_blur_table = build_semantic_blur_table(dataset_summaries)
    output_path = args.output_path.expanduser().resolve()
    table_output_path = (
        args.table_output_path.expanduser().resolve()
        if args.table_output_path is not None
        else output_path.with_name(f"{output_path.stem}_table.csv")
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": "checkpoint_cluster_overlap_delta",
            "description": (
                "For each dataset, embed train queries with the trained normal baseline "
                "checkpoint and the trained normal functional-margin checkpoint. Compute "
                "cluster overlap as the share of samples whose rival centroid score is "
                "within overlap_margin of the true tool centroid score, then compare pair "
                "overlap values between checkpoints."
            ),
            "baseline_loss": normalize_analysis_loss(args.baseline_loss),
            "comparison_loss": normalize_analysis_loss(args.comparison_loss),
            "delta_definition": "comparison - baseline; negative delta_mutual_overlap means functional_margin reduced overlap.",
        },
        "settings": {
            "dataset_glob": args.dataset_glob,
            "dataset_paths": [str(path) for path in dataset_paths],
            "checkpoint_root_name": args.checkpoint_root_name,
            "architecture": args.architecture,
            "checkpoint_filename": args.checkpoint_filename,
            "table_output_path": str(table_output_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "overlap_margin": args.overlap_margin,
            "min_tool_samples": args.min_tool_samples,
            "top_pairs_per_model": args.top_pairs_per_model,
            "top_deltas_per_dataset": args.top_deltas_per_dataset,
            "top_global_deltas": args.top_global_deltas,
            "cluster_min_overlap": args.cluster_min_overlap,
            "cluster_min_cosine": args.cluster_min_cosine,
            "max_clusters_per_model": args.max_clusters_per_model,
            "examples_per_direction": args.examples_per_direction,
            "include_full_pair_table": args.include_full_pair_table,
        },
        "dataset_count": len(dataset_summaries),
        "ok_dataset_count": sum(1 for item in dataset_summaries if item.get("status") == "ok"),
        "skipped_dataset_count": sum(1 for item in dataset_summaries if item.get("status") != "ok"),
        "semantic_blur_table": semantic_blur_table,
        "datasets": dataset_summaries,
        "top_pair_deltas_global": global_deltas[: max(args.top_global_deltas, 0)],
        "largest_overlap_increases_global": global_direction_rows["largest_overlap_increases"][
            : max(args.top_global_deltas, 0)
        ],
        "largest_overlap_decreases_global": global_direction_rows["largest_overlap_decreases"][
            : max(args.top_global_deltas, 0)
        ],
    }

    write_json(output_path, payload)
    write_csv(
        table_output_path,
        semantic_blur_table,
        fieldnames=("dataset", "loss_type", "total_tools", "semantic_blur_percent"),
    )
    print(f"Wrote blur comparison summary to {output_path}")
    print(f"Wrote semantic blur table to {table_output_path}")


if __name__ == "__main__":
    main()
