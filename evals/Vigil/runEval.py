"""Evaluate all Vigil embedding-space variants on the benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import embed_texts, load_checkpoint_bundle


DATA_DIR = REPO_ROOT / "data" / "Vigil"
DEFAULT_BENCHMARK_PATH = DATA_DIR / "benchmark.json"
DEFAULT_TOOLS_PATH = DATA_DIR / "tools.json"
DEFAULT_CHECKPOINT_ROOT = DATA_DIR / "output"
DEFAULT_OUTPUT_PATH = DEFAULT_CHECKPOINT_ROOT / "eval" / "eval_summary.json"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    architecture: str
    loss_name: str
    checkpoint_path: Path
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved Vigil embedding-space checkpoint variant against the "
            "Vigil benchmark and write a combined JSON summary."
        )
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the Vigil benchmark.json file.",
    )
    parser.add_argument(
        "--tools-path",
        type=Path,
        default=DEFAULT_TOOLS_PATH,
        help="Path to the Vigil tools.json file.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root directory containing saved embedding-space checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default="best.pt",
        help="Checkpoint filename to evaluate for each variant.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the combined eval summary JSON.",
    )
    parser.add_argument(
        "--embed-device",
        default="cuda:5",
        help='Torch device for embedding inference. Uses CUDA when available if set to "auto".',
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding benchmark inputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top-ranked predictions to store per scenario.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_runtime_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda:5" if torch.cuda.is_available() else "cpu"
    return device_name


def round_float(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round_float(sum(values) / len(values))


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def first_numeric(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def max_numeric(history: list[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in history:
        value = first_numeric(row, *keys)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return max(values)


def safe_metric_sort_value(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def read_training_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        return []

    payload = load_json(metrics_path)
    if not isinstance(payload, list):
        return []

    return [row for row in payload if isinstance(row, dict)]


def summarize_training_history(
    history: list[dict[str, Any]],
    architecture: str,
) -> dict[str, Any] | None:
    if not history:
        return None

    last_row = history[-1]
    summary: dict[str, Any] = {
        "epochs": len(history),
        "best_epoch": (
            int(last_row["best_epoch"])
            if isinstance(last_row.get("best_epoch"), int)
            else None
        ),
    }

    if architecture == "hierarchical":
        summary.update(
            {
                "best_val_tool_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_tool_retrieval_accuracy",
                        "val_tool_retrieval_accuracy",
                    )
                ),
                "final_val_tool_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_tool_retrieval_accuracy")
                ),
                "best_val_parent_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_parent_retrieval_accuracy",
                        "val_parent_retrieval_accuracy",
                    )
                ),
                "final_val_parent_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_parent_retrieval_accuracy")
                ),
            }
        )
    else:
        summary.update(
            {
                "best_val_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_retrieval_accuracy",
                        "val_retrieval_accuracy",
                    )
                ),
                "final_val_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_retrieval_accuracy")
                ),
                "final_val_classification_accuracy": round_float(
                    first_numeric(last_row, "val_classification_accuracy")
                ),
            }
        )

    return summary


def discover_variants(
    checkpoint_root: Path,
    checkpoint_filename: str,
) -> list[VariantSpec]:
    pattern = f"*/*/{checkpoint_filename}"
    variants: list[VariantSpec] = []

    for checkpoint_path in sorted(checkpoint_root.glob(pattern)):
        if not checkpoint_path.is_file():
            continue

        architecture = checkpoint_path.parent.parent.name
        loss_name = checkpoint_path.parent.name
        variant_id = f"{architecture}/{loss_name}"
        variants.append(
            VariantSpec(
                variant_id=variant_id,
                architecture=architecture,
                loss_name=loss_name,
                checkpoint_path=checkpoint_path,
                metrics_path=checkpoint_path.parent / "metrics.json",
            )
        )

    return variants


def evaluate_ranked_prediction(
    ranked_tools: list[str],
    expected_tools: list[str],
) -> dict[str, Any]:
    expected_unique = unique_in_order(expected_tools)
    if not expected_unique:
        return {
            "top_1_hit": None,
            "top_3_hit": None,
            "top_5_hit": None,
            "all_expected_in_top_3": None,
            "all_expected_in_top_5": None,
            "expected_recall_at_1": None,
            "expected_recall_at_3": None,
            "expected_recall_at_5": None,
            "expected_tool_rank": None,
            "reciprocal_rank": None,
        }

    expected_set = set(expected_unique)
    expected_tool_rank: int | None = None
    for index, tool_name in enumerate(ranked_tools, start=1):
        if tool_name in expected_set:
            expected_tool_rank = index
            break

    def any_hit_at(k: int) -> bool:
        limit = min(k, len(ranked_tools))
        return any(tool in expected_set for tool in ranked_tools[:limit])

    def all_expected_at(k: int) -> bool:
        limit = min(k, len(ranked_tools))
        return set(expected_unique).issubset(set(ranked_tools[:limit]))

    def recall_at(k: int) -> float:
        limit = min(k, len(ranked_tools))
        top_tools = set(ranked_tools[:limit])
        hits = sum(1 for tool in expected_unique if tool in top_tools)
        return hits / len(expected_unique)

    reciprocal_rank = 0.0 if expected_tool_rank is None else 1.0 / float(expected_tool_rank)

    return {
        "top_1_hit": any_hit_at(1),
        "top_3_hit": any_hit_at(3),
        "top_5_hit": any_hit_at(5),
        "all_expected_in_top_3": all_expected_at(3),
        "all_expected_in_top_5": all_expected_at(5),
        "expected_recall_at_1": round_float(recall_at(1)),
        "expected_recall_at_3": round_float(recall_at(3)),
        "expected_recall_at_5": round_float(recall_at(5)),
        "expected_tool_rank": expected_tool_rank,
        "reciprocal_rank": round_float(reciprocal_rank),
    }


def build_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def collect_bool(field_name: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row["evaluation"].get(field_name)
            if value is not None:
                values.append(1.0 if value else 0.0)
        return values

    def collect_number(field_name: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row["evaluation"].get(field_name)
            if value is not None:
                values.append(float(value))
        return values

    return {
        "scenario_count": len(rows),
        "top_1_accuracy": mean_or_none(collect_bool("top_1_hit")),
        "top_3_accuracy": mean_or_none(collect_bool("top_3_hit")),
        "top_5_accuracy": mean_or_none(collect_bool("top_5_hit")),
        "all_expected_in_top_3_rate": mean_or_none(collect_bool("all_expected_in_top_3")),
        "all_expected_in_top_5_rate": mean_or_none(collect_bool("all_expected_in_top_5")),
        "mean_expected_recall_at_1": mean_or_none(collect_number("expected_recall_at_1")),
        "mean_expected_recall_at_3": mean_or_none(collect_number("expected_recall_at_3")),
        "mean_expected_recall_at_5": mean_or_none(collect_number("expected_recall_at_5")),
        "mean_reciprocal_rank": mean_or_none(collect_number("reciprocal_rank")),
    }


def build_aggregate_metrics(scenario_reports: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in scenario_reports if row.get("expected_tools")]
    single_tool = [row for row in evaluable if len(row["expected_tools"]) == 1]
    multi_tool = [row for row in evaluable if len(row["expected_tools"]) > 1]

    titles = sorted({str(row["title"]) for row in evaluable if row.get("title")})
    expected_tool_names = sorted(
        {tool for row in evaluable for tool in row.get("expected_tools", [])}
    )

    return {
        "overall": build_group_metrics(evaluable),
        "single_tool": build_group_metrics(single_tool),
        "multi_tool": build_group_metrics(multi_tool),
        "by_title": {
            title: build_group_metrics(
                [row for row in evaluable if str(row.get("title")) == title]
            )
            for title in titles
        },
        "by_expected_tool": {
            tool_name: build_group_metrics(
                [row for row in evaluable if tool_name in row.get("expected_tools", [])]
            )
            for tool_name in expected_tool_names
        },
    }


def evaluate_variant(
    variant: VariantSpec,
    scenarios: list[dict[str, Any]],
    tool_registry: dict[str, dict[str, Any]],
    benchmark_tool_names: set[str],
    embed_device: str,
    embedding_batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    history = read_training_history(variant.metrics_path)
    training_summary = summarize_training_history(history, variant.architecture)

    warnings: list[str] = []
    try:
        bundle = load_checkpoint_bundle(variant.checkpoint_path, device=embed_device)
        tool_names = [str(tool_name) for tool_name in bundle["tool_names"]]
        tool_name_set = set(tool_names)
        missing_benchmark_tools = sorted(benchmark_tool_names - tool_name_set)
        if missing_benchmark_tools:
            warnings.append(
                "Benchmark tools missing from checkpoint centroids: "
                + ", ".join(missing_benchmark_tools)
            )

        tools_without_registry_entry = sorted(tool_name_set - set(tool_registry))
        if tools_without_registry_entry:
            warnings.append(
                "Checkpoint tools missing from tools.json: "
                + ", ".join(tools_without_registry_entry)
            )

        centroids = F.normalize(bundle["centroids"].to(embed_device), dim=-1)
        texts = [str(scenario.get("input", "")) for scenario in scenarios]
        embeddings = embed_texts(
            model=bundle["model"],
            tokenizer=bundle["tokenizer"],
            texts=texts,
            device=embed_device,
            max_length=int(bundle["max_length"]),
            batch_size=min(len(texts), embedding_batch_size) if texts else embedding_batch_size,
            progress_desc=f"Embedding Vigil benchmark for {variant.variant_id}",
        )
        embeddings = F.normalize(embeddings.to(centroids.device), dim=-1)
        score_matrix = embeddings @ centroids.T

        scenario_reports: list[dict[str, Any]] = []
        candidate_limit = max(1, min(top_k, len(tool_names)))
        for scenario, scores in zip(scenarios, score_matrix, strict=True):
            expected_tools = [str(tool_name) for tool_name in scenario.get("expected_tools", [])]
            ranked_indices = torch.argsort(scores, descending=True).tolist()
            ranked_tools = [tool_names[index] for index in ranked_indices]
            ranked_scores = [float(scores[index]) for index in ranked_indices]

            evaluation = evaluate_ranked_prediction(ranked_tools, expected_tools)
            top_predictions = [
                {
                    "tool": ranked_tools[index],
                    "score": round_float(ranked_scores[index]),
                }
                for index in range(candidate_limit)
            ]

            row_warnings: list[str] = []
            missing_expected_tools = sorted(set(expected_tools) - tool_name_set)
            if missing_expected_tools:
                row_warnings.append(
                    "Expected tools missing from checkpoint centroids: "
                    + ", ".join(missing_expected_tools)
                )

            top_score = ranked_scores[0] if ranked_scores else None
            second_score = ranked_scores[1] if len(ranked_scores) > 1 else None
            score_margin = (
                None
                if top_score is None or second_score is None
                else round_float(top_score - second_score)
            )

            scenario_reports.append(
                {
                    "id": scenario.get("id"),
                    "category": scenario.get("category"),
                    "title": scenario.get("title"),
                    "input": str(scenario.get("input", "")),
                    "expected_tools": expected_tools,
                    "status": "ok",
                    "warnings": row_warnings,
                    "predicted_tool": ranked_tools[0] if ranked_tools else None,
                    "retrieval": {
                        "candidates": top_predictions,
                        "ranked_tools": ranked_tools,
                        "expected_tool_rank": evaluation["expected_tool_rank"],
                        "score_margin_top1_top2": score_margin,
                    },
                    "evaluation": evaluation,
                }
            )

        aggregate_metrics = build_aggregate_metrics(scenario_reports)
        success_count = sum(1 for row in scenario_reports if row["status"] == "ok")
        error_count = len(scenario_reports) - success_count

        return {
            "variant_id": variant.variant_id,
            "architecture": variant.architecture,
            "loss_name": variant.loss_name,
            "status": "ok",
            "warnings": warnings,
            "error_message": None,
            "error_traceback": None,
            "paths": {
                "checkpoint_path": str(variant.checkpoint_path.resolve()),
                "metrics_path": str(variant.metrics_path.resolve()),
            },
            "runtime": {
                "embed_device": embed_device,
                "embedding_batch_size": embedding_batch_size,
                "top_k": top_k,
                "tool_count": len(tool_names),
                "embed_max_length": int(bundle["max_length"]),
                "encoder_model": getattr(bundle["model"], "encoder_model", None),
                "checkpoint_architecture": bundle.get("architecture"),
                "checkpoint_loss_name": bundle.get("loss_name"),
            },
            "training": training_summary,
            "counts": {
                "total_scenarios": len(scenario_reports),
                "successful_scenarios": success_count,
                "error_scenarios": error_count,
                "evaluable_scenarios": sum(
                    1 for row in scenario_reports if row.get("expected_tools")
                ),
                "single_tool_scenarios": sum(
                    1 for row in scenario_reports if len(row.get("expected_tools", [])) == 1
                ),
                "multi_tool_scenarios": sum(
                    1 for row in scenario_reports if len(row.get("expected_tools", [])) > 1
                ),
            },
            "metrics": aggregate_metrics,
            "tool_names": tool_names,
            "scenarios": scenario_reports,
        }
    except Exception as exc:
        return {
            "variant_id": variant.variant_id,
            "architecture": variant.architecture,
            "loss_name": variant.loss_name,
            "status": "error",
            "warnings": warnings,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=5),
            "paths": {
                "checkpoint_path": str(variant.checkpoint_path.resolve()),
                "metrics_path": str(variant.metrics_path.resolve()),
            },
            "runtime": {
                "embed_device": embed_device,
                "embedding_batch_size": embedding_batch_size,
                "top_k": top_k,
            },
            "training": training_summary,
            "counts": {
                "total_scenarios": len(scenarios),
                "successful_scenarios": 0,
                "error_scenarios": len(scenarios),
                "evaluable_scenarios": sum(
                    1 for scenario in scenarios if scenario.get("expected_tools")
                ),
                "single_tool_scenarios": sum(
                    1 for scenario in scenarios if len(scenario.get("expected_tools", [])) == 1
                ),
                "multi_tool_scenarios": sum(
                    1 for scenario in scenarios if len(scenario.get("expected_tools", [])) > 1
                ),
            },
            "metrics": None,
            "tool_names": [],
            "scenarios": [],
        }


def build_leaderboard(variant_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for variant in variant_summaries:
        if variant.get("status") != "ok":
            continue

        overall = variant["metrics"]["overall"]
        training = variant.get("training") or {}
        entries.append(
            {
                "variant_id": variant["variant_id"],
                "architecture": variant["architecture"],
                "loss_name": variant["loss_name"],
                "top_1_accuracy": overall["top_1_accuracy"],
                "top_3_accuracy": overall["top_3_accuracy"],
                "top_5_accuracy": overall["top_5_accuracy"],
                "mean_reciprocal_rank": overall["mean_reciprocal_rank"],
                "best_epoch": training.get("best_epoch"),
                "best_val_retrieval_accuracy": training.get(
                    "best_val_retrieval_accuracy",
                    training.get("best_val_tool_retrieval_accuracy"),
                ),
            }
        )

    entries.sort(
        key=lambda row: (
            -safe_metric_sort_value(row["top_1_accuracy"]),
            -safe_metric_sort_value(row["mean_reciprocal_rank"]),
            -safe_metric_sort_value(row["top_3_accuracy"]),
            row["variant_id"],
        )
    )

    for index, row in enumerate(entries, start=1):
        row["rank"] = index
    return entries


def build_cross_variant_metrics(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    if not leaderboard:
        return {
            "best_variant_by_top_1_accuracy": None,
            "best_top_1_accuracy": None,
            "best_variant_by_mean_reciprocal_rank": None,
            "best_mean_reciprocal_rank": None,
        }

    best_by_top_1 = max(
        leaderboard,
        key=lambda row: (
            safe_metric_sort_value(row["top_1_accuracy"]),
            safe_metric_sort_value(row["mean_reciprocal_rank"]),
        ),
    )
    best_by_mrr = max(
        leaderboard,
        key=lambda row: (
            safe_metric_sort_value(row["mean_reciprocal_rank"]),
            safe_metric_sort_value(row["top_1_accuracy"]),
        ),
    )

    return {
        "best_variant_by_top_1_accuracy": best_by_top_1["variant_id"],
        "best_top_1_accuracy": best_by_top_1["top_1_accuracy"],
        "best_variant_by_mean_reciprocal_rank": best_by_mrr["variant_id"],
        "best_mean_reciprocal_rank": best_by_mrr["mean_reciprocal_rank"],
    }


def evaluate_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_payload = load_json(args.benchmark_path)
    scenarios = benchmark_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError(f"Expected {args.benchmark_path} to contain a 'scenarios' list.")

    tools_payload = load_json(args.tools_path)
    tool_registry_list = tools_payload.get("tools", [])
    if not isinstance(tool_registry_list, list):
        raise ValueError(f"Expected {args.tools_path} to contain a 'tools' list.")

    tool_registry = {
        str(tool["name"]): tool
        for tool in tool_registry_list
        if isinstance(tool, dict) and "name" in tool
    }
    benchmark_tool_names = {
        str(tool_name)
        for scenario in scenarios
        for tool_name in scenario.get("expected_tools", [])
    }

    variants = discover_variants(args.checkpoint_root, args.checkpoint_filename)
    if not variants:
        raise FileNotFoundError(
            f"No checkpoint variants matching */*/{args.checkpoint_filename} found under "
            f"{args.checkpoint_root}."
        )

    embed_device = resolve_runtime_device(args.embed_device)
    variant_summaries = [
        evaluate_variant(
            variant=variant,
            scenarios=scenarios,
            tool_registry=tool_registry,
            benchmark_tool_names=benchmark_tool_names,
            embed_device=embed_device,
            embedding_batch_size=args.embedding_batch_size,
            top_k=args.top_k,
        )
        for variant in variants
    ]
    leaderboard = build_leaderboard(variant_summaries)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "Vigil",
        "mode": "embedding_space_variant_sweep",
        "description": (
            "Evaluates every saved Vigil embedding-space variant by embedding each benchmark "
            "input, ranking tool centroids, and scoring retrieval against the expected tool labels."
        ),
        "paths": {
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "benchmark_path": str(args.benchmark_path.resolve()),
            "tools_path": str(args.tools_path.resolve()),
            "output_path": str(args.output_path.resolve()),
        },
        "runtime": {
            "embed_device": embed_device,
            "embedding_batch_size": args.embedding_batch_size,
            "top_k": args.top_k,
            "checkpoint_filename": args.checkpoint_filename,
            "variant_count": len(variants),
            "benchmark_tool_count": len(benchmark_tool_names),
        },
        "counts": {
            "total_variants": len(variant_summaries),
            "successful_variants": sum(
                1 for variant in variant_summaries if variant.get("status") == "ok"
            ),
            "error_variants": sum(
                1 for variant in variant_summaries if variant.get("status") != "ok"
            ),
            "total_scenarios": len(scenarios),
            "evaluable_scenarios": sum(
                1 for scenario in scenarios if scenario.get("expected_tools")
            ),
            "single_tool_scenarios": sum(
                1 for scenario in scenarios if len(scenario.get("expected_tools", [])) == 1
            ),
            "multi_tool_scenarios": sum(
                1 for scenario in scenarios if len(scenario.get("expected_tools", [])) > 1
            ),
        },
        "metrics": build_cross_variant_metrics(leaderboard),
        "leaderboard": leaderboard,
        "tool_names": sorted(tool_registry),
        "variants": variant_summaries,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_benchmark(args)
    best_variant = summary["metrics"]["best_variant_by_top_1_accuracy"]
    best_top_1 = summary["metrics"]["best_top_1_accuracy"]
    best_mrr = summary["metrics"]["best_mean_reciprocal_rank"]
    print(f"Wrote eval summary to {args.output_path}")
    print(
        "Best variant: "
        f"{best_variant}, "
        f"top_1_accuracy={best_top_1}, "
        f"mean_reciprocal_rank={best_mrr}"
    )


if __name__ == "__main__":
    main()
