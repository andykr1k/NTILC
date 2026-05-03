from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters import EmbeddingSelectionAdapter
from benchmark.common import (
    DEFAULT_RANKING_LIMIT,
    EmbeddingVariantSpec,
    build_benchmark_summary,
    load_benchmark_rows,
    load_tool_catalog,
    now_utc_iso,
    slugify,
    write_json,
    write_jsonl,
)

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "OSS" / "tool_embedding_dataset_test.jsonl"
DEFAULT_TOOLS_PATH = REPO_ROOT / "data" / "OSS" / "tools.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "data" / "OSS" / "output" / "normal" / "functional_margin" / "best.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one embedding checkpoint and update an aggregate sweep leaderboard."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--tools-path", type=Path, default=DEFAULT_TOOLS_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset directory name used under output-root. Inferred from --dataset-path when omitted.",
    )
    parser.add_argument("--architecture", default="normal")
    parser.add_argument("--loss-name", default="functional_margin")
    parser.add_argument(
        "--variant-name",
        default="",
        help="Optional checkpoint variant, for example compatibility_weight_0_1.",
    )
    parser.add_argument(
        "--compatibility-weight",
        default="",
        help="Optional compatibility weight recorded in the model summary metadata.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on benchmark examples for quick debug runs.")
    parser.add_argument("--ranking-limit", type=int, default=DEFAULT_RANKING_LIMIT)
    parser.add_argument("--embedding-device", default="auto")
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def infer_dataset_name(dataset_path: Path, dataset_name: str) -> str:
    requested = str(dataset_name).strip()
    if requested:
        return requested

    parts = dataset_path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "data" and index + 1 < len(parts):
            return parts[index + 1]
    return dataset_path.parent.name


def infer_variant_name(checkpoint_path: Path, loss_name: str, variant_name: str) -> str:
    requested = str(variant_name).strip()
    if requested:
        return requested
    parent_name = checkpoint_path.parent.name
    return "" if parent_name == loss_name else parent_name


def build_variant_id(architecture: str, loss_name: str, variant_name: str) -> str:
    base = f"{architecture}/{loss_name}"
    return f"{base}/{variant_name}" if variant_name else base


def parse_compatibility_weight(value: str) -> float | str | None:
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped


def load_model_summaries(model_summary_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(model_summary_dir.glob("*.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {summary_path} to contain a JSON object.")
        summaries.append(payload)
    return sorted(summaries, key=model_summary_sort_key)


def model_summary_sort_key(summary: dict[str, Any]) -> tuple[int, float | str, str]:
    metadata = summary.get("metadata", {})
    weight = metadata.get("compatibility_weight") if isinstance(metadata, dict) else None
    adapter_id = str(summary.get("adapter_id", ""))
    if isinstance(weight, (int, float)):
        return (0, float(weight), adapter_id)
    if weight is not None:
        return (1, str(weight), adapter_id)
    return (2, adapter_id, adapter_id)


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_path = args.dataset_path.expanduser()
    tools_path = args.tools_path.expanduser()
    dataset_name = infer_dataset_name(dataset_path, args.dataset_name)
    architecture = str(args.architecture).strip() or "normal"
    loss_name = str(args.loss_name).strip() or "functional_margin"
    variant_name = infer_variant_name(checkpoint_path, loss_name, args.variant_name)
    variant_id = build_variant_id(architecture, loss_name, variant_name)

    output_dir = args.output_root.expanduser() / dataset_name / architecture / loss_name
    results_dir = output_dir / "results"
    model_summary_dir = output_dir / "model_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = load_benchmark_rows(dataset_path)
    if args.limit > 0:
        benchmark_rows = benchmark_rows[: args.limit]
    tools = load_tool_catalog(tools_path)

    variant = EmbeddingVariantSpec(
        variant_id=variant_id,
        architecture=architecture,
        loss_name=loss_name,
        checkpoint_path=checkpoint_path,
    )
    adapter = EmbeddingSelectionAdapter(
        variant,
        device=args.embedding_device,
        ranking_limit=args.ranking_limit,
    )
    summary, results = adapter.evaluate(benchmark_rows, tools)

    result_stem = slugify(adapter.adapter_id)
    results_path = results_dir / f"{result_stem}.jsonl"
    model_summary_path = model_summary_dir / f"{result_stem}.json"
    write_jsonl(results_path, results)

    raw_metadata = summary.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["variant_name"] = variant_name
    compatibility_weight = parse_compatibility_weight(args.compatibility_weight)
    if compatibility_weight is not None:
        metadata["compatibility_weight"] = compatibility_weight
        metadata["compatibility_weight_label"] = str(args.compatibility_weight).strip()

    updated_summary = dict(summary)
    updated_summary["results_path"] = str(results_path.resolve())
    updated_summary["metadata"] = metadata
    write_json(model_summary_path, updated_summary)

    model_summaries = load_model_summaries(model_summary_dir)
    aggregate_summary = build_benchmark_summary(
        benchmark_name="tool_selection_embedding_sweep",
        dataset_path=dataset_path,
        tools_path=tools_path,
        output_dir=output_dir,
        config={
            "dataset_name": dataset_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "ranking_limit": args.ranking_limit,
            "example_count": len(benchmark_rows),
            "embedding_device": args.embedding_device,
            "architecture": architecture,
            "loss_name": loss_name,
            "variant_name": variant_name,
            "compatibility_weight": compatibility_weight,
            "model_summary_dir": str(model_summary_dir.resolve()),
            "results_dir": str(results_dir.resolve()),
        },
        model_summaries=model_summaries,
    )
    write_json(output_dir / "summary.json", aggregate_summary)
    write_json(
        output_dir / "config.json",
        {
            "updated_at": now_utc_iso(),
            "args": make_json_safe(vars(args)),
            "latest_model_summary_path": str(model_summary_path.resolve()),
            "summary_path": str((output_dir / "summary.json").resolve()),
        },
    )

    print(f"Wrote benchmark results to {results_path}")
    print(f"Wrote model summary to {model_summary_path}")
    print(f"Wrote aggregate summary to {output_dir / 'summary.json'}")
    print("Leaderboard:")
    for row in aggregate_summary["leaderboard"]:
        print(
            f"  #{row['rank']} {row['adapter_id']}: "
            f"top1={row['top_1_accuracy']}, "
            f"mrr={row['mean_reciprocal_rank']}, "
            f"latency_ms={row['mean_latency_ms']}"
        )


if __name__ == "__main__":
    main()
