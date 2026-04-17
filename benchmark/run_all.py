from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters import (
    AnthropicSelectionAdapter,
    EmbeddingSelectionAdapter,
    GeminiSelectionAdapter,
    HybridEmbeddingRerankAdapter,
    LocalHFSelectionAdapter,
    LocalHFSelectionEngine,
    OpenAISelectionAdapter,
    build_model_summary,
)
from benchmark.common import (
    DEFAULT_RANKING_LIMIT,
    build_benchmark_summary,
    discover_embedding_variants,
    load_benchmark_rows,
    load_tool_catalog,
    now_utc_iso,
    slugify,
    write_json,
    write_jsonl,
)

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "OSS" / "tool_embedding_dataset_test.jsonl"
DEFAULT_TOOLS_PATH = REPO_ROOT / "data" / "OSS" / "tools.json"
DEFAULT_EMBEDDING_ROOT = REPO_ROOT / "data" / "OSS" / "output"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "output"
DEFAULT_HYBRID_RERANKER = "Qwen/Qwen3.5-27B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark tool selection across embedding checkpoints, local HF models, "
            "provider APIs, and a hybrid embedding+Qwen reranker."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--tools-path", type=Path, default=DEFAULT_TOOLS_PATH)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--checkpoint-filename", default="best.pt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on benchmark examples for quick debug runs.")
    parser.add_argument("--ranking-limit", type=int, default=DEFAULT_RANKING_LIMIT)
    parser.add_argument("--embedding-device", default="cuda:4")
    parser.add_argument("--embedding-top-k", type=int, default=5)
    parser.add_argument("--hf-model", action="append", default=[], help="Repeat to benchmark local HF instruction models.")
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--hf-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--hf-max-new-tokens", type=int, default=160)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hybrid-reranker-model", default=DEFAULT_HYBRID_RERANKER)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--hybrid-top-k", type=int, default=5)
    parser.add_argument("--openai-model", action="append", default=[], help="Repeat to benchmark OpenAI API models.")
    parser.add_argument("--anthropic-model", action="append", default=[], help="Repeat to benchmark Anthropic API models.")
    parser.add_argument("--gemini-model", action="append", default=[], help="Repeat to benchmark Gemini API models.")
    parser.add_argument("--api-max-output-tokens", type=int, default=160)
    parser.add_argument("--api-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--pricing-path",
        type=Path,
        default=None,
        help="Optional JSON file mapping model_name -> {input_per_million_usd, output_per_million_usd}.",
    )
    return parser.parse_args()


def load_pricing(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return payload


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def build_run_name(args: argparse.Namespace) -> str:
    if str(args.run_name).strip():
        return str(args.run_name).strip()
    return f"tool-selection-{slugify(now_utc_iso())}"


def persist_model_results(
    *,
    run_dir: Path,
    adapter_id: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{slugify(adapter_id)}.jsonl"
    write_jsonl(results_path, results)
    updated_summary = dict(summary)
    updated_summary["results_path"] = str(results_path.resolve())
    return updated_summary


def main() -> None:
    args = parse_args()
    run_name = build_run_name(args)
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = load_benchmark_rows(args.dataset_path)
    if args.limit > 0:
        benchmark_rows = benchmark_rows[: args.limit]
    tools = load_tool_catalog(args.tools_path)
    pricing = load_pricing(args.pricing_path)
    embedding_variants = discover_embedding_variants(args.embedding_root, args.checkpoint_filename)

    model_summaries: list[dict[str, Any]] = []

    for variant in embedding_variants:
        adapter = EmbeddingSelectionAdapter(
            variant,
            device=args.embedding_device,
            ranking_limit=args.ranking_limit,
        )
        summary, results = adapter.evaluate(benchmark_rows, tools, pricing=pricing)
        model_summaries.append(
            persist_model_results(
                run_dir=run_dir,
                adapter_id=adapter.adapter_id,
                summary=summary,
                results=results,
            )
        )

    reranker_engine: LocalHFSelectionEngine | None = None
    try:
        if not args.no_hybrid and args.hybrid_reranker_model and embedding_variants:
            reranker_engine = LocalHFSelectionEngine(
                args.hybrid_reranker_model,
                device=args.hf_device,
                dtype=args.hf_dtype,
                max_new_tokens=args.hf_max_new_tokens,
                ranking_limit=args.ranking_limit,
                local_files_only=args.hf_local_files_only,
            )
            for variant in embedding_variants:
                adapter = HybridEmbeddingRerankAdapter(
                    variant,
                    reranker_engine,
                    device=args.embedding_device,
                    embedding_top_k=args.hybrid_top_k,
                    ranking_limit=args.ranking_limit,
                    pricing=pricing,
                )
                summary, results = adapter.evaluate(benchmark_rows, tools)
                model_summaries.append(
                    persist_model_results(
                        run_dir=run_dir,
                        adapter_id=adapter.adapter_id,
                        summary=summary,
                        results=results,
                    )
                )

        for model_name in args.hf_model:
            adapter = LocalHFSelectionAdapter(
                model_name,
                device=args.hf_device,
                dtype=args.hf_dtype,
                ranking_limit=args.ranking_limit,
                max_new_tokens=args.hf_max_new_tokens,
                local_files_only=args.hf_local_files_only,
                pricing=pricing,
            )
            summary, results = adapter.evaluate(benchmark_rows, tools)
            model_summaries.append(
                persist_model_results(
                    run_dir=run_dir,
                    adapter_id=adapter.adapter_id,
                    summary=summary,
                    results=results,
                )
            )
            gc.collect()

    finally:
        if reranker_engine is not None:
            reranker_engine.close()

    for model_name in args.openai_model:
        try:
            adapter = OpenAISelectionAdapter(
                model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=pricing,
            )
            summary, results = adapter.evaluate(benchmark_rows, tools)
        except Exception as exc:
            summary = build_model_summary(
                adapter_id=f"openai/{slugify(model_name)}",
                provider="openai",
                mode="llm_api",
                model_name=model_name,
                results=[],
                status="error",
                error_message=str(exc),
            )
            results = []
        model_summaries.append(
            persist_model_results(
                run_dir=run_dir,
                adapter_id=summary["adapter_id"],
                summary=summary,
                results=results,
            )
        )

    for model_name in args.anthropic_model:
        try:
            adapter = AnthropicSelectionAdapter(
                model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=pricing,
            )
            summary, results = adapter.evaluate(benchmark_rows, tools)
        except Exception as exc:
            summary = build_model_summary(
                adapter_id=f"anthropic/{slugify(model_name)}",
                provider="anthropic",
                mode="llm_api",
                model_name=model_name,
                results=[],
                status="error",
                error_message=str(exc),
            )
            results = []
        model_summaries.append(
            persist_model_results(
                run_dir=run_dir,
                adapter_id=summary["adapter_id"],
                summary=summary,
                results=results,
            )
        )

    for model_name in args.gemini_model:
        try:
            adapter = GeminiSelectionAdapter(
                model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=pricing,
            )
            summary, results = adapter.evaluate(benchmark_rows, tools)
        except Exception as exc:
            summary = build_model_summary(
                adapter_id=f"gemini/{slugify(model_name)}",
                provider="gemini",
                mode="llm_api",
                model_name=model_name,
                results=[],
                status="error",
                error_message=str(exc),
            )
            results = []
        model_summaries.append(
            persist_model_results(
                run_dir=run_dir,
                adapter_id=summary["adapter_id"],
                summary=summary,
                results=results,
            )
        )

    summary = build_benchmark_summary(
        benchmark_name="tool_selection",
        dataset_path=args.dataset_path,
        tools_path=args.tools_path,
        output_dir=run_dir,
        config={
            "checkpoint_root": str(args.embedding_root.resolve()),
            "checkpoint_filename": args.checkpoint_filename,
            "ranking_limit": args.ranking_limit,
            "example_count": len(benchmark_rows),
            "embedding_top_k": args.embedding_top_k,
            "hybrid_top_k": args.hybrid_top_k,
            "hybrid_reranker_model": args.hybrid_reranker_model if not args.no_hybrid else "",
            "hf_models": list(args.hf_model),
            "openai_models": list(args.openai_model),
            "anthropic_models": list(args.anthropic_model),
            "gemini_models": list(args.gemini_model),
            "pricing_path": str(args.pricing_path.resolve()) if args.pricing_path is not None else "",
        },
        model_summaries=model_summaries,
    )
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "config.json",
        {
            "created_at": now_utc_iso(),
            "args": make_json_safe(vars(args)),
            "run_name": run_name,
        },
    )

    print(f"Wrote benchmark summary to {run_dir / 'summary.json'}")
    print("Leaderboard:")
    for row in summary["leaderboard"]:
        print(
            f"  #{row['rank']} {row['adapter_id']}: "
            f"top1={row['top_1_accuracy']}, "
            f"mrr={row['mean_reciprocal_rank']}, "
            f"latency_ms={row['mean_latency_ms']}"
        )


if __name__ == "__main__":
    main()
