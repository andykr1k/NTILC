"""CLI for benchmarking NTILC against a prompt-only baseline."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - current shell may not have tqdm installed
    def tqdm(iterable, *args, **kwargs):
        del args, kwargs
        return iterable

from evaluation.metrics import (
    FULL_CLEAN_SPLIT,
    TRAIN_OVERLAP_SPLIT,
    UNSEEN_ONLY_SPLIT,
    aggregate_predictions,
    build_tool_metadata,
    dataset_partition_counts,
    evaluate_command_prediction,
    load_json_or_jsonl,
    prepare_eval_rows,
    select_eval_rows,
    write_json,
    write_jsonl,
)
from evaluation.prompt_baseline import PromptBaselineModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NTILC vs prompt-baseline evaluation.")
    parser.add_argument(
        "--clean-data",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_clean_v2.json"),
        help="Flat clean dataset used for evaluation candidates.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_train_v2.json"),
        help="Training reference used to define unseen rows.",
    )
    parser.add_argument(
        "--raw-tools-json",
        type=Path,
        default=Path("data/man/raw_ai.json"),
        help="Raw man-page tool registry.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/evaluations"),
        help="Directory where eval artifacts are written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output subdirectory name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--split",
        choices=[FULL_CLEAN_SPLIT, TRAIN_OVERLAP_SPLIT, UNSEEN_ONLY_SPLIT],
        default=UNSEEN_ONLY_SPLIT,
        help="Evaluation split to use.",
    )
    parser.add_argument(
        "--include-complex",
        action="store_true",
        help="Include shell-composition rows with pipes/control operators/redirections.",
    )
    parser.add_argument("--num-samples", type=int, default=0, help="Optional sample cap. 0 means all rows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Base Qwen model name/path for both NTILC and baseline.",
    )
    parser.add_argument(
        "--intent-embedder-path",
        type=str,
        default="checkpoints/intent_embedder/best_model.pt",
        help="Intent embedder checkpoint for API compatibility.",
    )
    parser.add_argument(
        "--query-encoder-path",
        type=str,
        default="checkpoints/cluster_retrieval/best_model.pt",
        help="Cluster retrieval checkpoint.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default="checkpoints/lora_nl_command_full",
        help="Current NTILC LoRA adapter path.",
    )
    parser.add_argument(
        "--ntilc-device",
        type=str,
        default="cuda:1",
        help="Device for the NTILC retrieval and Qwen runtime.",
    )
    parser.add_argument(
        "--baseline-device",
        type=str,
        default="cuda:0",
        help="Device for the prompt-only baseline model.",
    )
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--plan-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-plan-actions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--baseline-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for the prompt-only baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    clean_rows = load_json_or_jsonl(args.clean_data)
    train_rows = load_json_or_jsonl(args.train_data)
    raw_tool_rows = load_json_or_jsonl(args.raw_tools_json)
    tool_metadata = build_tool_metadata(raw_tool_rows)

    prepared_rows = prepare_eval_rows(clean_rows, train_rows, tool_metadata)
    selected_rows = select_eval_rows(
        prepared_rows,
        split=args.split,
        include_complex=args.include_complex,
        num_samples=args.num_samples if args.num_samples > 0 else None,
        seed=args.seed,
    )
    if not selected_rows:
        raise ValueError("No evaluation rows selected. Adjust split or include-complex settings.")

    run_name = args.run_name.strip() or datetime.now().strftime("tool_calling_eval_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_counts = dataset_partition_counts(prepared_rows)
    config_payload = {
        "run_name": run_name,
        "clean_data": str(args.clean_data),
        "train_data": str(args.train_data),
        "raw_tools_json": str(args.raw_tools_json),
        "split": args.split,
        "include_complex": bool(args.include_complex),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "qwen_model": args.qwen_model,
        "intent_embedder_path": args.intent_embedder_path,
        "query_encoder_path": args.query_encoder_path,
        "lora_adapter_path": args.lora_adapter_path,
        "ntilc_device": args.ntilc_device,
        "baseline_device": args.baseline_device,
        "top_k_candidates": int(args.top_k_candidates),
        "max_retries": int(args.max_retries),
        "max_new_tokens": int(args.max_new_tokens),
        "plan_max_new_tokens": int(args.plan_max_new_tokens),
        "max_plan_actions": int(args.max_plan_actions),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "baseline_max_new_tokens": int(args.baseline_max_new_tokens),
        "dataset_partition_counts": dataset_counts,
        "selected_examples": len(selected_rows),
    }
    write_json(output_dir / "config.json", config_payload)

    ntilc_rows = evaluate_ntilc(selected_rows, tool_metadata, args)
    _release_memory()
    baseline_rows = evaluate_prompt_baseline(selected_rows, raw_tool_rows, tool_metadata, args)
    _release_memory()

    predictions = merge_predictions(selected_rows, ntilc_rows, baseline_rows)
    summary = aggregate_predictions(predictions)
    summary["config"] = {
        "run_name": run_name,
        "split": args.split,
        "include_complex": bool(args.include_complex),
        "selected_examples": len(selected_rows),
    }

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def evaluate_ntilc(
    rows: Sequence[Dict[str, Any]],
    tool_metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    from inference import NTILCOrchestratorAgent

    agent = NTILCOrchestratorAgent.from_pretrained(
        intent_embedder_path=args.intent_embedder_path,
        query_encoder_path=args.query_encoder_path,
        qwen_model_name_or_path=args.qwen_model,
        lora_adapter_path=args.lora_adapter_path,
        lora_mode="full",
        auto_register_shell_tools=True,
        device=args.ntilc_device,
    )

    outputs: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="NTILC eval", unit="example"):
        start = time.perf_counter()
        run = agent.run(
            request=row["nl_query"],
            execute_tools=False,
            top_k_candidates=args.top_k_candidates,
            max_retries=args.max_retries,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_plan_actions=args.max_plan_actions,
            plan_max_new_tokens=args.plan_max_new_tokens,
        )
        latency = time.perf_counter() - start

        final_step = run.final_step
        predicted_command = final_step.command if final_step is not None else ""
        command_metrics = evaluate_command_prediction(predicted_command, row, tool_metadata)

        candidates = [candidate.to_dict() for candidate in run.candidates]
        top1_match = bool(run.candidates) and run.candidates[0].tool_name == row["tool"]
        hit_at_k = any(candidate.tool_name == row["tool"] for candidate in run.candidates)
        structured_output = bool(run.plan_block.strip()) and bool(run.steps) and all(
            step.dispatch_block.strip() and step.response_block.strip() for step in run.steps
        )

        outputs.append(
            {
                "example_id": row["example_id"],
                "latency_seconds": latency,
                "success": bool(run.success),
                "structured_output": structured_output,
                "retrieval_top1_label_match": top1_match,
                "retrieval_hit_at_k": hit_at_k,
                "candidate_count": len(run.candidates),
                "step_count": len(run.steps),
                "plan_block": run.plan_block,
                "action_failures": list(run.action_failures),
                "predicted_command": command_metrics["predicted_command"],
                "predicted_command_raw_normalized": command_metrics["predicted_command_raw_normalized"],
                "predicted_command_canonical": command_metrics["predicted_command_canonical"],
                "raw_exact_match": command_metrics["raw_exact_match"],
                "canonical_exact_match": command_metrics["canonical_exact_match"],
                "command_tool_match": command_metrics["command_tool_match"],
                "generated_text": final_step.generated_text if final_step is not None else "",
                "final_candidate_tool": final_step.candidate.tool_name if final_step is not None else "",
                "raw_run": {
                    "success": bool(run.success),
                    "candidates": candidates,
                    "steps": [step.to_dict() for step in run.steps],
                    "atomic_actions": list(run.atomic_actions),
                    "action_failures": list(run.action_failures),
                },
            }
        )

    del agent
    return outputs


def evaluate_prompt_baseline(
    rows: Sequence[Dict[str, Any]],
    raw_tool_rows: Sequence[Dict[str, Any]],
    tool_metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    baseline = PromptBaselineModel.from_pretrained(
        model_name_or_path=args.qwen_model,
        raw_rows=raw_tool_rows,
        max_new_tokens=args.baseline_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.baseline_device,
    )

    outputs: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Prompt baseline eval", unit="example"):
        start = time.perf_counter()
        prediction = baseline.generate(row["nl_query"])
        latency = time.perf_counter() - start
        command_metrics = evaluate_command_prediction(
            prediction["predicted_command"],
            row,
            tool_metadata,
        )

        outputs.append(
            {
                "example_id": row["example_id"],
                "latency_seconds": latency,
                "strict_json_parse": bool(prediction["strict_json_parse"]),
                "predicted_tool": prediction["predicted_tool"],
                "predicted_command": command_metrics["predicted_command"],
                "predicted_command_raw_normalized": command_metrics["predicted_command_raw_normalized"],
                "predicted_command_canonical": command_metrics["predicted_command_canonical"],
                "raw_exact_match": command_metrics["raw_exact_match"],
                "canonical_exact_match": command_metrics["canonical_exact_match"],
                "command_tool_match": command_metrics["command_tool_match"],
                "raw_text": prediction["raw_text"],
                "parse_error": prediction["parse_error"],
                "payload": prediction["payload"],
            }
        )

    del baseline
    return outputs


def merge_predictions(
    rows: Sequence[Dict[str, Any]],
    ntilc_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ntilc_by_id = {row["example_id"]: row for row in ntilc_rows}
    baseline_by_id = {row["example_id"]: row for row in baseline_rows}

    merged: List[Dict[str, Any]] = []
    for row in rows:
        example_id = row["example_id"]
        merged.append(
            {
                "example_id": example_id,
                "query": row["nl_query"],
                "tool": row["tool"],
                "source_url": row.get("source_url", ""),
                "split": row["split"],
                "is_complex_shell": bool(row["is_complex_shell"]),
                "gold_command": row["command"],
                "gold_command_raw_normalized": row["raw_command_normalized"],
                "gold_command_canonical": row["canonical_command"],
                "canonical_tool_prefix": row["canonical_tool_prefix"],
                "ntilc": dict(ntilc_by_id.get(example_id, {})),
                "baseline": dict(baseline_by_id.get(example_id, {})),
            }
        )
    return merged


def _release_memory() -> None:
    gc.collect()
    try:  # pragma: no cover - only relevant in ML runtime
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


if __name__ == "__main__":
    main()
