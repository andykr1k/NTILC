"""
Evaluation script for comparing baseline approaches.
"""

import json
import argparse
import time
from typing import List, Dict
from pathlib import Path

from .baseline_naive import NaivePromptingBaseline
from .baseline_cross_attention import CrossAttentionBaseline
from evaluation.metrics import exact_match_accuracy, tool_accuracy, parameter_accuracy, per_tool_metrics


def load_test_data(data_path: str) -> List[Dict[str, str]]:
    """
    Load test data with queries and ground truth tool calls.
    
    Expected format: JSON lines with 'query' and 'ground_truth' fields
    """
    test_data = []
    
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            test_data = json.load(f)
    else:
        # Assume text format: query | ground_truth
        with open(data_path, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|', 1)
                    test_data.append({
                        "query": parts[0].strip(),
                        "ground_truth": parts[1].strip()
                    })
    
    return test_data


def evaluate_baseline(
    baseline_name: str,
    test_data: List[Dict[str, str]],
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    output_dir: str = "./output/ablation_results",
    num_gpus: int = 1,
    save_predictions: bool = True,
    **kwargs
) -> Dict:
    """
    Evaluate a baseline approach.
    
    Args:
        baseline_name: "naive" or "cross_attention"
        test_data: List of test examples
        model_name: Model to use
        device: Device to run on
        **kwargs: Additional arguments for baseline
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating {baseline_name} baseline...")
    
    if baseline_name == "naive":
        baseline = NaivePromptingBaseline(
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            **kwargs
        )
    elif baseline_name == "cross_attention":
        baseline = CrossAttentionBaseline(
            model_name=model_name,
            device=device,
            num_gpus=num_gpus,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    # Evaluate with timing
    start_time = time.perf_counter()
    metrics = baseline.evaluate(test_data)
    total_time = time.perf_counter() - start_time
    
    # Add per-tool metrics (reuse predictions from evaluate to avoid duplicate computation)
    queries = [item["query"] for item in test_data]
    ground_truth = [item["ground_truth"] for item in test_data]
    
    # Reuse predictions from evaluate if available, otherwise compute
    if "predictions" in metrics:
        predictions = metrics["predictions"]
        original_predictions = metrics.get("original_predictions", [None] * len(predictions))
        # Remove predictions from metrics dict (it's just for internal use)
        del metrics["predictions"]
        if "original_predictions" in metrics:
            del metrics["original_predictions"]
    else:
        # predict_batch now returns tuples
        predictions_with_original = baseline.predict_batch(queries)
        predictions = [tool_call for _, tool_call in predictions_with_original]
        original_predictions = [orig for orig, _ in predictions_with_original]
    
    per_tool = per_tool_metrics(ground_truth, predictions)
    metrics["per_tool"] = per_tool
    
    # Add timing metrics
    num_predictions = len(predictions)
    metrics["total_time_seconds"] = total_time
    metrics["avg_time_per_prediction_seconds"] = total_time / num_predictions if num_predictions > 0 else 0.0
    
    # Print timing information
    print(f"\nTiming Information:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per prediction: {metrics['avg_time_per_prediction_seconds']:.4f} seconds")
    print(f"  Number of predictions: {num_predictions}")
    
    # Save predictions and ground truth to file
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        predictions_file = output_path / f"{baseline_name}_predictions.jsonl"
        with open(predictions_file, 'w') as f:
            for query, orig_pred, pred, gt in zip(queries, original_predictions, predictions, ground_truth):
                f.write(json.dumps({
                    "query": query,
                    "original_prediction": orig_pred,
                    "prediction": pred,
                    "ground_truth": gt
                }) + "\n")
        
        print(f"Predictions saved to: {predictions_file}")
    
    return metrics


def compare_baselines(
    test_data: List[Dict[str, str]],
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    output_dir: str = "./output/ablation_results",
    num_gpus: int = 1
) -> Dict:
    """
    Compare all baseline approaches.
    
    Returns:
        Dictionary with results for each baseline
    """
    results = {}
    
    # Track total time for the whole run
    total_start_time = time.perf_counter()
    
    # Evaluate naive baseline
    print("=" * 60)
    print("BASELINE 1: Naive Prompting")
    print("=" * 60)
    results["naive"] = evaluate_baseline(
        "naive",
        test_data,
        model_name=model_name,
        device=device,
        num_gpus=num_gpus
    )
    
    # Evaluate cross-attention baseline
    print("\n" + "=" * 60)
    print("BASELINE 2: Cross-Attention LLM")
    print("=" * 60)
    results["cross_attention"] = evaluate_baseline(
        "cross_attention",
        test_data,
        model_name=model_name,
        device=device,
        num_gpus=num_gpus,
        cross_attention_layers=1
    )
    
    # Calculate total time
    total_time = time.perf_counter() - total_start_time
    results["total_run_time_seconds"] = total_time
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Naive':<15} {'Cross-Attention':<15}")
    print("-" * 60)
    
    for metric in ["exact_match_accuracy", "tool_accuracy"]:
        naive_val = results["naive"].get(metric, 0.0)
        cross_attn_val = results["cross_attention"].get(metric, 0.0)
        print(f"{metric:<30} {naive_val:<15.4f} {cross_attn_val:<15.4f}")
    
    # Print timing information
    print("\n" + "=" * 60)
    print("TIMING INFORMATION")
    print("=" * 60)
    print(f"{'Metric':<40} {'Naive':<20} {'Cross-Attention':<20}")
    print("-" * 80)
    
    naive_total = results["naive"].get("total_time_seconds", 0.0)
    naive_avg = results["naive"].get("avg_time_per_prediction_seconds", 0.0)
    cross_attn_total = results["cross_attention"].get("total_time_seconds", 0.0)
    cross_attn_avg = results["cross_attention"].get("avg_time_per_prediction_seconds", 0.0)
    
    print(f"{'Total time (seconds)':<40} {naive_total:<20.4f} {cross_attn_total:<20.4f}")
    print(f"{'Avg time per prediction (seconds)':<40} {naive_avg:<20.4f} {cross_attn_avg:<20.4f}")
    print(f"\n{'Total run time (all baselines)':<40} {total_time:<20.4f} seconds")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_path = output_path / "baseline_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline approaches")
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data file (JSON, JSONL, or text format)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (creates separate model instances per GPU)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/ablation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["naive", "cross_attention", "both"],
        default="both",
        help="Which baseline(s) to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Track total run time
    main_start_time = time.perf_counter()
    
    # Evaluate
    if args.baseline == "both":
        results = compare_baselines(
            test_data,
            model_name=args.model_name,
            device=args.device,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus
        )
    elif args.baseline == "naive":
        results = {
            "naive": evaluate_baseline(
                "naive",
                test_data,
                model_name=args.model_name,
                device=args.device,
                output_dir=args.output_dir,
                num_gpus=args.num_gpus
            )
        }
        # Add total run time for single baseline
        total_time = time.perf_counter() - main_start_time
        results["total_run_time_seconds"] = total_time
        print(f"\nTotal run time: {total_time:.4f} seconds")
    elif args.baseline == "cross_attention":
        results = {
            "cross_attention": evaluate_baseline(
                "cross_attention",
                test_data,
                model_name=args.model_name,
                device=args.device,
                output_dir=args.output_dir,
                num_gpus=args.num_gpus
            )
        }
        # Add total run time for single baseline
        total_time = time.perf_counter() - main_start_time
        results["total_run_time_seconds"] = total_time
        print(f"\nTotal run time: {total_time:.4f} seconds")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
