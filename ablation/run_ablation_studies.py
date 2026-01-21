#!/usr/bin/env python3
"""
Main script to run ablation studies with comprehensive logging.
Supports wandb integration for reproducibility.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from .generate_ablation_data import AblationDataGenerator, OutputFormat
from .evaluate_baselines import compare_baselines, load_test_data, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies: naive prompting vs cross-attention vs full training"
    )
    
    # Data arguments
    parser.add_argument("--generate_data", action="store_true",
                       help="Generate test data for ablation studies")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of test samples (if generating)")
    parser.add_argument("--test_data", type=str, default="./data/ablation/test_data.jsonl",
                       help="Path to test data file")
    parser.add_argument("--output_format", type=str, choices=["python", "json"], default="json",
                       help="Format for tool calls (json recommended to match main training)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                       help="Base model to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./output/ablation_results",
                       help="Output directory for results")
    parser.add_argument("--baseline", type=str, choices=["naive", "cross_attention", "both"],
                       default="both", help="Which baseline(s) to evaluate")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="ntilc",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "experiment_type": "ablation_study",
                    "model_name": args.model_name,
                    "num_samples": args.num_samples,
                    "output_format": args.output_format,
                    "device": args.device,
                    "num_gpus": args.num_gpus,
                    "baseline": args.baseline,
                },
                tags=["ablation", "baseline-comparison"]
            )
            print(f"Wandb run initialized: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            args.use_wandb = False
    
    start_time = time.perf_counter()
    
    # Generate data if requested
    if args.generate_data:
        print("=" * 60)
        print("STEP 1: Generating ablation test data")
        print("=" * 60)
        
        output_format = OutputFormat.JSON if args.output_format == "json" else OutputFormat.PYTHON
        generator = AblationDataGenerator(output_format=output_format)
        dataset = generator.generate_dataset(args.num_samples)
        
        output_path = Path(args.test_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_dataset(dataset, str(output_path), format="jsonl")
        
        print(f"Generated {len(dataset)} examples in {args.output_format} format")
        print(f"Saved to: {output_path}")
        
        # Log data stats to wandb
        if args.use_wandb:
            from collections import Counter
            tool_counts = Counter(item["tool"] for item in dataset)
            wandb.log({
                "data/num_samples": len(dataset),
                "data/output_format": args.output_format,
                **{f"data/tool_count_{k}": v for k, v in tool_counts.items()}
            })
            
            # Save data as artifact
            artifact = wandb.Artifact("ablation_test_data", type="dataset")
            artifact.add_file(str(output_path))
            wandb.log_artifact(artifact)
        
        print()
    
    # Run ablation studies
    print("=" * 60)
    print("STEP 2: Running ablation studies")
    print("=" * 60)
    
    # Load test data
    if not Path(args.test_data).exists():
        print(f"Error: Test data not found at {args.test_data}")
        print("Run with --generate_data first, or provide --test_data path")
        return
    
    print(f"Loading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples\n")
    
    # Compare baselines
    if args.baseline == "both":
        results = compare_baselines(
            test_data,
            model_name=args.model_name,
            device=args.device,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus
        )
    else:
        results = {
            args.baseline: evaluate_baseline(
                args.baseline,
                test_data,
                model_name=args.model_name,
                device=args.device,
                output_dir=args.output_dir,
                num_gpus=args.num_gpus
            )
        }
    
    total_time = time.perf_counter() - start_time
    results["total_run_time_seconds"] = total_time
    
    # Log results to wandb
    if args.use_wandb:
        for baseline_name, baseline_results in results.items():
            if baseline_name == "total_run_time_seconds":
                continue
            
            # Flatten per_tool metrics
            flat_results = {}
            for k, v in baseline_results.items():
                if k == "per_tool" and isinstance(v, dict):
                    for tool_name, tool_metrics in v.items():
                        for metric_name, metric_value in tool_metrics.items():
                            flat_results[f"{baseline_name}/per_tool/{tool_name}/{metric_name}"] = metric_value
                elif isinstance(v, (int, float)):
                    flat_results[f"{baseline_name}/{k}"] = v
            
            wandb.log(flat_results)
        
        wandb.log({"total_run_time_seconds": total_time})
        
        # Create summary table
        if args.baseline == "both":
            table = wandb.Table(columns=["Metric", "Naive Prompting", "Cross-Attention"])
            for metric in ["exact_match_accuracy", "tool_accuracy", "total_time_seconds"]:
                naive_val = results.get("naive", {}).get(metric, 0.0)
                cross_val = results.get("cross_attention", {}).get(metric, 0.0)
                table.add_data(metric, naive_val, cross_val)
            wandb.log({"comparison_table": table})
        
        # Save results as artifact
        results_path = Path(args.output_dir) / "baseline_comparison.json"
        artifact = wandb.Artifact("ablation_results", type="results")
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
    
    print("\n" + "=" * 60)
    print("Ablation studies complete!")
    print("=" * 60)
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}/baseline_comparison.json")
    
    if args.use_wandb and wandb_run:
        print(f"Wandb run: {wandb_run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
