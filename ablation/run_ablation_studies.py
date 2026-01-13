#!/usr/bin/env python3
"""
Main script to run ablation studies.
"""

import argparse
from pathlib import Path

from .generate_ablation_data import AblationDataGenerator
from .evaluate_baselines import compare_baselines, load_test_data


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies: naive prompting vs cross-attention vs full training"
    )
    
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="Generate test data for ablation studies"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples (if generating)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/ablation/test_data.jsonl",
        help="Path to test data file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model to use"
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
    
    args = parser.parse_args()
    
    # Generate data if requested
    if args.generate_data:
        print("=" * 60)
        print("STEP 1: Generating ablation test data")
        print("=" * 60)
        
        generator = AblationDataGenerator()
        dataset = generator.generate_dataset(args.num_samples)
        
        output_path = Path(args.test_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_dataset(dataset, str(output_path), format="jsonl")
        
        print(f"Generated {len(dataset)} examples")
        print(f"Saved to: {output_path}")
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
    results = compare_baselines(
        test_data,
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus
    )
    
    print("\n" + "=" * 60)
    print("Ablation studies complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/baseline_comparison.json")


if __name__ == "__main__":
    main()
