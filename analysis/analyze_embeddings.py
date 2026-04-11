"""
Script to analyze trained autoencoder embeddings.
"""

import sys
from pathlib import Path
import torch
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.autoencoder import ToolInvocationAutoencoder
from training.config import AutoencoderConfig
from training.data_generator import ToolInvocationGenerator, DataGeneratorConfig
from evaluation.metrics import compute_metrics, per_tool_metrics
from evaluation.visualizations import analyze_embedding_space


def main():
    """Analyze embeddings from trained model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze NTILC autoencoder embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Output directory for analysis")
    args = parser.parse_args()
    
    # Load config
    config = AutoencoderConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Update config if saved
    if "config" in checkpoint:
        config.__dict__.update(checkpoint["config"])
    
    model = ToolInvocationAutoencoder(
        embedding_dim=config.embedding_dim,
        encoder_model=config.encoder_model,
        decoder_model=config.decoder_model,
        pooling_strategy=config.pooling_strategy,
        max_length=config.max_length,
        dropout=config.dropout
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Generate test data
    print(f"Generating {args.num_samples} test samples...")
    data_config = DataGeneratorConfig()
    generator = ToolInvocationGenerator(data_config)
    test_tool_calls = generator.generate_dataset(args.num_samples)
    
    # Get embeddings and reconstructions
    print("Computing embeddings and reconstructions...")
    with torch.no_grad():
        embeddings = model.encode(test_tool_calls)
        reconstructed = model.decode(embeddings)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(test_tool_calls, reconstructed, embeddings)
    
    print("\n=== Overall Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    # Per-tool metrics
    print("\n=== Per-Tool Metrics ===")
    per_tool = per_tool_metrics(test_tool_calls, reconstructed)
    for tool, tool_metrics in per_tool.items():
        print(f"\n{tool}:")
        for key, value in tool_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Save metrics
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump({**metrics, "per_tool": per_tool}, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyze_embedding_space(embeddings, test_tool_calls, output_dir=args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
