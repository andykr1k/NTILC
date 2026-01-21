# NTILC: Neural Tool Invocation via Learned Compression

**Train an LLM to invoke function calls via learned continuous embeddings instead of text generation.**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the autoencoder (Phase 1)
python -m training.train_autoencoder

# 3. Train LLM integration (Phase 2 - after autoencoder converges)
python -m training.train_llm_integration --autoencoder_checkpoint checkpoints/best_model.pt

# 4. Run inference
python inference.py --query "Find me the last 10 orders in California"
```

## Overview

NTILC introduces a novel approach to language model tool use by replacing text-based tool invocation with learned continuous embeddings. Instead of generating text like:

```json
{"tool": "search", "arguments": {"query": "cats", "max_results": 10}}
```

The model predicts a **single 256-dimensional embedding** that encodes the complete tool invocation, which is then decoded back into an executable tool call.

### Why This Approach?

| Aspect | Text-Based | NTILC |
|--------|-----------|-------|
| Generation | O(n) tokens | O(1) embedding |
| Errors | Parsing failures | Learned recovery |
| Similarity | "cats" ≠ "dogs" in tokens | Similar embeddings |
| Gradients | Sparse through tokens | Dense through embeddings |

## Architecture

### Phase 1: Tool Invocation Autoencoder

```
Tool Call String → [Encoder] → Embedding (R^256) → [Decoder] → Reconstructed Tool Call
```

- **Encoder**: T5-based transformer with attention pooling → projection to embedding dim
- **Decoder**: T5-based transformer that autoregressively generates the tool call from embedding
- **Loss**: Cross-entropy + contrastive loss + embedding regularization

### Phase 2: LLM Integration

```
Natural Language Query → [LLM] → Predicted Embedding → [Frozen Decoder] → Tool Call
```

- **LLM**: Takes query and predicts tool embedding via projection head
- **Training**: MSE loss between predicted embedding and encoder output for ground truth
- **Inference**: LLM predicts embedding → Decoder generates tool call

## Project Structure

```
NTILC/
├── models/
│   ├── autoencoder.py      # Main autoencoder module
│   ├── encoder.py          # Tool call → embedding encoder
│   ├── decoder.py          # Embedding → tool call decoder
│   ├── llm_integration.py  # LLM with embedding prediction
│   └── tool_call_utils.py  # Parsing/validation utilities
├── training/
│   ├── config.py           # All configuration (hyperparameters)
│   ├── data_generator.py   # Synthetic tool call generation
│   ├── losses.py           # Loss functions (contrastive, reconstruction)
│   ├── train_autoencoder.py      # Phase 1 training script
│   └── train_llm_integration.py  # Phase 2 training script
├── evaluation/
│   └── metrics.py          # Evaluation metrics
├── ablation/
│   ├── baseline_naive.py         # Naive prompting baseline
│   ├── baseline_cross_attention.py  # Cross-attention baseline
│   ├── generate_ablation_data.py # Test data generation
│   ├── run_ablation_studies.py   # Main ablation script
│   └── README.md
├── data/                   # Generated training data
├── checkpoints/            # Saved model checkpoints
├── output/                 # Evaluation results
└── requirements.txt
```

## Supported Tools

The system supports 6 tool types with various parameter configurations:

| Tool | Description | Parameters |
|------|-------------|------------|
| `search` | Web search | query, max_results, date_filter? |
| `calculate` | Math expressions | expression |
| `database_query` | SQL queries | sql, timeout |
| `send_email` | Email sending | to, subject, body, cc? |
| `web_fetch` | HTTP requests | url, method |
| `file_read` | File reading | path, encoding |

## Training

### Configuration

All hyperparameters are in `training/config.py`:

```python
@dataclass
class AutoencoderConfig:
    # Model
    embedding_dim: int = 256
    encoder_model: str = "google/flan-t5-base"
    decoder_model: str = "google/flan-t5-base"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 50
    
    # Loss
    use_contrastive_loss: bool = True
    contrastive_loss_weight: float = 0.3
    label_smoothing: float = 0.1
    
    # Data
    output_format: str = "json"  # "json" or "python"
    num_train_samples: int = 1000000
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "ntilc"
```

### Wandb Logging

All experiments are fully reproducible via wandb:

```bash
# Training with wandb
python -m training.train_autoencoder

# Ablation studies with wandb
python -m ablation.run_ablation_studies \
    --use_wandb \
    --wandb_project ntilc \
    --generate_data \
    --num_samples 500
```

Logged items:
- All hyperparameters and config
- Training/validation losses per step
- Embedding statistics (norm, variance)
- Per-tool accuracy metrics
- Model checkpoints as artifacts
- Test examples and predictions

### Data Format

Tool calls are generated in JSON format (default):

```json
{"tool": "search", "arguments": {"query": "machine learning", "max_results": 10}}
{"tool": "calculate", "arguments": {"expression": "2 + 2"}}
{"tool": "database_query", "arguments": {"sql": "SELECT * FROM users", "timeout": 30}}
```

## Ablation Studies

Compare NTILC against baselines:

```bash
# Generate test data and run comparisons
python -m ablation.run_ablation_studies \
    --generate_data \
    --num_samples 500 \
    --model_name google/flan-t5-base \
    --use_wandb

# Results saved to output/ablation_results/
```

### Expected Performance

| Approach | Tool Accuracy | Exact Match |
|----------|--------------|-------------|
| Naive Prompting | ~40-60% | ~20-40% |
| Cross-Attention | ~60-75% | ~40-60% |
| **NTILC** | ~80-95% | ~70-90%+ |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `exact_match_accuracy` | Perfectly reconstructed tool calls |
| `tool_accuracy` | Correct tool selection |
| `param_str_accuracy` | String parameter accuracy |
| `param_int_accuracy` | Integer parameter accuracy |
| `embedding_mean_norm` | Average embedding L2 norm |
| `embedding_mean_variance` | Embedding variance (diversity) |

## Theoretical Foundations

### Information Bottleneck

A tool invocation contains ~60-210 bits of information. A 256-dimensional float32 embedding has 8,192 bits capacity — plenty for lossless compression.

### Manifold Hypothesis

Valid tool invocations lie on a low-dimensional manifold in the space of all strings. The autoencoder learns this manifold structure.

### Continuous Optimization

Unlike discrete token prediction, continuous embeddings enable smooth gradients throughout the entire pipeline.

## Research Questions

1. **Optimal Embedding Dimension**: How does dim ∈ {128, 256, 512} affect quality?
2. **Pooling Strategy**: Mean vs CLS vs attention pooling?
3. **Embedding Space Structure**: Do similar tools cluster? Does arithmetic work?
4. **Generalization**: Zero-shot to unseen parameter values?
5. **Multi-Tool Composition**: Can we chain embeddings for workflows?

## Success Metrics

| Level | Reconstruction | End-to-End | Speedup |
|-------|---------------|------------|---------|
| Minimum | 90%+ | 80%+ | 5x+ |
| Strong | 95%+ | 90%+ | 10x+ |
| Breakthrough | 98%+ | 95%+ | 20x+ |

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
faker>=18.0.0
tqdm>=4.65.0
```

## Citation

If you use this work, please cite:

```bibtex
@misc{ntilc2024,
  title={NTILC: Neural Tool Invocation via Learned Compression},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/NTILC}
}
```

## License

MIT License
