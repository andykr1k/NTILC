# Ablation Studies for NTILC

This directory contains baseline experiments to evaluate the contribution of different components before training the full autoencoder system.

## Experimental Design

We compare three approaches in progressive order:

1. **Baseline 1: Naive Prompting** - Simply provide tools in the prompt and ask LLM to generate tool calls
2. **Baseline 2: Cross-Attention LLM** - Add cross-attention mechanism to make LLM aware of tool embeddings
3. **Full NTILC System** - Complete autoencoder + LLM integration (Phase 1 + Phase 2)

## Data Format

The ablation studies support both Python-style and JSON format tool calls:

**Python format:**
```
search(query='machine learning', max_results=10)
```

**JSON format (recommended):**
```json
{"tool": "search", "arguments": {"query": "machine learning", "max_results": 10}}
```

Use `--output_format json` to match the main training pipeline format.

## Quick Start

### 1. Generate Test Data

```bash
# Generate JSON format data (recommended - matches main training)
python -m ablation.generate_ablation_data \
    --num_samples 500 \
    --output ./data/ablation/test_data.jsonl \
    --output_format json

# Or Python format for legacy compatibility
python -m ablation.generate_ablation_data \
    --num_samples 500 \
    --output ./data/ablation/test_data_python.jsonl \
    --output_format python
```

### 2. Run Ablation Studies

```bash
python -m ablation.run_ablation_studies \
    --test_data ./data/ablation/test_data.jsonl \
    --model_name google/flan-t5-base \
    --device cuda \
    --output_dir ./output/ablation_results \
    --num_gpus 1
```

Or generate data and run in one go:

```bash
python -m ablation.run_ablation_studies \
    --generate_data \
    --num_samples 500 \
    --model_name google/flan-t5-base \
    --output_format json
```

## Baseline 1: Naive Prompting

**File**: `baseline_naive.py`

Simply provides tool descriptions in the prompt and asks the LLM to generate tool calls.

**Key Features**:
- No architectural changes
- Tools described in natural language
- Optional example tool calls in prompt
- Standard text generation

**Usage**:
```python
from ablation.baseline_naive import NaivePromptingBaseline

baseline = NaivePromptingBaseline(
    model_name="google/flan-t5-base",
    device="cuda",
    include_examples=True
)

tool_call = baseline.predict("Find me information about machine learning")
```

## Baseline 2: Cross-Attention LLM

**File**: `baseline_cross_attention.py`

Adds cross-attention layers that allow the LLM to attend to tool embeddings during generation.

**Key Features**:
- Learnable tool embeddings (encoded from example tool calls)
- Cross-attention layers between LLM hidden states and tool embeddings
- Architectural awareness of tools
- Can freeze or fine-tune base LLM

**Architecture**:
```
LLM Hidden States → Cross-Attention → Tool Embeddings
                    ↓
              Enhanced Hidden States → LM Head → Tokens
```

**Usage**:
```python
from ablation.baseline_cross_attention import CrossAttentionBaseline

baseline = CrossAttentionBaseline(
    model_name="google/flan-t5-base",
    device="cuda",
    num_tools=6,
    cross_attention_layers=1
)

tool_call = baseline.predict("Find me information about machine learning")
```

## Evaluation Metrics

**File**: `evaluate_baselines.py`

Evaluates baselines using the same metrics as the full system:

| Metric | Description |
|--------|-------------|
| `exact_match_accuracy` | Fraction of perfectly reconstructed tool calls |
| `tool_accuracy` | Fraction of correct tool selections |
| `param_str_accuracy` | Accuracy for string parameters |
| `param_int_accuracy` | Accuracy for integer parameters |
| `per_tool` | Metrics broken down by tool type |
| `total_time_seconds` | Total evaluation time |
| `avg_time_per_prediction_seconds` | Average inference latency |

## Expected Results

Based on typical performance:

| Approach | Tool Accuracy | Exact Match | Notes |
|----------|--------------|-------------|-------|
| Naive Prompting | ~40-60% | ~20-40% | Struggles with parameter formatting |
| Cross-Attention | ~60-75% | ~40-60% | Better parameter understanding |
| **Full NTILC** | ~80-95% | ~70-90%+ | Best generalization, single embedding prediction |

## Integration with Main Pipeline

The ablation baselines should be evaluated on the same test data as the full NTILC system for fair comparison:

```bash
# 1. Generate consistent test data
python -m ablation.generate_ablation_data \
    --num_samples 1000 \
    --output ./data/ablation/test_data.jsonl \
    --output_format json

# 2. Run baseline evaluations
python -m ablation.run_ablation_studies \
    --test_data ./data/ablation/test_data.jsonl \
    --model_name google/flan-t5-base

# 3. After training NTILC, compare on same test set
python inference.py  # Uses checkpoints/best_model.pt
```

## Files

| File | Description |
|------|-------------|
| `tool_schemas.py` | Tool definitions and formatting utilities |
| `baseline_naive.py` | Naive prompting baseline implementation |
| `baseline_cross_attention.py` | Cross-attention baseline implementation |
| `evaluate_baselines.py` | Evaluation framework |
| `generate_ablation_data.py` | Test data generation (supports JSON/Python formats) |
| `run_ablation_studies.py` | Main script to run all experiments |

## Wandb Logging

Results are automatically logged to wandb if enabled:

```bash
python -m ablation.run_ablation_studies \
    --test_data ./data/ablation/test_data.jsonl \
    --use_wandb \
    --wandb_project ntilc \
    --wandb_entity your_entity
```

Logged metrics include:
- Accuracy metrics per baseline
- Per-tool breakdowns
- Timing information
- Comparison tables
