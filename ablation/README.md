# Ablation Studies for NTILC

This directory contains baseline experiments to evaluate the contribution of different components before training the full autoencoder system.

## Experimental Design

We compare three approaches in progressive order:

1. **Baseline 1: Naive Prompting** - Simply provide tools in the prompt and ask LLM to generate tool calls
2. **Baseline 2: Cross-Attention LLM** - Add cross-attention mechanism to make LLM aware of tool embeddings
3. **Full Training** - Complete autoencoder + LLM integration (Phase 1 + Phase 2)

## Why This Approach?

This ablation study helps us understand:
- **Baseline performance**: How well can a standard LLM do with just prompting?
- **Cross-attention benefit**: Does architectural awareness of tools help?
- **Full system value**: What additional benefit does the learned embedding space provide?

## Quick Start

### 1. Generate Test Data

```bash
python -m ablation.generate_ablation_data \
    --num_samples 100 \
    --output ./data/ablation/test_data.jsonl
```

### 2. Run Ablation Studies

```bash
python -m ablation.run_ablation_studies \
    --test_data ./data/ablation/test_data.jsonl \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda \
    --output_dir ./output/ablation_results \
    --num_gpus 4

```

Or generate data and run in one go:

```bash
python -m ablation.run_ablation_studies \
    --generate_data \
    --num_samples 100 \
    --model_name Qwen/Qwen2.5-1.5B-Instruct
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
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    include_examples=True
)

tool_call = baseline.predict("Find me information about machine learning")
```

## Baseline 2: Cross-Attention LLM

**File**: `baseline_cross_attention.py`

Adds cross-attention layers that allow the LLM to attend to tool embeddings during generation.

**Key Features**:
- Learnable tool embeddings
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
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    num_tools=6,
    cross_attention_layers=1
)

tool_call = baseline.predict("Find me information about machine learning")
```

## Evaluation

**File**: `evaluate_baselines.py`

Evaluates baselines using the same metrics as the full system:
- Exact match accuracy
- Tool selection accuracy
- Parameter accuracy (by type)
- Per-tool metrics

**Metrics**:
- `exact_match_accuracy`: Fraction of perfectly reconstructed tool calls
- `tool_accuracy`: Fraction of correct tool selections
- `param_str_accuracy`: Accuracy for string parameters
- `param_int_accuracy`: Accuracy for integer parameters
- `per_tool`: Metrics broken down by tool type

## Data Format

Test data should be in JSONL format with the following structure:

```json
{"query": "Find me information about machine learning", "ground_truth": "search(query='machine learning', max_results=10)", "tool": "search"}
{"query": "Calculate 2 + 2", "ground_truth": "calculate(expression='2 + 2')", "tool": "calculate"}
```

## Expected Results

Based on typical performance:

1. **Naive Prompting**: 
   - Tool accuracy: ~40-60%
   - Exact match: ~20-40%
   - Struggles with parameter formatting

2. **Cross-Attention**:
   - Tool accuracy: ~60-75%
   - Exact match: ~40-60%
   - Better parameter understanding

3. **Full System** (after training):
   - Tool accuracy: ~80-95%
   - Exact match: ~70-90%
   - Best generalization

## Next Steps

After running ablation studies:

1. **Review results** in `./output/ablation_results/baseline_comparison.json`
2. **Train autoencoder** (Phase 1) - see `training/train_autoencoder.py`
3. **Integrate with LLM** (Phase 2) - to be implemented
4. **Compare all three approaches** on same test set

## Files

- `tool_schemas.py`: Tool definitions and formatting utilities
- `baseline_naive.py`: Naive prompting baseline implementation
- `baseline_cross_attention.py`: Cross-attention baseline implementation
- `evaluate_baselines.py`: Evaluation framework
- `generate_ablation_data.py`: Test data generation
- `run_ablation_studies.py`: Main script to run all experiments
