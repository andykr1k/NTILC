# Next Steps - Updated Plan

## ✅ Completed
- Finished naive ablation implementation
- Finished cross attention implementation
- Fixed data generator with more deterministic results
- Visualized embeddings of tool calls in notebook
- **Fixed data format consistency (JSON throughout)**
- **Added contrastive loss for embedding diversity**
- **Added embedding regularization (L2 + variance)**
- **Implemented partial layer freezing**
- **Added label smoothing and scheduled sampling**
- **Created LLM integration architecture**
- **Created NL-to-tool-call data generator**
- **Created end-to-end inference pipeline**
- **Updated README with comprehensive documentation**
- **Updated ablation scripts for JSON format + wandb**
- **Updated evaluation metrics for JSON format support**

## 🔄 In Progress

### Phase 1D: Retrain Autoencoder
Run the improved autoencoder training:

```bash
cd /scratch4/home/akrik/NTILC
python -m training.train_autoencoder
```

**Key improvements made:**
- JSON format for all data (consistent with evaluation)
- Contrastive loss to prevent embedding collapse
- Embedding regularization (L2 + variance)
- Unfrozen later transformer layers (partial freezing)
- Label smoothing (0.1) for better generalization
- Reduced embedding dim (512 -> 256)
- Better hyperparameters (lr: 5e-5, batch: 32)
- Full wandb logging for reproducibility

**Target:** >90% exact match accuracy on test set

## 📋 Pending

### Phase 2: LLM Integration
After autoencoder achieves >90% accuracy:

```bash
python -m training.train_llm_integration --autoencoder_checkpoint checkpoints/best_model.pt
```

This trains the model to:
- Take natural language query: "Get the last 10 orders from California"
- Predict tool embedding in the learned space
- Decode to: `{"tool": "database_query", "arguments": {"sql": "SELECT * FROM orders WHERE state='CA' LIMIT 10", "timeout": 30}}`

### Phase 3: End-to-End Evaluation

```bash
python inference.py
```

Or in Python:
```python
from inference import ToolCallingSystem

system = ToolCallingSystem.from_pretrained(
    autoencoder_path="checkpoints/best_model.pt",
    llm_path="checkpoints/llm_integration/best_model.pt"
)

result = system.predict("Get the last 10 orders from California")
print(result.tool_name)  # database_query
print(result.arguments)  # {"sql": "...", "timeout": 30}
```

### Ablation Studies
Compare baselines against NTILC:

```bash
# Generate test data and run comparisons
python -m ablation.run_ablation_studies \
    --generate_data \
    --num_samples 500 \
    --model_name google/flan-t5-base \
    --use_wandb \
    --output_format json
```

## 📊 Architecture Overview

```
Phase 1: Autoencoder Training
────────────────────────────
Tool Call String ──► Encoder (T5) ──► Embedding (256d) ──► Decoder (T5) ──► Reconstructed String
                      │                    │                   │
                      │    Contrastive     │                   │
                      │◄───Loss + L2 Reg───►                   │
                      │                    │                   │
                      └────────────────────┼───────────────────┘
                              Training Loss│(CrossEntropy + Label Smoothing)
                                           ▼

Phase 2: LLM Integration
────────────────────────
Natural Language ──► LLM Encoder (T5) ──► Tool Prediction Head ──► Predicted Embedding
       │                   │                      │                      │
       │                   │                      │     MSE Loss         │
       │                   │                      │◄─────────────────────┤
       │                   │                      │                      │
       │                   │                      │                      ▼
       │                   │                      │              Frozen Decoder (from Phase 1)
       │                   │                      │                      │
       │                   │                      │                      ▼
       │                   │                      │              Tool Call String
       │                   │                      │                      │
       │                   │             Auxiliary│                      │
       │                   │             Tool Cls │                      │
       │                   └─────────────────────►│                      │
       │                                          │                      │
       └──────────────────────────────────────────┴──────────────────────┘
                                    End-to-End Pipeline
```

## 🔧 Configuration Summary

### Autoencoder (config.py)
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| embedding_dim | 512 | 256 | Reduced for 6 tools |
| freeze_encoder | True | False | Need to train |
| freeze_decoder | True | False | Need to train |
| freeze_encoder_layers | - | 4 | Partial freeze |
| freeze_decoder_layers | - | 4 | Partial freeze |
| learning_rate | 1e-5 | 5e-5 | Faster convergence |
| batch_size | 64 | 32 | Stability |
| use_contrastive_loss | - | True | Prevent collapse |
| label_smoothing | - | 0.1 | Generalization |
| output_format | python | json | Consistency |

### Wandb Configuration
All experiments log to wandb for full reproducibility:
- `wandb_project`: "ntilc"
- `wandb_entity`: "andykr1k"

Logged metrics:
- Training/validation losses (reconstruction, contrastive, L2, variance)
- Embedding statistics (norm, variance)
- Per-tool accuracy breakdown
- Model checkpoints as artifacts
- Test examples and predictions

### Key Files Changed
- `training/data_generator.py` - JSON format + NL generator
- `training/config.py` - New hyperparameters
- `training/losses.py` - Contrastive + regularization
- `training/train_autoencoder.py` - Combined loss
- `models/encoder.py` - Partial freezing + normalization
- `models/decoder.py` - Partial freezing
- `models/autoencoder.py` - Parameter counting
- `models/llm_integration.py` - Phase 2 model
- `training/train_llm_integration.py` - Phase 2 training
- `inference.py` - End-to-end pipeline
- `evaluation/metrics.py` - JSON format support
- `ablation/generate_ablation_data.py` - JSON format support
- `ablation/tool_schemas.py` - JSON format support
- `ablation/run_ablation_studies.py` - Wandb logging

## 📈 Success Metrics

### Phase 1 (Autoencoder)
- **Minimum:** >90% exact match accuracy
- **Target:** >95% exact match accuracy
- **Stretch:** >98% exact match accuracy

### Phase 2 (LLM Integration)
- **Minimum:** >80% tool selection accuracy
- **Target:** >85% exact match accuracy
- **Stretch:** >90% exact match accuracy

### End-to-End
- Latency: <100ms per prediction
- Successful NL → Tool Call for diverse queries

## 🚀 Quick Commands

```bash
# Train autoencoder (Phase 1)
python -m training.train_autoencoder

# Train LLM integration (Phase 2)
python -m training.train_llm_integration \
    --autoencoder_checkpoint checkpoints/best_model.pt

# Run ablation baselines
python -m ablation.run_ablation_studies \
    --generate_data \
    --num_samples 500 \
    --use_wandb

# Run inference
python inference.py --query "Find the latest AI research papers"
```
