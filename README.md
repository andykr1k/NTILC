# NTILC: Neural Tool Invocation via Learned Compression (NEW ARCHITECTURE)

**Train a system to invoke function calls via cluster-based retrieval instead of text generation.**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train intent embedder (Phase 1)
python -m training.train_intent_embedding

# 3. Train cluster retrieval (Phase 2 - after intent embedder converges)
python -m training.train_cluster_retrieval

# 4. Run inference
python inference.py
```

## Overview

NTILC introduces a novel approach to language model tool use by replacing text-based tool invocation with **cluster-based retrieval**. Instead of generating text like:

```python
search(query="cats", max_results=10)
```

The system:
1. Embeds tool intents into a 1024-D space
2. Projects to 128-D for similarity computation
3. Retrieves cluster IDs based on query similarity
4. Maps cluster IDs to tools via a software layer
5. Infers arguments separately

### Why This Approach?

| Aspect | Text-Based | Old NTILC | New NTILC |
|--------|-----------|-----------|-----------|
| Generation | O(n) tokens | O(1) embedding | O(1) cluster lookup |
| Speed | Slow (autoregressive) | Medium (decoder) | Fast (similarity) |
| Errors | Parsing failures | Decoder errors | Cluster mismatch |
| Interpretability | Low | Medium | High (cluster IDs) |
| Extensibility | Retrain all | Retrain decoder | Add single embedding |

## Architecture (NEW)

### Phase 1: Intent Embedding

```
Tool Intent → [Intent Embedder] → 1024-D Embedding → [Projection Head] → 128-D
```

- **Intent Embedder**: T5-based transformer that embeds canonicalized tool intents
  - Includes: tool name, description, schema, examples, paraphrases
- **Projection Head**: Projects 1024-D → 128-D for similarity computation
- **Loss**: Circle Loss for metric learning + contrastive loss + regularization

### Phase 2: Cluster Retrieval

```
Natural Language Query → [Query Encoder] → 128-D Embedding → [Cluster Retrieval] → Cluster ID
Cluster ID → [Software Layer] → Tool
Query → [Argument Inference] → Arguments
```

- **Query Encoder**: Encodes NL queries to 128-D embeddings
- **Cluster Retrieval**: Computes similarity to cluster centroids, returns cluster ID
- **Software Layer**: Maps cluster IDs to tools (decouples model from execution)
- **Argument Inference**: Handles arguments separately from tool selection

## Project Structure

```
NTILC/
├── models/
│   ├── intent_embedder.py      # Tool intent → 1024-D embeddings (NEW)
│   ├── projection_head.py      # 1024-D → 128-D projection (NEW)
│   ├── cluster_retrieval.py     # Cluster ID retrieval (NEW)
│   ├── software_layer.py        # Cluster ID → Tool mapping (NEW)
│   ├── argument_inference.py    # Argument generation (NEW)
│   ├── autoencoder.py           # Legacy autoencoder
│   ├── encoder.py               # Legacy encoder
│   ├── decoder.py               # Legacy decoder
│   └── tool_call_utils.py      # Parsing/validation utilities
├── training/
│   ├── config.py                # Configuration (updated for new architecture)
│   ├── data_generator.py        # Synthetic data generation
│   ├── losses.py                # Loss functions (Circle Loss added)
│   ├── train_intent_embedding.py    # Phase 1 training (NEW)
│   ├── train_cluster_retrieval.py   # Phase 2 training (NEW)
│   ├── train_autoencoder.py        # Legacy training
│   └── train_llm_integration.py     # Legacy training
├── evaluation/
│   └── metrics.py               # Evaluation metrics (cluster metrics added)
├── inference.py                  # Inference pipeline (updated for clusters)
├── NEWIDEA.md                   # Original design document
└── REFACTORING_SUMMARY.md       # Refactoring notes
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
class IntentEmbeddingConfig:
    # Model
    intent_embedding_dim: int = 1024  # High-dimensional intent space
    projection_dim: int = 128          # Projected space for similarity
    encoder_model: str = "google/flan-t5-base"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 30
    
    # Loss
    use_circle_loss: bool = True
    circle_loss_weight: float = 1.0
    circle_loss_margin: float = 0.25
    circle_loss_gamma: float = 256.0
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "ntilc"
```

### Training Phases

**Phase 1: Intent Embedding**
```bash
python -m training.train_intent_embedding
```

Trains:
- Intent embedder (1024-D embeddings)
- Projection head (128-D for similarity)
- Uses Circle Loss for metric learning

**Phase 2: Cluster Retrieval**
```bash
python -m training.train_cluster_retrieval
```

Trains:
- Query encoder (NL → 128-D)
- Uses frozen intent embedder and projection head
- Optimizes for cluster retrieval

### Wandb Logging

All experiments are fully reproducible via wandb:

```bash
# Training with wandb
python -m training.train_intent_embedding
python -m training.train_cluster_retrieval
```

Logged items:
- All hyperparameters and config
- Training/validation losses per step
- Cluster metrics (intra/inter-cluster similarity)
- Embedding statistics
- Model checkpoints as artifacts

## Inference

```python
from inference import ClusterBasedToolSystem

# Load system
system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt"
)

# Predict tool call
result = system.predict("Get the last 10 orders from California")
print(result.tool_name)      # "database_query"
print(result.arguments)      # {"sql": "...", "timeout": 30}
print(result.cluster_id)     # 2
print(result.confidence)      # 0.95
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `cluster_accuracy` | Correct cluster/tool selection |
| `intra_cluster_similarity` | Average similarity within clusters |
| `inter_cluster_similarity` | Average similarity between clusters |
| `cluster_separation` | Inter - intra cluster similarity |
| `silhouette_score` | Cluster quality metric |
| `embedding_mean_norm` | Average embedding L2 norm |
| `embedding_mean_variance` | Embedding variance (diversity) |

## Key Benefits of New Architecture

1. **No Decoder**: Faster inference (cluster lookup vs. autoregressive generation)
2. **Interpretable**: Cluster IDs are human-readable
3. **Extensible**: New tools can be added by optimizing a single embedding
4. **Separated Concerns**: Tool selection vs. argument generation
5. **Geometry-Optimized**: Metric learning (Circle Loss) for similarity

## Migration from Old Architecture

The old autoencoder-based architecture is still available in:
- `models/autoencoder.py`
- `training/train_autoencoder.py`
- `training/train_llm_integration.py`

See `REFACTORING_SUMMARY.md` for details on the migration.

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
faker>=18.0.0
tqdm>=4.65.0
scikit-learn>=1.0.0
```

## References

- `NEWIDEA.md` - Original design document
- `REFACTORING_SUMMARY.md` - Refactoring notes
- `REFACTORING_NOTES.md` - Detailed refactoring guide
