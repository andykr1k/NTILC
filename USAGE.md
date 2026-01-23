# NTILC Usage Guide (NEW ARCHITECTURE)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

**Phase 1: Intent Embedding**
```bash
python -m training.train_intent_embedding
```

This trains:
- Intent embedder (1024-D embeddings)
- Projection head (128-D for similarity)
- Uses Circle Loss for metric learning

**Phase 2: Cluster Retrieval**
```bash
python -m training.train_cluster_retrieval
```

This trains:
- Query encoder (NL → 128-D)
- Uses frozen models from Phase 1

### 3. Run Inference

```bash
python inference.py
```

Or in Python:
```python
from inference import ClusterBasedToolSystem

# Load system
system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt"
)

# Predict tool call
result = system.predict("Get the last 10 orders from California")
print(result)
```

## Training

### Intent Embedding Training

```bash
python -m training.train_intent_embedding
```

**Configuration** (in `training/config.py`):
- `intent_embedding_dim`: 1024 (high-dimensional intent space)
- `projection_dim`: 128 (projected space for similarity)
- `circle_loss_weight`: 1.0
- `circle_loss_margin`: 0.25
- `circle_loss_gamma`: 256.0

**Output:**
- `checkpoints/best_model.pt` - Intent embedder + projection head

### Cluster Retrieval Training

```bash
python -m training.train_cluster_retrieval
```

**Requirements:**
- Phase 1 checkpoint must exist at `checkpoints/best_model.pt`

**Output:**
- `checkpoints/cluster_retrieval/best_model.pt` - Query encoder + cluster centroids

## Inference

### Basic Usage

```python
from inference import ClusterBasedToolSystem

# Load system
system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt"
)

# Single prediction
result = system.predict("Find information about machine learning")
print(f"Tool: {result.tool_name}")
print(f"Arguments: {result.arguments}")
print(f"Cluster ID: {result.cluster_id}")
print(f"Confidence: {result.confidence}")

# Batch prediction
queries = [
    "What is 25 plus 37?",
    "Get the last 10 orders from California",
    "Send an email to test@example.com"
]
results = system.predict_batch(queries)
for result in results:
    print(result)
```

### Advanced Usage

```python
# Custom similarity threshold
result = system.predict(
    "Find AI papers",
    top_k=3,  # Return top 3 clusters
    similarity_threshold=0.6  # Minimum similarity
)

# Access cluster information
if result.cluster_id != -1:
    print(f"Retrieved cluster: {result.cluster_id}")
    print(f"Confidence: {result.confidence:.3f}")
else:
    print("No cluster found above threshold")
```

## Evaluation

### Cluster Metrics

```python
from evaluation.metrics import compute_cluster_metrics
import torch

# Get embeddings and labels
embeddings = ...  # (num_samples, 128)
labels = ...      # (num_samples,) cluster IDs
tool_calls = ...  # List of tool call strings

# Compute metrics
metrics = compute_cluster_metrics(embeddings, labels, tool_calls)

print(f"Cluster Accuracy: {metrics['cluster_accuracy']:.3f}")
print(f"Intra-cluster Similarity: {metrics['intra_cluster_similarity']:.3f}")
print(f"Inter-cluster Similarity: {metrics['inter_cluster_similarity']:.3f}")
print(f"Cluster Separation: {metrics['cluster_separation']:.3f}")
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
```

## Configuration

### Intent Embedding Config

Edit `training/config.py`:

```python
@dataclass
class IntentEmbeddingConfig:
    # Model
    intent_embedding_dim: int = 1024
    projection_dim: int = 128
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

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
cd /scratch4/home/akrik/NTILC

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Out of Memory

Reduce batch size in config:
```python
batch_size: int = 16  # or 8
gradient_accumulation_steps: int = 4  # Increase to compensate
```

### Cluster Retrieval Not Working

Make sure Phase 1 checkpoint exists:
```bash
ls checkpoints/best_model.pt
```

## Examples

### Example 1: Search Query

```python
result = system.predict("Find recent papers on machine learning")
# Tool: search
# Arguments: {"query": "machine learning", "max_results": 10}
# Cluster ID: 0
# Confidence: 0.92
```

### Example 2: Database Query

```python
result = system.predict("Get orders from California")
# Tool: database_query
# Arguments: {"sql": "SELECT * FROM orders WHERE state='CA'", "timeout": 30}
# Cluster ID: 2
# Confidence: 0.88
```

### Example 3: Calculation

```python
result = system.predict("What is 25 plus 37?")
# Tool: calculate
# Arguments: {"expression": "25 + 37"}
# Cluster ID: 1
# Confidence: 0.95
```

## Background Training

```bash
# Train in background
nohup python -m training.train_intent_embedding > logs/train_intent.log 2>&1 &

# Monitor progress
tail -f logs/train_intent.log

# Check wandb dashboard
# Visit https://wandb.ai/your-entity/ntilc
```

## Next Steps

See `NEXTSTEPS.md` for:
- Detailed training phases
- Success metrics
- Future improvements

See `README.md` for:
- Architecture overview
- Project structure
- Theoretical foundations
