# NTILC: Neural Tool Invocation via Learned Compression

**A novel approach to language model tool use through cluster-based retrieval instead of text generation.**

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Supported Tools](#supported-tools)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Current Status](#current-status)
- [Future Directions](#future-directions)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

---

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

| Aspect | Text-Based | NTILC |
|--------|-----------|-------|
| Generation | O(n) tokens | O(1) cluster lookup |
| Speed | Slow (autoregressive) | Fast (similarity) |
| Errors | Parsing failures | Cluster mismatch |
| Interpretability | Low | High (cluster IDs) |
| Extensibility | Retrain all | Add single embedding |

### Key Benefits

1. **No Decoder**: Faster inference (cluster lookup vs. autoregressive generation)
2. **Interpretable**: Cluster IDs are human-readable
3. **Extensible**: New tools can be added by optimizing a single embedding
4. **Separated Concerns**: Tool selection vs. argument generation
5. **Geometry-Optimized**: Metric learning (Circle Loss) for similarity
6. **Preserves Debuggability**: Clear mapping from cluster ID to tool
7. **Matches Real-World Usage**: Geometry aligns with actual tool usage patterns

**Tool Call Format:** NTILC uses Python function-call strings (e.g., `search(query='...', max_results=10)`) rather than JSON.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train intent embedder (Phase 1)
python -m training.train_intent_embedding

# 3. Train cluster retrieval (Phase 2 - after Phase 1 converges)
python -m training.train_cluster_retrieval

# 4. Run inference
python inference.py
```

### Basic Python Usage

```python
from inference import ClusterBasedToolSystem

# Load system
system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt"
)

# Predict tool call
result = system.predict("Get the last 10 orders from California")
print(f"Tool: {result.tool_name}")           # "database_query"
print(f"Arguments: {result.arguments}")      # {"sql": "...", "timeout": 30}
print(f"Tool Call: {result.tool_call}")      # "database_query(sql='...', timeout=30)"
print(f"Cluster ID: {result.cluster_id}")    # 2
print(f"Confidence: {result.confidence}")    # 0.95
```

---

## Architecture

### High-Level System Flow

```text
User Query → Intent Embedder → 1024-D → Projection Head → 128-D
                                                              ↓
                                                    Cluster Retrieval
                                                              ↓
                                                         Cluster ID
                                                              ↓
                                                      Software Layer
                                                              ↓
                                                            Tool
                                                              +
Query → Argument Inference → Arguments → Tool Execution
```

### Component Breakdown

#### 1. Tool Intent Embedding Space (1024-D)

**Purpose:** Encode semantic intent behind tool usage.

Each tool is represented as a *canonicalized intent object*, including:
- Tool name
- Tool description
- Argument schema
- Example calls
- Natural language paraphrases

These are embedded into a high-dimensional space (1024-D) to preserve semantic richness.

**Adding new tools:**
- New tools can be inserted via a **key vector** in the frozen space
- Only the new vector is trained using Circle Loss; existing clusters remain untouched

#### 2. Projection Head (1024-D → 128-D)

**Purpose:** Map embeddings to a geometry-friendly space for similarity computation.

- Maps high-dimensional embeddings into a 128-D space
- Optimized with contrastive and Circle Loss objectives
- Not used for storage—only for similarity and loss computation

#### 3. Metric Learning with Circle Loss

**Purpose:** Form soft clusters of tool usage.

Circle Loss is used to:
- Pull semantically equivalent tool intents together
- Push unrelated intents apart
- Preserve angular margins between clusters

**Key properties:**
- Clusters are **soft**, not mutually exclusive
- Hard positives/negatives drive the embedding
- New tools can be added by optimizing a **single embedding** without retraining existing ones

#### 4. Query Inference (Cluster Retrieval)

At inference:

1. User query → embed in 1024-D space → project to 128-D
2. Compute similarity against all cluster centroids or prototypes
3. **Select top cluster(s)** → return **cluster ID(s)**

> No decoder or autoregressive generation is used. The cluster ID directly indexes the software layer.

**Output:**
- Cluster ID(s)
- Similarity / confidence score

#### 5. Cluster-to-Tool Mapping Layer (Software Layer)

**Purpose:** Decouple model from execution.

- Maps cluster IDs to one or more tools
- Maintains versioning, safety rules, and permissions
- Handles tool indices, arguments, and any overlying business logic

> All tool execution is handled here — the model only retrieves the correct cluster.

#### 6. Argument Inference

Arguments are handled **separately from tool selection**:

**Route A: Argument Necessity Detection**
- For each candidate tool, classify arguments as required, optional, or irrelevant

**Route B: Argument Value Generation**
- Only relevant arguments are generated or extracted
- Methods:
  - Deterministic extraction (IDs, strings)
  - Autoregressive generation for enums or short text
  - Diffusion or continuous-value generation for coordinates, layouts, or latent parameters

Arguments marked irrelevant are set to `null`.

#### 7. Null / Abstention Path

Explicit **no-tool / clarification** route:
- Triggered if all cluster similarities fall below a threshold or argument necessity is ambiguous
- Outcome: no tool invoked; system requests clarification

### Training Phases

```text
Phase 1: Intent Embedding
─────────────────────────
Tool Intent → Intent Embedder (T5) → 1024-D → Projection Head → 128-D
                │                              │
                │    Circle Loss + Contrastive │
                │◄─────────────────────────────►
                │                              │
                └──────────────────────────────┘
                          Cluster Formation

Phase 2: Cluster Retrieval
───────────────────────────
NL Query → Query Encoder (T5) → 128-D → Cluster Retrieval → Cluster ID
              │                              │
              │     MSE + Cosine Loss        │
              │◄─────────────────────────────┤
              │                              │
              │                              ▼
              │                    Cluster Centroids (frozen)
              │                              │
              └──────────────────────────────┘
```

---

## Project Structure

```
NTILC/
├── models/
│   ├── intent_embedder.py      # Tool intent → 1024-D embeddings
│   ├── projection_head.py      # 1024-D → 128-D projection
│   ├── cluster_retrieval.py    # Cluster ID retrieval
│   ├── query_encoder.py        # NL query → 128-D embedding
│   ├── software_layer.py       # Cluster ID → Tool mapping
│   ├── argument_inference.py   # Argument generation
│   ├── tool_call_utils.py      # Parsing/validation utilities
│   └── tool_schemas.py         # Tool definitions and schemas
├── training/
│   ├── config.py               # Configuration
│   ├── data_generator.py       # Synthetic data generation
│   ├── losses.py               # Loss functions (Circle Loss)
│   ├── train_intent_embedding.py    # Phase 1 training
│   └── train_cluster_retrieval.py   # Phase 2 training
├── evaluation/
│   ├── metrics.py              # Evaluation metrics
│   └── visualizations.py       # Embedding visualizations
├── inference.py                # Inference pipeline
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

---

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

---

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

### Phase 1: Intent Embedding Training

Train the intent embedder and projection head:

```bash
python -m training.train_intent_embedding
```

**What it does:**
- Embeds tool intents to 1024-D space
- Projects to 128-D for similarity computation
- Uses Circle Loss for metric learning
- Forms soft clusters of tool usage patterns

**Key features:**
- Circle Loss for robust cluster formation
- Contrastive loss (optional) for embedding diversity
- Embedding regularization (L2 + variance)
- Full wandb logging

**Target metrics:**
- Intra-cluster similarity > 0.8
- Inter-cluster similarity < 0.3
- Cluster separation > 0.5

**Output:**
- `checkpoints/best_model.pt` - Intent embedder + projection head

### Phase 2: Cluster Retrieval Training

After Phase 1 achieves good cluster formation:

```bash
python -m training.train_cluster_retrieval
```

**Requirements:**
- Phase 1 checkpoint must exist at `checkpoints/best_model.pt`

**What it does:**
- Trains query encoder to map NL queries to 128-D embeddings
- Uses frozen intent embedder and projection head from Phase 1
- Optimizes for cluster retrieval accuracy
- Computes cluster centroids

**Target metrics:**
- Cluster retrieval accuracy > 85%
- Average similarity to target > 0.9

**Output:**
- `checkpoints/cluster_retrieval/best_model.pt` - Query encoder + cluster centroids

### Data Requirements

- Synthetic and real tool invocation examples
- Paraphrased intents per tool
- Positive and negative intent pairs
- Argument relevance labels

### Training Objectives

- **Contrastive loss** for global embedding structure
- **Circle loss** for cluster geometry and new tool injection
- Optional auxiliary losses for argument necessity

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

### Background Training

```bash
# Train in background
nohup python -m training.train_intent_embedding > logs/train_intent.log 2>&1 &

# Monitor progress
tail -f logs/train_intent.log

# Check wandb dashboard
# Visit https://wandb.ai/your-entity/ntilc
```

---

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

### Command Line

```bash
python inference.py
```

### Example Outputs

**Example 1: Search Query**
```python
result = system.predict("Find recent papers on machine learning")
# Tool: search
# Arguments: {"query": "machine learning", "max_results": 10}
# Cluster ID: 0
# Confidence: 0.92
```

**Example 2: Database Query**
```python
result = system.predict("Get orders from California")
# Tool: database_query
# Arguments: {"sql": "SELECT * FROM orders WHERE state='CA'", "timeout": 30}
# Cluster ID: 2
# Confidence: 0.88
```

**Example 3: Calculation**
```python
result = system.predict("What is 25 plus 37?")
# Tool: calculate
# Arguments: {"expression": "25 + 37"}
# Cluster ID: 1
# Confidence: 0.95
```

---

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

### Available Metrics

| Metric | Description |
|--------|-------------|
| `cluster_accuracy` | Correct cluster/tool selection |
| `intra_cluster_similarity` | Average similarity within clusters |
| `inter_cluster_similarity` | Average similarity between clusters |
| `cluster_separation` | Inter - intra cluster similarity |
| `silhouette_score` | Cluster quality metric |
| `embedding_mean_norm` | Average embedding L2 norm |
| `embedding_mean_variance` | Embedding variance (diversity) |

### Success Metrics

**Phase 1 (Intent Embedding)**
- **Minimum:** Intra-cluster similarity > 0.7, separation > 0.3
- **Target:** Intra-cluster similarity > 0.8, separation > 0.5
- **Stretch:** Intra-cluster similarity > 0.9, separation > 0.7

**Phase 2 (Cluster Retrieval)**
- **Minimum:** Cluster accuracy > 80%, avg similarity > 0.85
- **Target:** Cluster accuracy > 85%, avg similarity > 0.9
- **Stretch:** Cluster accuracy > 90%, avg similarity > 0.95

**End-to-End**
- Latency: <50ms per prediction (faster than decoder)
- Successful NL → Tool Call for diverse queries
- Interpretable cluster IDs

---

## Configuration

### Customizing Hyperparameters

Edit `training/config.py`:

```python
@dataclass
class IntentEmbeddingConfig:
    # Model dimensions
    intent_embedding_dim: int = 1024
    projection_dim: int = 128
    encoder_model: str = "google/flan-t5-base"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 30
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    
    # Circle Loss parameters
    use_circle_loss: bool = True
    circle_loss_weight: float = 1.0
    circle_loss_margin: float = 0.25
    circle_loss_gamma: float = 256.0
    
    # Regularization
    l2_reg_weight: float = 0.01
    variance_reg_weight: float = 0.001
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "ntilc"
    log_interval: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
```

### CUDA Out of Memory

Reduce batch size in config:
```python
batch_size: int = 16  # or 8
gradient_accumulation_steps: int = 4  # Increase to compensate
```

---

## Current Status

### ✅ Completed

**Architecture Refactoring**
- Created intent embedder (1024-D embeddings)
- Created projection head (1024-D → 128-D)
- Implemented Circle Loss for metric learning
- Created cluster retrieval system (replaces decoder)
- Created software layer (cluster ID → tool mapping)
- Created argument inference system (separate from tool selection)
- Rewrote training scripts for new architecture
- Updated inference pipeline for cluster-based retrieval
- Updated evaluation metrics for cluster-based metrics
- Updated documentation

### 🚀 Current Phase

**Phase 1: Intent Embedding Training**

Train the intent embedder and projection head to form robust clusters of tool usage patterns.

### 📋 Next Steps

1. **Complete Phase 1 Training**
   - Achieve target cluster metrics
   - Save best checkpoint

2. **Phase 2: Cluster Retrieval Training**
   - Train query encoder
   - Optimize for cluster retrieval accuracy

3. **End-to-End Evaluation**
   - Test on diverse NL queries
   - Measure latency and accuracy

4. **Argument Inference Enhancement**
   - Train argument necessity classifier
   - Train argument value generator
   - Support continuous values

5. **Multi-Tool Planning**
   - Extend to support top-k clusters
   - Multi-tool workflows
   - Tool composition graphs

---

## Future Directions

### Near-Term Improvements

- **Argument Inference Enhancement**
  - Train argument necessity classifier
  - Train argument value generator (autoregressive/diffusion)
  - Support for continuous values (coordinates, layouts)

- **Multi-Tool Planning**
  - Multiple cluster retrieval (top-k)
  - Multi-tool workflows
  - Tool composition graphs

### Long-Term Vision

- **Dynamic Cluster Creation**
  - Online adaptation of cluster centroids
  - Automatic cluster splitting/merging

- **Tool Composition**
  - Composite tools from primitives
  - Dependency-aware execution graphs

- **Reinforcement Learning**
  - Learn from execution success
  - Optimize for task completion

- **Extended Tool Support**
  - Continuous integration of new tools
  - Zero-shot tool adaptation

### All Possible Execution Routes

1. Single confident cluster → single tool → deterministic argument generation
2. Multiple clusters → arbitration → best tool
3. Single tool → partial arguments → request clarification
4. Single tool → missing continuous arguments → diffusion generation
5. Low similarity → no tool → natural language response
6. Overlapping clusters → multi-tool plan (future extension)

---

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
cd /path/to/NTILC

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Cluster Retrieval Not Working

Make sure Phase 1 checkpoint exists:
```bash
ls checkpoints/best_model.pt
```

If missing, train Phase 1 first:
```bash
python -m training.train_intent_embedding
```

### Low Cluster Quality

If cluster metrics are poor:
- Increase `num_epochs` in config
- Tune Circle Loss hyperparameters (`margin`, `gamma`)
- Increase `circle_loss_weight`
- Check data quality (paraphrases, examples)

### Slow Training

- Reduce `batch_size`
- Use gradient accumulation
- Enable mixed precision training
- Use smaller encoder model

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
wandb>=0.15.0
faker>=18.0.0
tqdm>=4.65.0
scikit-learn>=1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick Commands Reference

```bash
# Train intent embedder (Phase 1)
python -m training.train_intent_embedding

# Train cluster retrieval (Phase 2)
python -m training.train_cluster_retrieval

# Run inference
python inference.py

# Background training
nohup python -m training.train_intent_embedding > logs/train.log 2>&1 &

# Monitor logs
tail -f logs/train.log
```

---
