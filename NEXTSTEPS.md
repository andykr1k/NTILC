# Next Steps - NEW ARCHITECTURE

## ✅ Completed

### Architecture Refactoring
- **Created intent embedder** (1024-D embeddings)
- **Created projection head** (1024-D → 128-D)
- **Implemented Circle Loss** for metric learning
- **Created cluster retrieval system** (replaces decoder)
- **Created software layer** (cluster ID → tool mapping)
- **Created argument inference system** (separate from tool selection)
- **Rewrote training scripts** for new architecture
- **Updated inference pipeline** for cluster-based retrieval
- **Updated evaluation metrics** for cluster-based metrics
- **Updated documentation** (README, USAGE, NEXTSTEPS)

## 🚀 Current Phase

### Phase 1: Intent Embedding Training

Train the intent embedder and projection head:

```bash
cd /scratch4/home/akrik/NTILC
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

## 📋 Next Steps

### Phase 2: Cluster Retrieval Training

After Phase 1 achieves good cluster formation:

```bash
python -m training.train_cluster_retrieval
```

**What it does:**
- Trains query encoder to map NL queries to 128-D embeddings
- Uses frozen intent embedder and projection head from Phase 1
- Optimizes for cluster retrieval accuracy
- Computes cluster centroids

**Target metrics:**
- Cluster retrieval accuracy > 85%
- Average similarity to target > 0.9

### Phase 3: End-to-End Evaluation

```bash
python inference.py
```

Or in Python:
```python
from inference import ClusterBasedToolSystem

system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt"
)

result = system.predict("Get the last 10 orders from California")
print(result.tool_name)      # "database_query"
print(result.arguments)       # {"sql": "...", "timeout": 30}
print(result.cluster_id)      # 2
print(result.confidence)      # 0.95
```

### Phase 4: Argument Inference Enhancement

Current argument inference uses simple extraction. Future improvements:
- Train argument necessity classifier
- Train argument value generator (autoregressive/diffusion)
- Support for continuous values (coordinates, layouts)

### Phase 5: Multi-Tool Planning

Extend to support:
- Multiple cluster retrieval (top-k)
- Multi-tool workflows
- Tool composition graphs

## 📊 Architecture Overview (NEW)

```
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

Inference:
───────────────────────────
Query → Query Encoder → 128-D → Similarity → Cluster ID
                                                      │
                                                      ▼
Cluster ID → Software Layer → Tool
Query → Argument Inference → Arguments
```

## 📈 Success Metrics

### Phase 1 (Intent Embedding)
- **Minimum:** Intra-cluster similarity > 0.7, separation > 0.3
- **Target:** Intra-cluster similarity > 0.8, separation > 0.5
- **Stretch:** Intra-cluster similarity > 0.9, separation > 0.7

### Phase 2 (Cluster Retrieval)
- **Minimum:** Cluster accuracy > 80%, avg similarity > 0.85
- **Target:** Cluster accuracy > 85%, avg similarity > 0.9
- **Stretch:** Cluster accuracy > 90%, avg similarity > 0.95

### End-to-End
- Latency: <50ms per prediction (faster than decoder)
- Successful NL → Tool Call for diverse queries
- Interpretable cluster IDs

## 🚀 Quick Commands

```bash
# Train intent embedder (Phase 1)
python -m training.train_intent_embedding

# Train cluster retrieval (Phase 2)
python -m training.train_cluster_retrieval

# Run inference
python inference.py

# Evaluate clusters
python -c "from evaluation.metrics import compute_cluster_metrics; ..."
```

## 🔄 Migration Notes

The old autoencoder architecture is still available but deprecated:
- `training/train_autoencoder.py` - Old Phase 1
- `training/train_llm_integration.py` - Old Phase 2

New architecture files:
- `training/train_intent_embedding.py` - New Phase 1
- `training/train_cluster_retrieval.py` - New Phase 2

See `REFACTORING_SUMMARY.md` for details.
