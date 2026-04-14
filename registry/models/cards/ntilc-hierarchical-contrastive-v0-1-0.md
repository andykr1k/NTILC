# NTILC Hierarchical Contrastive v0.1.0

NTILC tool-embedding checkpoint prepared for `OpenToolEmbeddings/ntilc-hierarchical-contrastive-v0-1-0`.

## Variant

- Architecture: `hierarchical`
- Loss: `contrastive`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `128`
- Dataset version: `oss-qwen3.5-27b-26tools-312queries-v0.1.0`
- Synthetic generator: `Qwen/Qwen3.5-27B`
- Tool count: `26`
- Query count: `312`
- Checkpoint SHA256: `6b1be4ecd2535992ad1ba99505597df360ac5dbe4dcf6dda237a370951092363`

## Files

- `best.pt`: best validation checkpoint bundle
- `metrics.json`: per-epoch training and validation metrics

## Usage

These are NTILC checkpoint bundles, not plain `transformers` repositories. Load them with the local helper below.

```python
from training import load_checkpoint_bundle

bundle = load_checkpoint_bundle("best.pt", device="cpu")
print(bundle["architecture"], bundle["loss_name"], len(bundle["tool_names"]))
```

## Metrics

- `epochs`: `25`
- `best_epoch`: `12`
- `tool_count`: `26`
- `query_count`: `312`
- `best_val_tool_retrieval_accuracy`: `0.8077`
- `best_val_parent_retrieval_accuracy`: `0.8654`
- `best_val_tool_classification_accuracy`: `0.5962`
- `best_val_parent_classification_accuracy`: `0.9038`
- `tool_silhouette_score`: `0.0096`
- `parent_silhouette_score`: `0.2732`
