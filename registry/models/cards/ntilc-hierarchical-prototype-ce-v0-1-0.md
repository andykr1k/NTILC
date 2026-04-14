# NTILC Hierarchical Prototype-CE v0.1.0

NTILC tool-embedding checkpoint prepared for `OpenToolEmbeddings/ntilc-hierarchical-prototype-ce-v0-1-0`.

## Variant

- Architecture: `hierarchical`
- Loss: `prototype_ce`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `128`
- Dataset version: `oss-qwen3.5-27b-26tools-312queries-v0.1.0`
- Synthetic generator: `Qwen/Qwen3.5-27B`
- Tool count: `26`
- Query count: `312`
- Checkpoint SHA256: `7eae6130003a091bef3c81d1d084b35d137974d0f4f5bc2fd61bd43684cc446f`

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
- `best_epoch`: `19`
- `tool_count`: `26`
- `query_count`: `312`
- `best_val_tool_retrieval_accuracy`: `0.7885`
- `best_val_parent_retrieval_accuracy`: `0.9038`
- `best_val_tool_classification_accuracy`: `0.7692`
- `best_val_parent_classification_accuracy`: `0.9038`
- `tool_silhouette_score`: `0.0825`
- `parent_silhouette_score`: `0.069`
