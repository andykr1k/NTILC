# NTILC Normal Circle v0.1.0

NTILC tool-embedding checkpoint prepared for `OpenToolEmbeddings/ntilc-normal-circle-v0-1-0`.

## Variant

- Architecture: `normal`
- Loss: `circle`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `128`
- Dataset version: `oss-qwen3.5-27b-26tools-312queries-v0.1.0`
- Synthetic generator: `Qwen/Qwen3.5-27B`
- Tool count: `26`
- Query count: `312`
- Checkpoint SHA256: `a1cdae6bf2266e1bacd0adb63ca51fe63c8678f50c75b0eae56e221d632de705`

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
- `best_epoch`: `23`
- `tool_count`: `26`
- `query_count`: `312`
- `best_val_retrieval_accuracy`: `0.8077`
- `best_val_classification_accuracy`: `0.7885`
- `silhouette_score`: `0.1559`
