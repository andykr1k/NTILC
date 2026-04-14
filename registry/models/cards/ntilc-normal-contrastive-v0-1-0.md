# NTILC Normal Contrastive v0.1.0

NTILC tool-embedding checkpoint prepared for `OpenToolEmbeddings/ntilc-normal-contrastive-v0-1-0`.

## Variant

- Architecture: `normal`
- Loss: `contrastive`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `128`
- Dataset version: `oss-qwen3.5-27b-26tools-312queries-v0.1.0`
- Synthetic generator: `Qwen/Qwen3.5-27B`
- Tool count: `26`
- Query count: `312`
- Checkpoint SHA256: `00523d995468866ebdc51f23e7f5ed2f12fce8e09437272393c049aee6b98ef8`

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
- `best_epoch`: `25`
- `tool_count`: `26`
- `query_count`: `312`
- `best_val_retrieval_accuracy`: `0.7692`
- `best_val_classification_accuracy`: `0.8077`
- `silhouette_score`: `0.1581`
