# Registry

This directory is the canonical source of truth for public tool definitions,
examples, and model release metadata.

## Layout

```text
registry/
  categories.yaml
  models/
    releases.yaml
  tools/
    <tool-id>/
      tool.yaml
      examples.jsonl
  generated/
    tools.json
    models.json
    hierarchy.json
    tool_embedding_dataset.jsonl
    registry_manifest.json
```

## Workflow

1. Add or update a tool under `registry/tools/<tool-id>/`.
2. Run `python3 scripts/build_registry.py`.
3. Train from `registry/generated/tool_embedding_dataset.jsonl`.
4. Publish weights to your model host, then update `registry/models/releases.yaml`.
5. Re-run `python3 scripts/build_registry.py` so the website picks up the new release metadata.

## Tool Manifest

Each `tool.yaml` must include:

- `id`
- `display_name`
- `description`
- `interface_type`
- `source_repo`
- `license`
- `maintainers`
- `parent_category`
- `tags`
- `parameters`

The `parent_category` must match an entry in `registry/categories.yaml`.

## Examples

`examples.jsonl` is a newline-delimited JSON file. Each row must contain a
`query` field and can optionally include `split`, `language`, and `notes`.

Example:

```json
{"query":"Search recursively for CUDA references in this repo","split":"train","language":"en"}
```

## Training

Train every normal + hierarchical variant from the registry snapshot:

```bash
bash scripts/train_registry_embedding_spaces.sh
```

Or run the underlying trainers directly.

Normal embedding training:

```bash
python3 -m training.train_embedding_space \
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \
  --output-dir output/registry_embeddings \
  --loss-type contrastive
```

Hierarchical embedding training:

```bash
python3 -m training.train_hierarchical_embedding_space \
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \
  --hierarchy-path registry/generated/hierarchy.json \
  --output-dir output/registry_embeddings \
  --loss-type contrastive
```
