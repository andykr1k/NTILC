# NTILC

Neural Tool Invocation via Learned Compression for Linux tool-calling.

NTILC replaces direct prompt-to-tool-call generation with a staged controller architecture that is easier to debug, faster to route, and less brittle than text-only tool invocation pipelines.

## Why This Architecture

Compared to prompt-only tool calling, this repository is built to address the main failure modes highlighted in the paper plan:

- decoding-heavy inference that increases latency
- parsing failures from free-form generations
- no explicit tool-calling stage between query and answer
- context rot when tool registries become large

The system therefore separates planning, retrieval, argument filling, and dispatch into explicit blocks.

## Method (Paper-Aligned)

This README is organized to match the paper method structure:

1. Qwen + LoRA (planner and argument filler)
2. Tool Embeddings (intent geometry + retrieval)
3. Argument Generation (deterministic + model-assisted filling)
4. Orchestration (plan, dispatch, response, retry)

## Canonical Inference Flow

This section is the canonical inference flow for the repository.

### Step-by-step

1. User query enters Qwen (optionally with LoRA adapter).
2. Model emits a `<plan>` block with atomic actions.
3. Controller pauses at `</plan>` and processes each action.
4. Each action is embedded and sent to tool embedding retrieval.
5. Retrieval returns `top_k` cluster candidates `(cluster_id, score)`.
6. Tool registry maps clusters to tools and schemas.
7. Argument generator is conditioned on query + intent + selected tool metadata.
8. Controller emits `<dispatch>` block.
9. Dispatcher validates and optionally executes the call.
10. Dispatcher emits `<response>` block.
11. Controller either retries next candidate or returns answer/results.

### Plan Block

```xml
<plan>
  <action><len:10>list files</len></action>
  <action><len:12>search cuda</len></action>
</plan>
```

### Dispatch Block

```xml
<dispatch>
  <tool><len:4>grep</len></tool>
  <arg name="opt"><len:2>-R</len></arg>
  <arg name="pattern"><len:6>"cuda"</len></arg>
  <arg name="path"><len:1>.</len></arg>
</dispatch>
```

### Response Block

```xml
<response>
  <tool><len:4>grep</len></tool>
  <status><len:4>fail</len></status>
  <text><len:9>it failed</len></text>
  <retry><len:4>true</len></retry>
</response>
```

## Canonical Workflow

This section is the canonical training/analysis workflow for the repository.

### 1) Build cleaned datasets

```bash
bash scripts/cleanPairs.sh
```

Outputs:

- `data/man/nl_command_pairs_cleaned_v2.json(.l)`
- `data/man/nl_command_pairs_flat_clean_v2.json(.l)`
- `data/man/nl_command_pairs_flat_train_v2.json(.l)`

### 2) Train Phase 1 (intent embeddings)

```bash
bash scripts/trainIE.sh
```

Checkpoint:

- `checkpoints/intent_embedder/best_model.pt`

### 3) Train Phase 2 (cluster retrieval)

```bash
bash scripts/trainCR.sh
```

Checkpoint:

- `checkpoints/cluster_retrieval/best_model.pt`

### 4) Train LoRA command model

```bash
bash scripts/trainLora.sh
```

Checkpoint:

- `checkpoints/lora_nl_command_full/`

### 5) Test LoRA

```bash
bash scripts/testLora.sh
```

### 6) Analyze

Primary notebooks:

- `notebooks/embedding_space_analysis_v2.ipynb`
- `notebooks/retrieval_and_lora_analysis_v2.ipynb`
- `notebooks/full_inference_pipeline.ipynb`

Workflow notes:

- `source_url` is metadata for traceability/splits/error analysis, not a model feature.
- Inference uses `tool_names` saved in retrieval checkpoints for mapper initialization when available.

## Dataset

NTILC uses man-page-derived NL-command pairs.

Pipeline:

1. scrape/generate command-oriented examples from man-page metadata
2. clean and validate examples (`cleanPairs.sh`)
3. train on flattened records from `data/man/nl_command_pairs_flat_train_v2.jsonl`

This repository is now NL-command-pairs only (no synthetic data generator path).

## Training Components

### Tool Embeddings

- `training/train_intent_embedding.py`
- learns intent embedding geometry for tools/commands
- outputs phase-1 checkpoint consumed by retrieval training

### Cluster Retrieval

- `training/train_cluster_retrieval.py`
- trains query encoder to retrieve correct cluster/tool IDs
- stores cluster centroids + tool-name mapping

### LoRA (Optional)

- `training/train_lora_nl_command.py`
- supports `full` and `tail` command generation modes
- can be used for planner/argument-filler specialization

## Inference and Orchestration

### Direct retrieval inference

```python
from inference import ClusterBasedToolSystem

system = ClusterBasedToolSystem.from_pretrained(
    intent_embedder_path="checkpoints/intent_embedder/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt",
)

result = system.predict("search recursively for cuda in current dir", top_k=3)
print(result)
```

### Full orchestrator run

```python
from orchestrator.agent import NTILCOrchestratorAgent

agent = NTILCOrchestratorAgent.from_pretrained(
    intent_embedder_path="checkpoints/intent_embedder/best_model.pt",
    query_encoder_path="checkpoints/cluster_retrieval/best_model.pt",
    qwen_model_name_or_path="Qwen/Qwen3.5-9B",
    # lora_adapter_path="checkpoints/lora_nl_command_full",
)

run = agent.run(
    request="Find all cuda references and summarize top files",
    execute_tools=False,
    top_k_candidates=3,
    max_retries=2,
)

print(run.plan_block)
print(run.atomic_actions)
print(run.action_failures)
for step in run.steps:
    print(step.dispatch_block)
    print(step.response_block)
```

## Execution Layer and Safety

`models/software_layer.py` handles:

- cluster-level and tool-level callable registration
- argument validation against schema
- safety rule enforcement
- optional permission checks
- dispatch via `ToolDispatcher`

Example mapper + dispatcher wiring:

```python
from models.software_layer import ClusterToolMapper, ToolDispatcher

mapper = ClusterToolMapper.from_retrieval_checkpoint(
    "checkpoints/cluster_retrieval/best_model.pt"
)
mapper.register_shell_tools_for_all_clusters(timeout_seconds=20, cwd=".")

dispatcher = ToolDispatcher(mapper=mapper, fail_on_nonzero_exit=True)
result = dispatcher.dispatch_cluster(
    cluster_id=0,
    arguments={"command": "ls -la"},
    execute=False,  # dry-run
)
print(result.to_dict())
```

Example safety-rule formats:

- `requires_permission:filesystem.read`
- `forbid_arg:path`
- `forbid_value:mode=unsafe`
- `non_empty:query`
- `regex:path:^/safe/`

## Experiments and Analysis

This aligns with the paper plan’s experiments/results framing:

- retrieval quality and cluster behavior
- plan/dispatch/response block behavior
- LoRA command generation quality
- failure-case and retry-path inspection

Use:

- `bash scripts/testLora.sh`
- notebooks under `notebooks/` for embedding/retrieval analysis

## Setup

```bash
pip install -r requirements.txt
```
