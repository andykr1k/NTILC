# NTILC

NTILC stands for **Neural Tool Invocation via Learned Compression**.

This repository is the main research codebase for a staged tool-calling system:
planner-style language modeling, embedding-based tool retrieval, schema-grounded
argument generation, and controller-managed execution. It also contains the
public registry, release pipeline, and website for **Open Tool Embeddings**,
which is the open-source surface of that work.

The right way to read this repo is:

1. **Research first**: NTILC is an architecture for making tool invocation more
   structured, inspectable, and debuggable than direct prompt-to-tool-call
   generation.
2. **Open infrastructure second**: the registry, releases, and site are how that
   research is being pushed toward an open, reusable foundation layer for tool
   embeddings.

## Research Focus

NTILC is built around a simple claim: tool use should not be treated as one
monolithic text-generation problem.

Instead of asking a model to go directly from user request to tool name to
arguments to final answer in one free-form decode, this repo splits the problem
into explicit stages:

- planning and control
- tool retrieval in an embedding space
- schema-constrained argument generation
- execution and response handling

That separation matters for research because it makes failure modes visible.
You can inspect search queries, selected tools, argument payloads, controller
observations, retries, and tool outputs instead of treating the whole system as
an opaque generation trace.

## Current Architecture

The current runtime lives primarily in [agent/runtime.py](/scratch4/home/akrik/NTILC/agent/runtime.py)
and [agent/protocol.py](/scratch4/home/akrik/NTILC/agent/protocol.py).

### 1. Controller-Led Tool Use

The assistant does not call tools directly. It emits a strict XML control
protocol:

- `<search_tools>` to describe the capability it needs
- `<select_tool>` to choose from retrieved candidates
- `<dispatch>` after the controller returns the exact schema
- `<response>` from the controller back to the model after tool execution

This keeps the language model inside a narrow, inspectable interface instead of
letting it invent tool schemas or call formats.

### 2. Embedding-Based Tool Retrieval

The retrieval layer uses learned tool embedding spaces rather than plain string
matching over tool names.

Two model families are trained in this repo:

- **normal** tool embedding models
- **hierarchical** tool embedding models that explicitly use parent categories

Each family supports four losses:

- `prototype_ce`
- `contrastive`
- `circle`
- `functional_margin`

The runtime loads a trained checkpoint bundle with
`training.load_checkpoint_bundle(...)`, embeds the incoming search request, and
retrieves top-k tool candidates by similarity to learned tool centroids.

### 3. Schema-Grounded Argument Generation

Once the controller has selected a tool, the model does not improvise argument
keys. It is conditioned on the exact tool schema and uses structured generation
to produce only the allowed argument fields.

That makes argument generation a separate, measurable subproblem rather than a
side effect of unconstrained response generation.

### 4. Execution Layer

The execution layer uses the tool catalog plus the executable implementations in
[REPL/tools.py](/scratch4/home/akrik/NTILC/REPL/tools.py).

The runtime checks that:

- tool names in the checkpoint match the tool catalog
- tool names in the catalog match executable tools
- controller steps stay within the expected protocol

This keeps retrieval, schema, and execution aligned.

## Main Code Paths

The repo has changed enough that the old phase-1 / phase-2 README is no longer
the right mental model. The most important active surfaces are:

- [agent/runtime.py](/scratch4/home/akrik/NTILC/agent/runtime.py): controller, retriever, model adapter, runtime assembly
- [agent/agent.py](/scratch4/home/akrik/NTILC/agent/agent.py): Streamlit research UI for running the agent and inspecting events/stats
- [agent/protocol.py](/scratch4/home/akrik/NTILC/agent/protocol.py): XML control block parsing and serialization
- [training/train_embedding_space.py](/scratch4/home/akrik/NTILC/training/train_embedding_space.py): normal tool embedding training
- [training/train_hierarchical_embedding_space.py](/scratch4/home/akrik/NTILC/training/train_hierarchical_embedding_space.py): hierarchy-aware tool embedding training
- [training/wandb_diagnostics.py](/scratch4/home/akrik/NTILC/training/wandb_diagnostics.py): evaluation diagnostics and logging
- [registry/](/scratch4/home/akrik/NTILC/registry): public tool definitions, examples, model releases, and generated artifacts
- [scripts/import_oss_registry.py](/scratch4/home/akrik/NTILC/scripts/import_oss_registry.py): import OSS tool data into registry manifests
- [scripts/build_registry.py](/scratch4/home/akrik/NTILC/scripts/build_registry.py): compile registry YAML into generated JSON/JSONL
- [scripts/sync_model_releases.py](/scratch4/home/akrik/NTILC/scripts/sync_model_releases.py): compute checksums, summarize metrics, generate model cards
- [scripts/publish_huggingface_models.py](/scratch4/home/akrik/NTILC/scripts/publish_huggingface_models.py): publish prepared releases to Hugging Face
- [site/](/scratch4/home/akrik/NTILC/site): Next.js site for the public registry and model release surface

## Running the Research System

### Python Environment

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

The runtime expects:

- a Qwen model checkpoint available to `transformers`
- a tool catalog
- a trained tool embedding checkpoint

The current runtime defaults are:

- tool catalog: `data/OSS/tools.json`
- embedding checkpoint: `data/OSS/output/normal/circle/best.pt`
- model family: `Qwen/Qwen3.5-27B`

### Launch the Agent UI

The main interactive entry point is the Streamlit app:

```bash
streamlit run agent/agent.py
```

That UI exposes:

- runtime configuration
- live controller events
- retrieved tools
- tool responses
- per-turn stats and token accounting

### Notes

- The default runtime config is GPU-oriented.
- If your local devices differ, change them in the sidebar or update the
  runtime config.
- The runtime will fail fast if the tool catalog, embedding checkpoint, and
  executable tool registry disagree.

## Training Workflow

The current embedding work is driven by the registry snapshot, not by the older
phase-1 / phase-2 training flow described in the previous README.

### 1. Import or update the public tool registry

```bash
python3 scripts/import_oss_registry.py
python3 scripts/build_registry.py
```

This produces:

- `registry/generated/tools.json`
- `registry/generated/models.json`
- `registry/generated/hierarchy.json`
- `registry/generated/tool_embedding_dataset.jsonl`
- `registry/generated/registry_manifest.json`

### 2. Train every release variant

```bash
bash scripts/train_registry_embedding_spaces.sh
```

That wrapper trains:

- `normal/prototype_ce`
- `normal/contrastive`
- `normal/circle`
- `normal/functional_margin`
- `hierarchical/prototype_ce`
- `hierarchical/contrastive`
- `hierarchical/circle`
- `hierarchical/functional_margin`

### 3. Or run the trainers directly

Normal:

```bash
python3 -m training.train_embedding_space \
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \
  --output-dir output/registry_embeddings \
  --loss-type contrastive
```

Hierarchical:

```bash
python3 -m training.train_hierarchical_embedding_space \
  --dataset-path registry/generated/tool_embedding_dataset.jsonl \
  --hierarchy-path registry/generated/hierarchy.json \
  --output-dir output/registry_embeddings \
  --loss-type contrastive
```

### 4. Sync release metadata

```bash
python3 scripts/sync_model_releases.py
```

This script:

- computes checkpoint hashes
- summarizes the best metrics from local runs
- generates model cards
- prepares the release manifest used for publishing

### 5. Publish the released checkpoints

```bash
python3 scripts/publish_huggingface_models.py
python3 scripts/build_registry.py
```

This is the step that moves local checkpoints into the public release surface
and refreshes the generated registry so the site reflects the latest published
model state.

## Current Public Snapshot

The generated registry currently tracks:

- `26` tools
- `312` example prompts
- `6` categories
- `6` published model variants

See [registry/generated/registry_manifest.json](/scratch4/home/akrik/NTILC/registry/generated/registry_manifest.json)
for the current compiled snapshot.

## Open Tool Embeddings

Everything above explains the research system. This is the part that explains
the broader open-source direction.

**Open Tool Embeddings** is the public registry and release layer that sits on
top of the NTILC embedding work. The goal is not just to have a private tool
retriever for one agent. The goal is to build a **public foundation layer for
tool embeddings**.

That means:

- the tool ontology should be public
- the examples should be public
- the hierarchy should be explicit
- the training snapshot should be reproducible
- the checkpoint releases should be public

The long-term point is straightforward: tool use should not depend on a closed,
vendor-specific, constantly hidden registry. There should be an open-source base
representation of tools that people can inspect, improve, retrain, benchmark,
and build on.

In practice, that is what this repo is trying to become:

- **GitHub** as the collaboration surface for tool definitions and examples
- **Hugging Face** as the distribution surface for released checkpoints
- **`registry/`** as the canonical source of truth
- **`site/`** as the public index for the registry and model downloads

This is why the registry work matters. It is not side documentation. It is the
public interface for an **open-source foundation tool embedding model**:

- a shared embedding space over tools
- a hierarchy-aware representation of tool families
- a living dataset that can grow with open-source software
- public checkpoints that downstream systems can reuse for retrieval, routing,
  clustering, and evaluation

If the research is successful, the output should not be only a paper or a
single agent demo. It should be a reusable open layer for tool understanding.

## Related Public Surfaces

- GitHub repo: `https://github.com/andykr1k/NTILC`
- Hugging Face org: `https://huggingface.co/OpenToolEmbeddings`
- Site app: `site/`

To run the site locally, use Node 20+:

```bash
cd site
npm install
npm run dev
```
