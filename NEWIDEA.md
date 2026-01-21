# Neural Tool Invocation vie Learned Compression (Refined)

## High-Level Idea

At a high level, the system works as follows:

1. Represent tool *intents* as points/manifolds in a high-dimensional embedding space.  
2. Project these embeddings into a compact metric space optimized for similarity.  
3. Use metric learning, primarily **Circle Loss**, to form soft clusters of tool usage patterns.  
4. Match user queries to clusters, obtaining a **cluster ID**, instead of generating full tool calls.  
5. Map cluster IDs to concrete tools using a symbolic software layer that contains all mappings and indices.  
6. Infer which arguments are needed.  
7. Generate argument values using appropriate generative or deterministic mechanisms, only for the arguments themselves.  

This separates **what task is requested** from **how it is executed**, eliminating the need for a model decoder for tool calls.

---

## System Architecture

### 1. Tool Intent Embedding Space (1024-D)

**Purpose:** Encode semantic intent behind tool usage.

Each tool is represented as a *canonicalized intent object*, including:

* Tool name  
* Tool description  
* Argument schema  
* Example calls  
* Natural language paraphrases  

These are embedded into a high-dimensional space (e.g., 1024-D) to preserve semantic richness.

**Adding new tools:**  

- New tools can be inserted via a **key vector** in the frozen space.  
- Only the new vector is trained using Circle Loss; existing clusters remain untouched.  

---

### 2. Projection Head (1024 → 128)

**Purpose:** Map embeddings to a geometry-friendly space for similarity computation.

* Maps high-dimensional embeddings into a 128-D space.  
* Optimized with contrastive and Circle Loss objectives.  
* Not used for storage—only for similarity and loss computation.

---

### 3. Metric Learning with Circle Loss

**Purpose:** Form soft clusters of tool usage.

Circle Loss is used to:

* Pull semantically equivalent tool intents together.  
* Push unrelated intents apart.  
* Preserve angular margins between clusters.  

**Key properties for retrieval-only design:**

* Clusters are **soft**, not mutually exclusive.  
* Hard positives/negatives drive the embedding, enabling robust cluster formation.  
* New tools can be added by optimizing a **single embedding** without retraining existing ones.  

---

### 4. Query Inference (Cluster Retrieval Only)

At inference:

1. User query → embed in 1024-D space → project to 128-D.  
2. Compute similarity against all cluster centroids or prototypes.  
3. **Select top cluster(s)** → return **cluster ID(s)**.  

> No decoder or autoregressive generation is used. The cluster ID directly indexes the software layer.

**Output:**

* Cluster ID(s)  
* Similarity / confidence score  

---

### 5. Cluster-to-Tool Mapping Layer (Software Layer)

**Purpose:** Decouple model from execution.

* Maps cluster IDs to one or more tools  
* Maintains versioning, safety rules, and permissions  
* Handles tool indices, arguments, and any overlying business logic  

> All tool execution is handled here — the model only retrieves the correct cluster.

---

### 6. Argument Inference

Arguments are handled **separately from tool selection**:

#### Route A: Argument Necessity Detection

* For each candidate tool, classify arguments as required, optional, or irrelevant.  

#### Route B: Argument Value Generation

* Only relevant arguments are generated or extracted.  
* Methods:
  * Deterministic extraction (IDs, strings)  
  * Autoregressive generation for enums or short text  
  * Diffusion or continuous-value generation for coordinates, layouts, or latent parameters  

Arguments marked irrelevant are set to `null`.

---

### 7. Diffusion (Optional, Targeted Use)

Used only for continuous, multi-modal, or under-specified argument values (e.g., coordinates, layouts, time windows).  

**Not used** for discrete symbols or fixed schema arguments.

---

### 8. Null / Abstention Path

Explicit **no-tool / clarification** route:

* Triggered if all cluster similarities fall below a threshold or argument necessity is ambiguous.  
* Outcome: no tool invoked; system requests clarification.  

---

## System Flow Diagram (with Argument Generation)

```text
+--------------------+
|     User Query     |
+--------------------+
         |
         v
+--------------------+
|     Encoder        |  <- Frozen 1024-D embedding space
|     (Intent        |
|    Embedding)      |
+--------------------+
         |
         v
+--------------------+
|     Projection     |  <- 1024-D → 128-D geometry-friendly space
+--------------------+
         |
         v
+--------------------+
|    Similarity      |
|    Computation     |  <- Compare query embedding to cluster centroids
+--------------------+
         |
         v
+--------------------+
|   Cluster ID(s)    |  <- Retrieval output (no decoder)
+--------------------+
         |
         v
+--------------------+
|  Software Layer    |  <- Maps cluster ID to candidate tool(s)
|  - Cluster → Tool  |
|  - Permissions     |
|  - Safety checks   |
+--------------------+
         |
         v
+--------------------+
| Argument Handling  |
| Necessity Detection|  <- Which arguments are required, optional, irrelevant
|  Value Generation  |  <- Deterministic / autoregressive / diffusion
+--------------------+
         |
         v
+--------------------+
|    Tool Execution  |
+--------------------+


## Training Overview

### Data Requirements

* Synthetic and real tool invocation examples  
* Paraphrased intents per tool  
* Positive and negative intent pairs  
* Argument relevance labels  

---

### Objectives

* **Contrastive loss** for global embedding structure.  
* **Circle loss** for cluster geometry and new tool injection.  
* Optional auxiliary losses for argument necessity.

**New tool workflow (retrieval-only):**  

1. Encoder frozen.  
2. Optimize only the new tool embedding via Circle Loss.  
3. Return cluster ID for retrieval — no decoder needed.  

---

## All Possible Execution Routes

1. Single confident cluster → single tool → deterministic argument generation  
2. Multiple clusters → arbitration → best tool  
3. Single tool → partial arguments → request clarification  
4. Single tool → missing continuous arguments → diffusion generation  
5. Low similarity → no tool → natural language response  
6. Overlapping clusters → multi-tool plan (future extension)  

---

## Why This Approach

* Removes brittle decoder-based tool generation  
* Improves speed and efficiency — cluster lookup is near-instant  
* Preserves interpretability and debuggability  
* Allows new tools to be added without retraining the encoder  
* Matches the geometry of real-world tool usage  

---

## Future Directions

* Multi-step tool planning across clusters  
* Dynamic cluster creation  
* Online adaptation of cluster centroids  
* Tool composition graphs  
* Reinforcement learning on execution success  
