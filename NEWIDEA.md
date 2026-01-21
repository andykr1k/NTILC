## High-Level Idea

At a high level, the system works as follows:

1. Represent tool *intents* as points/manifolds in a high-dimensional embedding space
2. Project these embeddings into a compact metric space optimized for similarity
3. Use metric learning (circle loss) to form soft clusters of tool usage patterns
4. Match user queries to clusters instead of directly predicting tools
5. Map clusters to concrete tools using a symbolic software layer
6. Infer which arguments are needed
7. Generate argument values using appropriate generative mechanisms

This separates **what task is being requested** from **how it is executed**.

---

## System Architecture

### 1. Tool Intent Embedding Space (1024-D)

**Purpose:** Encode semantic intent behind tool usage.

Each tool is represented not as raw JSON, but as a *canonicalized intent object*:

* Tool name
* Tool description
* Argument schema
* Example calls
* Natural language paraphrases of intent

These are embedded into a high-dimensional space (e.g. 1024-D) to preserve semantic richness.

**Why high-dimensional?**

* Captures subtle distinctions between similar tools
* Allows tools to occupy manifolds rather than single points

---

### 2. Projection Head (1024 → 128)

**Purpose:** Create a geometry-friendly space for similarity learning.

A learned projection maps embeddings into a smaller space (e.g. 128-D) optimized using contrastive objectives.

* Acts as a semantic bottleneck
* Encourages functional equivalence classes
* Improves clustering stability

This space is *not* used for storage—only for similarity and loss computation.

---

### 3. Metric Learning with Circle Loss

**Purpose:** Form soft clusters of tool usage.

Circle loss is used to:

* Pull semantically equivalent tool intents together
* Push unrelated intents apart
* Preserve angular margins between clusters

Important properties:

* Clusters are **soft**, not mutually exclusive
* A query can be close to multiple clusters
* Similarity scores are continuous

This naturally models overlapping tool functionality.

---

### 4. Query Inference

At inference time:

1. User query is embedded into the same 1024-D space
2. Projected into 128-D
3. Similarity is computed against all cluster centroids or prototypes

Output:

* Top-k cluster matches
* Confidence / similarity scores

No hard argmax is required at this stage.

---

### 5. Cluster-to-Tool Mapping Layer (Software Layer)

**Purpose:** Decouple learned geometry from execution logic.

This symbolic layer:

* Maps clusters to one or more tools
* Handles tool versioning
* Applies business rules, permissions, safety checks
* Allows tools to change without retraining embeddings

This is where **system control lives**, not inside the model.

---

### 6. Argument Inference

Argument handling is explicitly decomposed.

#### Route A: Argument Necessity Detection

For each candidate tool:

* Determine whether each argument is:

  * Required
  * Optional
  * Irrelevant

This is treated as a **classification problem**, not generation.

Possible implementations:

* Per-tool argument classifiers
* Schema-aware attention masks
* Learned argument gates

---

#### Route B: Argument Value Generation

Only arguments deemed relevant proceed to value generation.

Possible generation routes:

1. **Deterministic extraction**

   * IDs
   * Strings explicitly present in the query

2. **Autoregressive generation**

   * Enums
   * Booleans
   * Short text fields

3. **Diffusion-based generation (optional)**

   * Continuous values
   * Underspecified or multi-modal arguments
   * Spatial, temporal, or configuration-like parameters

Arguments marked irrelevant are explicitly set to `null`.

---

### 7. Diffusion (Optional, Targeted Use)

Diffusion is used only where appropriate.

Good use cases:

* Coordinates
* Time windows
* Layouts
* Latent configuration vectors

Not recommended for:

* Discrete symbols
* Identifiers
* Fixed schemas

A hybrid argument generator is encouraged.

---

### 8. Null / Abstention Path

The system includes an explicit **no-tool / clarification** route.

If:

* All cluster similarities fall below a threshold
* Or argument necessity cannot be resolved

Then:

* No tool is invoked
* The system requests clarification or responds in natural language

This prevents over-invocation.

---

## Training Overview

### Data Requirements

* Synthetic and real tool invocation examples
* Paraphrased intents per tool
* Positive and negative intent pairs
* Argument relevance labels

### Objectives

* Contrastive loss on projected embeddings
* Circle loss for cluster geometry
* Optional auxiliary losses for argument necessity

---

## All Possible Execution Routes

1. **Single confident cluster → single tool → deterministic arguments**
2. **Multiple clusters → arbitration → best tool**
3. **Single tool → partial arguments → request clarification**
4. **Single tool → missing continuous arguments → diffusion generation**
5. **Low similarity → no tool → natural language response**
6. **Overlapping clusters → multi-tool plan (future extension)**

---

## Why This Approach

* Removes brittle prompt-based control
* Improves interpretability and debuggability
* Scales to large tool libraries
* Separates learning from system logic
* Matches the geometry of real-world tasks

---

## Future Directions

* Multi-step tool planning across clusters
* Dynamic cluster creation
* Online adaptation of cluster centroids
* Tool composition graphs
* Reinforcement learning on execution success
