# NTILC: Neural Tool Invocation via Learned Compression

## Overview

NTILC introduces a novel approach to language model tool use by replacing text-based tool invocation with learned continuous embeddings. The core goal is to **predict the next tool call given previous tool calls and context**. Instead of generating text like "search(query='cats', max_results=10)" which must be parsed, the model directly predicts a 512-dimensional embedding that encodes the complete tool invocation. This embedding is then decoded back into an executable tool call.

## The Fundamental Problem

Current state-of-the-art agentic models use text-based tool invocation:
- LLM generates: "<search>quantum computing</search>"
- Parser extracts structured data
- System executes the tool call

This approach has several limitations:
1. Sequential token generation is slow (15-20 tokens per tool call)
2. Parsing errors from malformed text
3. No semantic similarity between related tool calls
4. Discrete token space makes optimization difficult

## Core Innovation

NTILC learns a continuous embedding space for tool invocations to predict the **next tool call** given previous tool calls and context. The key innovations are:
- Each tool call maps to a point in R^d (e.g., 512 dimensions)
- Similar tool calls cluster together in embedding space
- A single embedding prediction replaces sequential token generation
- Embeddings are differentiable and semantically meaningful
- The model learns to predict embeddings conditioned on conversation context and tool call history

### Information-Theoretic Foundation

A tool invocation contains:
- Tool identity: log₂(|T|) bits (e.g., ~3 bits for 6 tools)
- Parameters: Σᵢ I(pᵢ|T) bits (e.g., ~50-200 bits for text query)
- Total: ~60-210 bits of information

A 512-dimensional float32 embedding has capacity of 512 × 32 = 16,384 bits, providing ~100x overhead. This massive redundancy ensures lossless compression is theoretically achievable.

### Manifold Hypothesis

High-dimensional data (tool invocations) actually lie on a lower-dimensional manifold:
- All possible strings: infinite dimensional
- Valid tool invocations:
    - Encoder: ToolCall → R^d
    - Decoder: R^d → ToolCall 
This form a 2D manifold where:
- One dimension captures query semantics
- Another dimension captures result count

## Architecture

### Phase 1: Tool Invocation Autoencoder

We train an encoder-decoder pair:

Encoder: φ_θ: ToolCall → R^d
- Input: Tokenized tool call string
- Transformer encoder processes tokens
- Pool hidden states (mean/attention)
- Project to d-dimensional embedding

Decoder: ψ_φ: R^d → ToolCall  
- Input: d-dimensional embedding
- Project to decoder hidden dimension
- Autoregressive transformer generates tokens
- Output: Reconstructed tool call string

Training objective:
L_reconstruction = -Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁, z)

This is sequence-to-sequence learning with an information bottleneck that forces the model to learn efficient representations.

### Phase 2: LLM Integration

Once the autoencoder is trained, we integrate with an LLM to predict the **next tool call** given previous tool calls and context:

Standard LLM: P(token_t | context)
NTILC LLM: P(token_t | context) ∪ P(embedding | context, previous_tool_calls)

The model learns to predict tool embeddings at appropriate points:

At each position t with hidden state h_t:
- Standard path: p_token = Softmax(W_lm · h_t)
- Tool path: z_predicted = W_tool · h_t

Training:
- Input context includes:
  - Conversation history (user queries, responses)
  - Previous tool calls: [tool_call₁, tool_call₂, ..., tool_callₙ₋₁]
  - Current state/context
- LLM processes context → hidden state h_t
- Ground truth next tool call → autoencoder embedding z_target
- Loss: MSE(z_predicted, z_target)
- Backprop updates LLM to predict correct embeddings for the next tool call

### Complete Data Flow

**Training autoencoder:**
1. Input: "search(query='machine learning', max_results=10)"
2. Tokenize: [42, 123, 456, ...]
3. Encode: Transformer → h_avg → z = [0.23, -0.45, 0.67, ..., 0.12]
4. Decode: z → h₀ → autoregressively generate tokens
5. Loss: CrossEntropy(generated, original)

**Training LLM integration:**
1. Input context: conversation history + previous tool calls [tool_call₁, ..., tool_callₙ₋₁]
2. Ground truth next tool call: "search(query='transformer papers', max_results=10)"
3. Encode ground truth: z_target = φ_θ(ground_truth) = [0.25, -0.43, ...]
4. LLM processes context → h_t
5. LLM predicts: z_predicted = W_tool · h_t
6. Loss: MSE(z_predicted, z_target)

**Inference (predicting next tool call):**
1. User: "Find me recent AI news"
2. Context includes: previous tool calls (if any) + conversation history
3. LLM processes context → h_t
4. Predict next tool call embedding: z = W_tool · h_t
5. Decode: tool_call = Decoder(z) → "search(query='recent AI news', max_results=10)"
6. Execute tool call
7. Add executed tool call to history for next prediction
8. Return results to LLM

## Why This Works: Theoretical Foundations

### 1. Universal Approximation
Neural networks can approximate any continuous function, so given sufficient dimensions, we can learn mappings:
- φ: ToolCall → R^d (encoder)
- ψ: R^d → ToolCall (decoder)
Such that ψ(φ(x)) ≈ x

### 2. Information Bottleneck
The bottleneck forces the model to:
- Discard irrelevant information
- Preserve task-critical information
- Learn invariances (e.g., "cat" ≈ "cats")

This is a classic principle from information theory: compress while preserving task-relevant structure.

### 3. Continuous Optimization
Unlike discrete token prediction, continuous embeddings enable smooth gradients throughout the entire pipeline, making optimization easier and more stable.

### 4. Semantic Geometry
Continuous spaces enable:
- Similarity: ||embed(search "cats") - embed(search "dogs")|| is small
- Interpolation: weighted combinations of embeddings
- Composition: embedding arithmetic may yield meaningful results

## Advantages Over Text-Based Tool Calling

### Speed
- Text: O(sequence_length) token generations
- NTILC: O(1) single embedding prediction
- Expected speedup: 10-20x for tool invocation

### Robustness
- Text: Parsing errors from malformed output
- NTILC: Decoder learns to handle noisy embeddings
- Errors don't cascade (single prediction vs sequential)

### Semantics
- Text: "search cats" and "search dogs" are completely different token sequences
- NTILC: Similar embeddings with small semantic offset
- Better generalization to unseen parameter values

### Training Dynamics
- Text: Sparse gradients through discrete tokens
- NTILC: Dense gradients through continuous embeddings
- Faster convergence, easier optimization

### Compositionality
Continuous space may enable:
- embed(search "cats") + α · (embed(search "dogs") - embed(search "cats"))
- Could discover searches for related concepts
- Tool composition through embedding operations

## Implementation Strategy

### Data Generation

Create synthetic training data covering:

Tool schemas:
- search(query: str, max_results: int, date_filter: Optional[DateRange])
- calculate(expression: str)
- database_query(sql: str, timeout: int)
- send_email(to: email, subject: str, body: str, cc: Optional[List[email]])
- web_fetch(url: str, method: enum["GET", "POST"])
- file_read(path: str, encoding: str)

Generate 100K+ diverse examples:
- Use Faker library for realistic parameter values
- Vary parameter combinations
- Include edge cases
- Add real-world examples from API docs

### Data Curation - Another Possible Route

Start with Question:
- Grab questions from a dataset like HLE
- Hire undergrad or use chatgpt to create tool call annotations

Start with Tool Call:
- Grab MCP server configs
- Parse config to our structure

**Training Configuration:**

Phase 1 - Autoencoder (50 epochs):
- Batch size: 64
- Learning rate: 1e-4
- Optimizer: AdamW
- Loss: CrossEntropy for reconstruction
- Metrics: Exact match accuracy, tool accuracy, parameter accuracy
- Early stopping on validation loss

Phase 2 - LLM Integration (20 epochs):
- Freeze autoencoder weights
- Initialize tool prediction head
- Learning rate: 2e-4
- Loss: λ₁·LM_loss + λ₂·MSE(z_pred, z_target)
- Balance hyperparameters λ₁, λ₂

### Evaluation Metrics

Autoencoder quality:
- Exact reconstruction accuracy
- Per-tool accuracy
- Per-parameter-type accuracy
- Embedding space properties (norm, variance, clustering)
    - Robustness to context variations

End-to-end system:
- Next tool call prediction accuracy (given previous tool calls and context)
- Tool selection accuracy
- Parameter correctness (exact match)
    - Take methods from previous work (How are LLMs "accuracy" measured?)
- Parameter correctness (semantic similarity for strings)
    - Take methods from previous work (How are LLMs "accuracy" measured?)
- Latency vs text baseline
- Total energy usage (watts used)

## Research Questions

### Q1: Optimal Embedding Dimension
How does d ∈ {128, 256, 512, 1024} affect:
- Reconstruction quality
- LLM learning difficulty
- Inference speed

Hypothesis: Sweet spot around 512 - enough capacity without overwhelming the LLM predictor.

### Q2: Pooling Strategy
Compare encoder pooling methods:
- Mean pooling: (1/L)Σᵢ hᵢ
- CLS token: h₀
- Max pooling: maxᵢ hᵢ
- Attention pooling: Σᵢ αᵢhᵢ where α = softmax(...)

Hypothesis: Attention pooling captures variable-length tool calls best.

### Q3: Embedding Space Structure
Analyze learned embedding space:
- Do similar tools cluster? (t-SNE visualization)
- Does embedding arithmetic work? (search "cats" + trend_vector = search "trending cats")
- Can we discover tool relationships? (hierarchical clustering)

Hypothesis: Semantic structure emerges naturally from reconstruction objective.

### Q4: Generalization
Test zero-shot capabilities:
- Unseen parameter values (new query strings)
- Unseen parameter combinations
- New tools (add tool, fine-tune decoder only)

Hypothesis: Continuous space enables interpolation to unseen regions better than discrete tokens.

### Q5: Information Capacity
Theoretical analysis:
- Shannon entropy of tool call distribution
- Mutual information I(Z; X)
- Rate-distortion trade-offs

Hypothesis: Can derive optimal embedding dimension from information-theoretic bounds.

### Q6: Multi-Tool Composition
Can the model learn to:
- Predict sequences of tool calls?
- Compose tools? (search → filter → summarize)
- Chain dependent calls? (search → fetch URL → extract)

Hypothesis: Extend to sequence of embeddings: [z₁, z₂, z₃, ...] for complex workflows.

## Mathematical Formalization

Definitions:
- X: Space of tool invocations (strings)
- Z ⊂ R^d: Embedding space
- C: Context space (conversation history + previous tool calls)
- H: History of previous tool calls [x₁, x₂, ..., xₙ₋₁]
- φ_θ: X → Z (encoder with parameters θ)
- ψ_φ: Z → X (decoder with parameters φ)
- f_ω: (C, H) → Z (LLM tool predictor with parameters ω, predicts next tool call embedding)

Phase 1 Objective (Autoencoder):
min_{θ,φ} E_{x~p(X)} [L_recon(x, ψ_φ(φ_θ(x)))]

Where L_recon = -Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁, z)

Phase 2 Objective (LLM Integration):
min_ω E_{(c,h,x)~p(C,H,X)} [||f_ω(c, h) - φ_θ(x)||²₂]

Where:
- c is the conversation context
- h = [x₁, ..., xₙ₋₁] is the history of previous tool calls
- x is the ground truth next tool call
- φ_θ is frozen from Phase 1

Inference (predicting next tool call):
Given context c and previous tool calls h:
1. z = f_ω(c, h)  (predict next tool call embedding)
2. x̂ = ψ_φ(z)  (decode to tool call string)
3. Execute(x̂)
4. Add x̂ to history h for next prediction

## Comparison to Related Work

### vs. ToolFormer
- ToolFormer: LLM generates text API calls, learns when to call tools
- NTILC: LLM generates embeddings, continuous action space
- Advantage: Faster, no parsing, semantic structure

### vs. ReAct
- ReAct: Interleaves reasoning and action in text
- NTILC: Actions are embeddings, reasoning is text
- Advantage: Clear separation, faster action prediction

### vs. Function Calling APIs (OpenAI, Anthropic)
- Function Calling: Structured JSON generation with constrained decoding
- NTILC: Embedding generation with learned decoding
- Advantage: Single-step prediction, continuous optimization

### vs. VQ-VAE - Ablation
- VQ-VAE: Discrete codebook, quantized latents
- NTILC: Continuous latents
- Difference: We want smooth gradients for LLM training

### vs. CLIP - Not neccesarily needed
- CLIP: Contrastive learning for vision-language embeddings
- NTILC: Reconstruction learning for tool invocation embeddings
- Similarity: Both learn embedding spaces
- Difference: We need decode capability, not just similarity

## Potential Extensions

### 1. Hierarchical Embeddings
Separate embeddings for:
- Tool selection (low-dim, e.g., 64)
- Parameter specification (high-dim, e.g., 448)
- Concatenate: [tool_emb || param_emb]

Advantage: Easier to learn, modularity

### 2. Multi-Tool Predictions
Predict sequence of embeddings:
- [z₁, z₂, z₃] for chained tool calls
- Attention mechanism to decide sequence length
- Enables complex workflows

### 3. Tool Composition Algebra
Learn operations in embedding space:
- z_search + z_filter = z_filtered_search
- z_tool1 ⊕ z_tool2 = z_composed
- Discover new tool combinations

### 4. Uncertainty Quantification
Extend to probabilistic embeddings:
- Predict μ, σ instead of deterministic z
- Sample during inference: z ~ N(μ, σ²)
- Provides confidence estimates

### 5. Few-Shot Tool Learning
For new tools:
- Provide k examples
- Fine-tune decoder only (few parameters)
- LLM learns to predict embeddings for new tool
- Fast adaptation

### 6. Cross-Modal Tools
Extend beyond text parameters:
- Image tools: embed(generate_image, description="cat")
- Audio tools: embed(text_to_speech, text="hello", voice="calm")
- Unified embedding space for all modalities

## Limitations and Challenges

### 1. Information Loss
Bottleneck may lose information:
- Very long parameter strings
- Complex nested structures
- Edge cases

Mitigation:
- Over-parameterize embedding dimension
- Hybrid approach: embeddings + text for complex cases
- Learn to route between embedding and text

### 2. Decoder Quality
Reconstruction errors propagate:
- Wrong tool selected
- Parameter corruption
- Type violations

Mitigation:
- Extensive autoencoder training
- Post-processing validation
- Confidence thresholds

### 3. Training Data Requirements
Need diverse, high-quality examples:
- 100K+ tool invocations
- Cover parameter space
- Real-world distributions

Mitigation:
- Synthetic generation
- Data augmentation
- Active learning

### 4. LLM Learning Difficulty
Predicting embeddings may be harder than tokens:
- No discrete targets
- Continuous optimization
- High-dimensional space

Mitigation:
- Curriculum learning (easy tools first)
- Auxiliary losses (tool classification + embedding)
- Pretrain on easier tasks

### 5. Interpretability
Embeddings are black boxes:
- Hard to debug
- Can't manually inspect
- Difficult to validate

Mitigation:
- Visualization (t-SNE, PCA)
- Decode and display
- Probe classifiers for properties

### 6. Generalization Bounds
No guarantees for:
- Zero-shot new tools
- Out-of-distribution parameters
- Compositionality

Mitigation:
- Empirical evaluation
- Nearest-neighbor fallbacks
- Hybrid text backup

## Expected Contributions

### To ML/AI Research:
1. Novel continuous action space for LLMs
2. Learned compression of structured data
3. Semantic embedding space for programs/API calls
4. Bridge between language models and program synthesis

### To Practical Systems:
1. Faster tool invocation (10-20x speedup)
2. More robust to noise (no parsing errors)
3. Better generalization (semantic interpolation)
4. Enables tool composition and discovery

### To Theory:
1. Information-theoretic analysis of tool invocations
2. Rate-distortion trade-offs for action spaces
3. Compositionality in learned program embeddings
4. Manifold learning for structured data

(New dataset that was created with lots of labor and has high quality data in a smart way)

## Implementation Roadmap

### Month 1: Foundation
- Why is this problem is new and needed to be worked on?
    - Collect many related works and read through all the problem settings/intros
- Prepare datasets and benchmarks
- **Ablation Studies** (see `ablation/README.md`):
    - Baseline 1: Naive prompting (tools in prompt)
    - Baseline 2: Cross-attention LLM (architectural tool awareness)
    - Compare baselines before full training
- Implement autoencoder architecture
- Build synthetic data generator
- Train and evaluate autoencoder
- Analyze embedding space

### Month 2: Integration
- Integrate with base LLM (GPT-2, LLaMA)
- Implement training pipeline
- Initial experiments

### Month 3: Evaluation
- Compare to text baseline
- Ablation studies
- Generalization tests
- Performance profiling

### Month 4: Extensions
- Multi-tool predictions
- Composition experiments
- Few-shot learning
- Real-world deployment tests

## Success Metrics

Minimum viable success:
- 90%+ autoencoder reconstruction accuracy
- 80%+ end-to-end tool selection accuracy
- 5x+ speedup over text baseline
- Comparable parameter accuracy to text

Strong success:
- 95%+ reconstruction accuracy
- 90%+ end-to-end accuracy
- 10x+ speedup
- Better generalization than text baseline
- Demonstrable compositionality

Breakthrough success:
- 98%+ reconstruction accuracy
- 95%+ end-to-end accuracy
- 20x+ speedup
- Zero-shot new tool learning
- Discovers novel tool combinations

## Conclusion

NTILC represents a paradigm shift in how language models interact with tools. By moving from discrete text-based invocation to continuous learned embeddings, we enable:

- Faster inference through single-step prediction
- Richer semantics through continuous representations  
- Better optimization through smooth gradients
- Potential compositionality through embedding arithmetic

The approach is theoretically grounded in information theory, representation learning, and neural network theory. While challenges remain around information loss, decoder quality, and generalization, the potential benefits justify deep exploration.

This research bridges language modeling, program synthesis, and representation learning, opening new directions for agentic AI systems.