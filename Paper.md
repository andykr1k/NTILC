# NTILC: Neural Tool Invocation via Learned Compression

## Abstract
Summarize NTILC as a cluster-retrieval alternative to text-based tool invocation and preview the main gains in speed, reliability, and interpretability. Mention that tool selection and argument inference are decoupled.

## Introduction
Introduce the limitations of autoregressive tool-call generation and motivate a retrieval-first framing. State the core idea: learn an intent space where tool selection becomes nearest-cluster lookup.

## Problem Statement
Define the tool invocation problem, including latency, parsing fragility, and extensibility constraints in text-based approaches. State the goal of accurate, fast, and interpretable tool selection.

## Contributions
List the primary contributions: learned intent embedding space, cluster-based retrieval for tool selection, and a software-layer mapping with separate argument inference. Note empirical validation and analysis.

## Related Work
Briefly position NTILC relative to LLM tool-use, function calling, and metric-learning retrieval. Contrast with decoder-based tool selection and prior embedding-based routing.

## Method Overview
Provide a high-level summary of the pipeline from canonicalized tool intents to clusters and from queries to cluster IDs. Emphasize the separation between tool selection and argument generation.

## Tool Intent Representation
Describe how tool intents are canonicalized using names, descriptions, schemas, and examples. Explain why this representation supports stable clustering and extensibility.

## Intent Embedding and Metric Learning
Explain the 1024-D embedding and 128-D projection with Circle Loss to form soft clusters. Highlight how metric learning improves separation and supports new tool injection.

## Query Encoder and Cluster Retrieval
Describe how queries are embedded into the same space and matched to cluster centroids. Note O(1) retrieval and confidence scoring without decoding.

## Software Layer and Argument Inference
Explain how cluster IDs map to tools with routing/safety logic. Outline the separate argument-necessity classification and value generation process.

## Abstention and Clarification
Describe the threshold-based abstention path when similarity is low or argument necessity is ambiguous. Explain how this improves reliability over forced tool calls.

## Training Setup
Summarize the two-phase training procedure (intent embedding, then cluster retrieval). Mention datasets (synthetic and real) and key hyperparameters.

## Experimental Setup
Describe baselines, evaluation tasks, and metrics (cluster accuracy, similarity, latency). Specify hardware and implementation details needed for reproducibility.

## Results
Report headline quantitative results comparing NTILC to text-based baselines. Include accuracy, latency, and error-type breakdowns.

## Ablations
Outline experiments isolating effects of projection size, loss functions, and clustering strategy. Explain which components contribute most to performance.

## Analysis
Provide qualitative examples of tool selection behavior and interpretability via cluster IDs. Discuss failure cases and when abstention triggers.

## Limitations
Note constraints such as dependence on high-quality tool intent definitions and challenges with overlapping tool semantics. Mention scalability considerations as tool count grows.

## Ethics and Societal Impact
Discuss safety implications of tool routing, potential misuse, and mitigations via software-layer governance. Note privacy considerations for tool inputs.

## Conclusion
Reiterate that NTILC reframes tool invocation as retrieval in a learned intent space. Summarize improvements and outline next steps.

## Acknowledgments
List contributors, funding sources, and any tool/library acknowledgments.

## References
Cite prior work on tool-use LLMs, function calling, and metric learning. Ensure consistent citation style.

## Appendix
Include supplementary details such as model configs, additional experiments, and extended examples.
