Problem
Text-based tool invocation in language models is slow, brittle, and hard to extend. Autoregressive decoding adds latency and error-prone parsing, and adding new tools often requires retraining or prompt re-engineering. This limits reliability, interpretability, and scalability in real-world tool use.

Key Observation
Tool intent is better treated as a geometric retrieval problem than a text generation problem. If we embed canonical tool intents into a semantic space, then tool selection reduces to a nearest-cluster lookup. This yields fast, interpretable decisions and decouples tool selection from argument generation.

Our Approach
We replace decoder-based tool invocation with cluster-based retrieval in a learned intent space.
Intent embedding and clustering: Canonicalized tool intents (name, description, schema, examples, paraphrases) are embedded into a 1024-D space and projected to 128-D for metric learning. Circle Loss forms soft, separable clusters of tool usage patterns.
Query-to-cluster retrieval: A query encoder maps natural language requests into the same 128-D space and retrieves the nearest cluster ID in O(1) time, producing a confidence score without autoregressive decoding.
Software-layer mapping and arguments: Cluster IDs are mapped to tools by a software layer that handles safety and routing. Argument inference is performed separately, classifying which arguments are needed and generating only those values, keeping tool selection and argument generation cleanly decoupled.
Abstention path: If similarity is below a threshold or argument necessity is ambiguous, the system abstains and requests clarification rather than making a brittle guess.