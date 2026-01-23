"""
NTILC Models Package

New Architecture (NEWIDEA):
- Intent Embedder: Tool intents → 1024-D embeddings
- Projection Head: 1024-D → 128-D for similarity
- Cluster Retrieval: Query → Cluster ID (no decoder!)
- Software Layer: Cluster ID → Tool mapping
- Argument Inference: Separate argument handling

Legacy (Autoencoder):
- Autoencoder: Tool call → Embedding → Reconstructed tool call
- Encoder/Decoder: Components of autoencoder
- LLM Integration: NL → Embedding → Decoder → Tool call
"""

# New Architecture Components
from .intent_embedder import ToolIntentEmbedder
from .projection_head import ProjectionHead
from .cluster_retrieval import ClusterRetrieval
from .software_layer import ClusterToToolMapper
from .argument_inference import ArgumentNecessityClassifier, ArgumentValueGenerator

# Legacy Components (kept for reference/migration)
from .autoencoder import ToolInvocationAutoencoder
from .encoder import ToolInvocationEncoder
from .decoder import ToolInvocationDecoder
from .llm_integration import ToolPredictionLLM, ToolPredictionHead

__all__ = [
    # New Architecture
    "ToolIntentEmbedder",
    "ProjectionHead",
    "ClusterRetrieval",
    "ClusterToToolMapper",
    "ArgumentNecessityClassifier",
    "ArgumentValueGenerator",
    # Legacy
    "ToolInvocationAutoencoder",
    "ToolInvocationEncoder",
    "ToolInvocationDecoder",
    "ToolPredictionLLM",
    "ToolPredictionHead",
]
