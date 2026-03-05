"""Cluster retrieval runtime used by the orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from models.cluster_retrieval import ClusterRetrieval
from models.query_encoder import QueryEncoder
from models.software_layer import ClusterToolMapper
from orchestrator.results import RetrievalCandidate


def _resolve_torch_dtype(dtype_name: Optional[str]) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name and dtype_name in mapping:
        return mapping[dtype_name]
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class ClusterBasedToolSystem:
    """
    Retrieval-only inference: query -> top-k cluster/tool candidates.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        cluster_centroids: torch.Tensor,
        mapper: ClusterToolMapper,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.query_encoder = query_encoder
        self.cluster_centroids = F.normalize(cluster_centroids.float(), p=2, dim=1)
        self.mapper = mapper
        self.max_length = int(max_length)
        self.device = device or next(query_encoder.parameters()).device

        self.cluster_retrieval = ClusterRetrieval(
            embedding_dim=int(self.cluster_centroids.shape[1]),
            num_clusters=int(self.cluster_centroids.shape[0]),
            similarity_type="cosine",
        )

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: Optional[str] = None,
        query_encoder_path: str = "checkpoints/cluster_retrieval/best_model.pt",
        device: Optional[str] = None,
    ) -> "ClusterBasedToolSystem":
        del intent_embedder_path  # API compatibility.
        checkpoint = torch.load(query_encoder_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        cluster_centroids = checkpoint["cluster_centroids"]
        if not isinstance(cluster_centroids, torch.Tensor):
            cluster_centroids = torch.tensor(cluster_centroids, dtype=torch.float32)

        num_clusters = int(cluster_centroids.shape[0])
        tool_names = checkpoint.get("tool_names", [])
        if not isinstance(tool_names, list):
            tool_names = []
        if len(tool_names) < num_clusters:
            tool_names = list(tool_names) + [f"cluster_{i}" for i in range(len(tool_names), num_clusters)]
        else:
            tool_names = list(tool_names[:num_clusters])

        mapper = ClusterToolMapper.from_tool_names(tool_names)

        target_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_dtype_name = str(config.get("torch_dtype", "float32"))
        model_dtype = _resolve_torch_dtype(model_dtype_name)
        if target_device.type == "cpu":
            model_dtype_name = "float32"
            model_dtype = torch.float32

        query_encoder = QueryEncoder(
            base_model=str(config.get("encoder_model", "Qwen/Qwen3.5-9B")),
            output_dim=int(config.get("projection_dim", cluster_centroids.shape[1])),
            dropout=float(config.get("dropout", 0.15)),
            torch_dtype=model_dtype_name
            if model_dtype_name in {"float16", "bfloat16", "float32"}
            else "float32",
        )
        query_encoder.load_state_dict(checkpoint["query_encoder_state_dict"])
        query_encoder = query_encoder.to(target_device).to(model_dtype)
        query_encoder.eval()

        return cls(
            query_encoder=query_encoder,
            cluster_centroids=cluster_centroids.to(target_device).to(model_dtype),
            mapper=mapper,
            max_length=int(config.get("max_length", 256)),
            device=target_device,
        )

    def encode_query(self, query: str) -> torch.Tensor:
        encoded = self.query_encoder.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            query_embedding = self.query_encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        return query_embedding

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = -1.0,
    ) -> List[RetrievalCandidate]:
        query_embedding = self.encode_query(query)
        with torch.no_grad():
            retrieval = self.cluster_retrieval(
                query_embeddings=query_embedding,
                cluster_embeddings=self.cluster_centroids.to(query_embedding.dtype),
                top_k=max(1, int(top_k)),
                threshold=float(threshold),
            )

        cluster_ids = retrieval["cluster_ids"][0].detach().cpu().tolist()
        scores = retrieval["similarities"][0].detach().cpu().tolist()
        candidates: List[RetrievalCandidate] = []
        for cluster_id, score in zip(cluster_ids, scores):
            cid = int(cluster_id)
            if cid < 0:
                continue
            try:
                tool_name = self.mapper.cluster_to_tool(cid)
            except KeyError:
                tool_name = f"cluster_{cid}"
            candidates.append(RetrievalCandidate(cluster_id=cid, score=float(score), tool_name=tool_name))
        return candidates

    def predict(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = -1.0,
    ) -> Dict[str, Any]:
        candidates = self.retrieve_candidates(query=query, top_k=top_k, threshold=threshold)
        return {
            "query": query,
            "top_k": top_k,
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
