"""
NTILC Cluster-Based Inference Pipeline.

End-to-end inference from natural language queries to tool selection via
cluster retrieval and software-layer mapping. Uses python function calling
as the tool call format.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Any, Union

import torch

from models.intent_embedder import ToolIntentEmbedder
from models.projection_head import ProjectionHead
from models.cluster_retrieval import ClusterRetrieval
from models.query_encoder import QueryEncoder
from models.software_layer import ClusterToToolMapper
from models.argument_inference import ArgumentValueGenerator
from models.tool_call_utils import get_required_args, get_optional_args
from models.tool_schemas import TOOL_SCHEMAS


def _format_python_value(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_python_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"'{k}': {_format_python_value(v)}" for k, v in value.items()) + "}"
    if value is None:
        return "None"
    return str(value)


def format_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    args_str = ", ".join(f"{k}={_format_python_value(v)}" for k, v in arguments.items())
    return f"{tool_name}({args_str})"


@dataclass
class ClusterToolCallResult:
    tool_name: Optional[str]
    arguments: Dict[str, Any]
    tool_call: Optional[str]
    cluster_id: int
    confidence: float
    needs_clarification: bool
    missing_required_args: List[str]
    timings_ms: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "tool_call": self.tool_call,
            "cluster_id": self.cluster_id,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "missing_required_args": self.missing_required_args,
            "timings_ms": self.timings_ms,
        }


class ClusterBasedToolSystem:
    """
    Cluster-based tool selection with software-layer mapping and
    separate argument inference.
    """

    def __init__(
        self,
        intent_embedder: ToolIntentEmbedder,
        projection_head: ProjectionHead,
        query_encoder: QueryEncoder,
        cluster_centroids: torch.Tensor,
        mapper: ClusterToToolMapper,
        device: Optional[str] = None
    ):
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
            device = "cuda"
        elif not str(device).startswith("cuda"):
            raise RuntimeError(f"CUDA device required, got: {device}")
        self.device = device
        self.intent_embedder = intent_embedder
        self.projection_head = projection_head
        self.query_encoder = query_encoder
        self.cluster_centroids = cluster_centroids
        self.mapper = mapper
        self.cluster_retrieval = ClusterRetrieval(
            embedding_dim=cluster_centroids.shape[1],
            num_clusters=cluster_centroids.shape[0],
            similarity_type="cosine"
        )
        self.arg_generator = ArgumentValueGenerator()
        self.mapper.build_cache()

        self.intent_embedder.eval()
        self.projection_head.eval()
        self.query_encoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: str,
        query_encoder_path: str,
        mapper_path: Optional[str] = None,
        device: Optional[str] = None
    ) -> "ClusterBasedToolSystem":
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available.")
            device = "cuda"
        elif not str(device).startswith("cuda"):
            raise RuntimeError(f"CUDA device required, got: {device}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }

        # Phase 1: intent embedder + projection head
        intent_ckpt = torch.load(intent_embedder_path, map_location=device, weights_only=False)
        intent_config = intent_ckpt.get("config", {})
        model_torch_dtype = intent_config.get("torch_dtype", "bfloat16")
        use_dtype = dtype_map.get(model_torch_dtype, torch.float32)
        intent_embedder = ToolIntentEmbedder(
            model_name=intent_config.get("encoder_model", "google/flan-t5-base"),
            embedding_dim=intent_config.get("intent_embedding_dim", 1024),
            pooling_strategy=intent_config.get("pooling_strategy", "attention"),
            dropout=intent_config.get("dropout", 0.15),
            torch_dtype=model_torch_dtype,
            max_length=intent_config.get("max_length", 512)
        )
        intent_embedder.load_state_dict(intent_ckpt["intent_embedder_state_dict"])
        intent_embedder = intent_embedder.to(device)

        projection_head = ProjectionHead(
            input_dim=intent_config.get("intent_embedding_dim", 1024),
            output_dim=intent_config.get("projection_dim", 128),
            dropout=intent_config.get("dropout", 0.15)
        )
        projection_head.load_state_dict(intent_ckpt["projection_head_state_dict"])
        projection_head = projection_head.to(device).to(use_dtype)

        # Phase 2: query encoder + cluster centroids
        query_ckpt = torch.load(query_encoder_path, map_location=device, weights_only=False)
        query_config = query_ckpt.get("config", {})
        query_encoder = QueryEncoder(
            base_model=query_config.get("encoder_model", "google/flan-t5-base"),
            output_dim=query_config.get("projection_dim", 128),
            dropout=query_config.get("dropout", 0.15),
            torch_dtype=query_config.get("torch_dtype", model_torch_dtype)
        ).to(device).to(use_dtype)
        query_encoder.load_state_dict(query_ckpt["query_encoder_state_dict"])

        cluster_centroids = query_ckpt["cluster_centroids"].to(device, dtype=use_dtype)

        # Software layer
        mapper = ClusterToToolMapper()
        if mapper_path:
            mapper.load(mapper_path)
        else:
            mapper.initialize_from_tools(list(TOOL_SCHEMAS.keys()))
        mapper.build_cache()

        return cls(
            intent_embedder=intent_embedder,
            projection_head=projection_head,
            query_encoder=query_encoder,
            cluster_centroids=cluster_centroids,
            mapper=mapper,
            device=device
        )

    def _infer_arguments(self, query: str, tool_name: str) -> Dict[str, Any]:
        required_args = get_required_args(tool_name)
        optional_args = get_optional_args(tool_name)

        # Start with required args only; optional args added if extracted.
        arguments: Dict[str, Any] = {}
        schema = TOOL_SCHEMAS.get(tool_name, {})

        for arg_name in required_args + optional_args:
            param_info = schema.get("parameters", {}).get(arg_name, {})
            arg_type = param_info.get("type", "str")

            value = self.arg_generator.extract_from_query(
                query=query,
                arg_name=arg_name,
                arg_type=arg_type,
                tool_name=tool_name
            )

            if value is None:
                if arg_name in required_args:
                    default = param_info.get("default", None)
                    if default is not None:
                        value = default
                else:
                    # Optional arg: only include if extracted
                    value = None

            if value is not None:
                arguments[arg_name] = value

        return arguments

    def _sync(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def predict(
        self,
        query: str,
        top_k: int = 1,
        similarity_threshold: float = 0.5,
        force_tool_call: bool = True,
        return_timings: bool = False
    ) -> Union[ClusterToolCallResult, List[ClusterToolCallResult]]:
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        with torch.no_grad():
            tokenize_start = time.perf_counter()
            encoded = self.query_encoder.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            timings["tokenize_ms"] = (time.perf_counter() - tokenize_start) * 1000.0

            self._sync()
            encode_start = time.perf_counter()
            query_embedding = self.query_encoder(encoded["input_ids"], encoded["attention_mask"])
            self._sync()
            timings["query_encode_ms"] = (time.perf_counter() - encode_start) * 1000.0

            self._sync()
            retrieve_start = time.perf_counter()
            results = self.cluster_retrieval(
                query_embeddings=query_embedding,
                cluster_embeddings=self.cluster_centroids,
                top_k=top_k,
                threshold=similarity_threshold
            )
            self._sync()
            timings["cluster_retrieval_ms"] = (time.perf_counter() - retrieve_start) * 1000.0

            cluster_ids = results["cluster_ids"][0].tolist()
            similarities = results["similarities"][0].tolist()

            outputs: List[ClusterToolCallResult] = []
            mapper_total = 0.0
            arg_infer_total = 0.0
            format_total = 0.0
            for cluster_id, similarity in zip(cluster_ids, similarities):
                if cluster_id < 0:
                    outputs.append(ClusterToolCallResult(
                        tool_name=None,
                        arguments={},
                        tool_call=None,
                        cluster_id=-1,
                        confidence=similarity,
                        needs_clarification=True,
                        missing_required_args=[],
                        timings_ms=timings if return_timings else None
                    ))
                    continue

                mapper_start = time.perf_counter()
                resolved = self.mapper.resolve_cluster_fast(
                    cluster_id=cluster_id,
                    similarity_score=similarity,
                    threshold=similarity_threshold
                )
                mapper_total += (time.perf_counter() - mapper_start)

                if not resolved:
                    outputs.append(ClusterToolCallResult(
                        tool_name=None,
                        arguments={},
                        tool_call=None,
                        cluster_id=cluster_id,
                        confidence=similarity,
                        needs_clarification=True,
                        missing_required_args=[],
                        timings_ms=timings if return_timings else None
                    ))
                    continue

                tool_name = resolved["tool"]
                schema = resolved.get("schema", TOOL_SCHEMAS.get(tool_name, {}))
                required_args = resolved.get("required_args") or get_required_args(tool_name)

                arg_start = time.perf_counter()
                arguments = self._infer_arguments(query, tool_name)
                arg_infer_total += (time.perf_counter() - arg_start)

                missing_required = [arg for arg in required_args if arg not in arguments]
                if force_tool_call and missing_required:
                    for arg_name in missing_required:
                        param_info = schema.get("parameters", {}).get(arg_name, {})
                        arg_type = param_info.get("type", "str")
                        fallback = self.arg_generator.fallback_value(
                            query=query,
                            arg_name=arg_name,
                            arg_type=arg_type,
                            tool_name=tool_name
                        )
                        if fallback is not None:
                            arguments[arg_name] = fallback

                needs_clarification = len(missing_required) > 0
                tool_call = None
                if force_tool_call or not needs_clarification:
                    format_start = time.perf_counter()
                    tool_call = format_tool_call(tool_name, arguments)
                    format_total += (time.perf_counter() - format_start)

                outputs.append(ClusterToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call=tool_call,
                    cluster_id=cluster_id,
                    confidence=similarity,
                    needs_clarification=needs_clarification,
                    missing_required_args=missing_required,
                    timings_ms=timings if return_timings else None
                ))

            timings["mapper_ms"] = mapper_total * 1000.0
            timings["arg_infer_ms"] = arg_infer_total * 1000.0
            timings["format_ms"] = format_total * 1000.0
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

            return outputs[0] if top_k == 1 else outputs

    def predict_batch(
        self,
        queries: List[str],
        top_k: int = 1,
        similarity_threshold: float = 0.5,
        force_tool_call: bool = True
    ) -> List[Union[ClusterToolCallResult, List[ClusterToolCallResult]]]:
        """Batch prediction for multiple queries."""
        with torch.no_grad():
            encoded = self.query_encoder.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            query_embeddings = self.query_encoder(encoded["input_ids"], encoded["attention_mask"])

            results = self.cluster_retrieval(
                query_embeddings=query_embeddings,
                cluster_embeddings=self.cluster_centroids,
                top_k=top_k,
                threshold=similarity_threshold
            )

            all_outputs: List[Union[ClusterToolCallResult, List[ClusterToolCallResult]]] = []
            for i in range(len(queries)):
                cluster_ids = results["cluster_ids"][i].tolist()
                similarities = results["similarities"][i].tolist()

                outputs: List[ClusterToolCallResult] = []
                for cluster_id, similarity in zip(cluster_ids, similarities):
                    if cluster_id < 0:
                        outputs.append(ClusterToolCallResult(
                            tool_name=None,
                            arguments={},
                            tool_call=None,
                            cluster_id=-1,
                            confidence=similarity,
                            needs_clarification=True,
                            missing_required_args=[]
                        ))
                        continue

                    resolved = self.mapper.resolve_cluster_fast(
                        cluster_id=cluster_id,
                        similarity_score=similarity,
                        threshold=similarity_threshold
                    )

                    if not resolved:
                        outputs.append(ClusterToolCallResult(
                            tool_name=None,
                            arguments={},
                            tool_call=None,
                            cluster_id=cluster_id,
                            confidence=similarity,
                            needs_clarification=True,
                            missing_required_args=[]
                        ))
                        continue

                    tool_name = resolved["tool"]
                    schema = resolved.get("schema", TOOL_SCHEMAS.get(tool_name, {}))
                    required_args = resolved.get("required_args") or get_required_args(tool_name)

                    arguments = self._infer_arguments(queries[i], tool_name)

                    missing_required = [arg for arg in required_args if arg not in arguments]
                    if force_tool_call and missing_required:
                        for arg_name in missing_required:
                            param_info = schema.get("parameters", {}).get(arg_name, {})
                            arg_type = param_info.get("type", "str")
                            fallback = self.arg_generator.fallback_value(
                                query=queries[i],
                                arg_name=arg_name,
                                arg_type=arg_type,
                                tool_name=tool_name
                            )
                            if fallback is not None:
                                arguments[arg_name] = fallback

                    needs_clarification = len(missing_required) > 0
                    tool_call = None
                    if force_tool_call or not needs_clarification:
                        tool_call = format_tool_call(tool_name, arguments)

                    outputs.append(ClusterToolCallResult(
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_call=tool_call,
                        cluster_id=cluster_id,
                        confidence=similarity,
                        needs_clarification=needs_clarification,
                        missing_required_args=missing_required
                    ))

                all_outputs.append(outputs[0] if top_k == 1 else outputs)

            return all_outputs


def demo():
    """Demo the cluster-based tool system."""
    intent_path = "checkpoints/best_model.pt"
    query_path = "checkpoints/cluster_retrieval/best_model.pt"

    system = ClusterBasedToolSystem.from_pretrained(
        intent_embedder_path=intent_path,
        query_encoder_path=query_path
    )

    queries = [
        "Find papers about contrastive learning, 5 results",
        "What is 25 plus 37?",
        "Get the last 10 orders from California",
        "Send an email to test@example.com about the meeting",
        "Fetch data from the GitHub API",
        "Calculate the square root of 256",
        "Send a message to John saying happy birthday",
        "Read the file /path/to/data.csv",
    ]

    for q in queries:
        result = system.predict(q, return_timings=True)
        print(f"\nQuery: {q}")
        print(f"Tool: {result.tool_name} (cluster {result.cluster_id}, conf {result.confidence:.2f})")
        # print(f"Arguments: {result.arguments}")
        # print(f"Tool call: {result.tool_call}")
        if result.timings_ms:
            print("Timings (ms):")
            for key, value in result.timings_ms.items():
                print(f"  - {key}: {value:.3f}")
        if result.needs_clarification:
            print(f"Missing required args: {result.missing_required_args}")


if __name__ == "__main__":
    demo()
