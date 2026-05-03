from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import EmbeddingVariantSpec, now_utc_iso, round_float, slugify
from training.dataset_utils import clean_rows, load_dataset_rows, write_json, write_jsonl


DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "MetaTool"
DEFAULT_ABLATION_ROOT = REPO_ROOT / "data" / "ablations" / "MetaTool"
DEFAULT_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_QWEN_DISPLAY_NAME = "Qwen3-Embedding-8B"
DEFAULT_DENSE_QUERY_INSTRUCTION = "Given a user request, retrieve the tool schema that should handle it."
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievalRun:
    adapter_id: str
    provider: str
    mode: str
    model_name: str
    display_name: str
    metadata: dict[str, Any]


class BM25Retriever:
    def __init__(
        self,
        tool_names: Sequence[str],
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(tool_names) != len(documents):
            raise ValueError("tool_names and documents must have the same length.")
        self.tool_names = list(tool_names)
        self.k1 = k1
        self.b = b
        self.doc_term_counts = [Counter(tokenize(document)) for document in documents]
        self.doc_lengths = [sum(counts.values()) for counts in self.doc_term_counts]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for counts in self.doc_term_counts:
            document_frequencies.update(counts.keys())
        self.document_frequencies = document_frequencies

    def rank(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return [(tool_name, 0.0) for tool_name in self.tool_names[:top_k]]

        total_docs = len(self.doc_term_counts)
        scores: list[float] = []
        for counts, doc_length in zip(self.doc_term_counts, self.doc_lengths, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency <= 0:
                    continue
                doc_frequency = self.document_frequencies.get(term, 0)
                idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
                normalizer = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0)
                )
                score += idf * (frequency * (self.k1 + 1.0)) / max(normalizer, 1e-12)
            scores.append(score)

        ranked_indices = sorted(
            range(len(self.tool_names)),
            key=lambda index: (-scores[index], self.tool_names[index]),
        )
        return [
            (self.tool_names[index], round_float(scores[index]) or 0.0)
            for index in ranked_indices[:top_k]
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MetaTool retrieval ablation used by the paper table: BM25, "
            "Qwen dense embeddings, Circle Loss only, and NTILC functional-dispatch loss."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "tool_embedding_dataset.jsonl",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "tool_embedding_dataset_train.jsonl",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "tool_embedding_dataset_test.jsonl",
    )
    parser.add_argument("--tools-path", type=Path, default=DEFAULT_DATASET_ROOT / "tools.json")
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--ranking-limit", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dry-run-training", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compatibility-weight", type=float, default=5.0)
    parser.add_argument("--compatibility-margin", type=float, default=0.5)
    parser.add_argument("--circle-margin", type=float, default=0.25)
    parser.add_argument("--circle-gamma", type=float, default=32.0)
    parser.add_argument("--qwen-embedding-model", default=DEFAULT_QWEN_EMBEDDING_MODEL)
    parser.add_argument("--qwen-display-name", default=DEFAULT_QWEN_DISPLAY_NAME)
    parser.add_argument("--dense-query-instruction", default=DEFAULT_DENSE_QUERY_INSTRUCTION)
    parser.add_argument("--dense-device", default="auto")
    parser.add_argument(
        "--dense-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--dense-max-length", type=int, default=8192)
    parser.add_argument("--dense-local-files-only", action="store_true")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--blur-similarity-threshold", type=float, default=0.35)
    parser.add_argument("--blur-top-n", type=int, default=5)
    return parser.parse_args()


def tokenize(text: Any) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))]


def load_tool_records(path: Path, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tools = payload.get("tools", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_tools, list):
        raise ValueError(f"Expected {path} to contain a list of tools.")

    records_by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        name = str(raw_tool.get("name", raw_tool.get("id", ""))).strip()
        if not name:
            continue
        record = normalize_tool_record(name, raw_tool)
        records_by_name[name] = record
        order.append(name)

    for row in rows:
        name = str(row.get("tool", "")).strip()
        if not name:
            continue
        existing = records_by_name.get(name)
        if existing is None:
            records_by_name[name] = normalize_tool_record(name, row)
            order.append(name)
            continue
        if not existing.get("description") and row.get("tool_description"):
            existing["description"] = str(row.get("tool_description", "")).strip()
        if not existing.get("parameters") and isinstance(row.get("parameters"), dict):
            existing["parameters"] = row["parameters"]

    return [records_by_name[name] for name in order if name in records_by_name]


def normalize_tool_record(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": str(
            payload.get("description", payload.get("tool_description", ""))
        ).strip(),
        "parameters": payload.get("parameters", {}) if isinstance(payload.get("parameters", {}), dict) else {},
    }


def normalize_schema_type(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(sorted(normalize_schema_type(item) for item in value if str(item).strip()))
    return str(value or "any").strip().lower()


def signature_atoms(tool: dict[str, Any]) -> tuple[str, ...]:
    parameters = tool.get("parameters", {})
    if not isinstance(parameters, dict):
        return tuple()
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    required_set = {str(item).strip().lower() for item in required if str(item).strip()}
    atoms: list[str] = []
    property_names: set[str] = set()
    for raw_name, raw_spec in sorted(properties.items(), key=lambda item: str(item[0]).lower()):
        name = str(raw_name).strip().lower()
        if not name:
            continue
        property_names.add(name)
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        requirement = "required" if name in required_set else "optional"
        field_type = normalize_schema_type(spec.get("type", "any"))
        enum_values = spec.get("enum", [])
        enum_atom = ""
        if isinstance(enum_values, list) and enum_values:
            enum_atom = "|".join(sorted(str(value).strip().lower() for value in enum_values if str(value).strip()))
        atoms.append(f"{requirement}:{name}:{field_type}:{enum_atom}")
    for missing_required_name in sorted(required_set - property_names):
        atoms.append(f"required:{missing_required_name}:any:")
    return tuple(atoms)


def signatures_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return signature_atoms(left) == signature_atoms(right)


def render_argument_summary(tool: dict[str, Any]) -> str:
    parameters = tool.get("parameters", {})
    if not isinstance(parameters, dict):
        return "no arguments"
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not properties:
        return "no arguments"
    required_set = {str(item).strip() for item in required if str(item).strip()}
    parts: list[str] = []
    for name in sorted(properties):
        spec = properties[name] if isinstance(properties[name], dict) else {}
        marker = " required" if name in required_set else " optional"
        field_type = normalize_schema_type(spec.get("type", "any"))
        parts.append(f"{name} {field_type}{marker}")
    return "; ".join(parts)


def render_tool_text(tool: dict[str, Any], *, include_arguments: bool = True) -> str:
    parts = [
        str(tool.get("name", "")).strip(),
        str(tool.get("description", "")).strip(),
    ]
    if include_arguments:
        parts.append(render_argument_summary(tool))
    return " ".join(part for part in parts if part).strip()


def build_semantic_blur_index(
    tools: Sequence[dict[str, Any]],
    *,
    similarity_threshold: float,
    top_n: int,
) -> dict[str, Any]:
    tool_names = [str(tool["name"]) for tool in tools]
    texts = [render_tool_text(tool, include_arguments=False) for tool in tools]
    if len(tools) < 2:
        return {"blur_tools": {}, "pairs": []}

    vectors = build_tfidf_vectors(texts)
    blur_tools: dict[str, list[dict[str, Any]]] = {name: [] for name in tool_names}
    pairs: list[dict[str, Any]] = []

    for left_index, left_tool in enumerate(tools):
        candidates: list[tuple[float, int]] = []
        for right_index, right_tool in enumerate(tools):
            if left_index == right_index:
                continue
            similarity = sparse_cosine(vectors[left_index], vectors[right_index])
            if similarity < similarity_threshold:
                continue
            if signatures_compatible(left_tool, right_tool):
                continue
            candidates.append((similarity, right_index))
        candidates.sort(key=lambda item: (-item[0], tool_names[item[1]]))
        for similarity, right_index in candidates[:top_n]:
            right_name = tool_names[right_index]
            blur_tools[tool_names[left_index]].append(
                {"tool": right_name, "similarity": round_float(similarity)}
            )
            if tool_names[left_index] < right_name:
                pairs.append(
                    {
                        "left_tool": tool_names[left_index],
                        "right_tool": right_name,
                        "similarity": round_float(similarity),
                    }
                )

    return {
        "blur_tools": {name: items for name, items in blur_tools.items() if items},
        "pairs": pairs,
    }


def build_tfidf_vectors(texts: Sequence[str]) -> list[dict[str, float]]:
    term_counts = [Counter(tokenize(text)) for text in texts]
    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())
    document_count = len(texts)

    vectors: list[dict[str, float]] = []
    for counts in term_counts:
        weighted: dict[str, float] = {}
        total_terms = sum(counts.values()) or 1
        for term, count in counts.items():
            tf = count / total_terms
            idf = math.log((1.0 + document_count) / (1.0 + document_frequency[term])) + 1.0
            weighted[term] = tf * idf
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        if norm > 0:
            weighted = {term: value / norm for term, value in weighted.items()}
        vectors.append(weighted)
    return vectors


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def is_signature_error(
    tool_lookup: dict[str, dict[str, Any]],
    expected_tool: str,
    selected_tool: str,
) -> bool:
    if expected_tool == selected_tool:
        return False
    expected = tool_lookup.get(expected_tool)
    selected = tool_lookup.get(selected_tool)
    if expected is None or selected is None:
        return True
    return not signatures_compatible(expected, selected)


def evaluate_ranker(
    run: RetrievalRun,
    rows: Sequence[dict[str, Any]],
    tool_lookup: dict[str, dict[str, Any]],
    blur_tools: dict[str, list[dict[str, Any]]],
    rank_fn: Callable[[str], list[tuple[str, float | None]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    valid_tool_names = set(tool_lookup)
    for row in rows:
        query = str(row.get("query", row.get("text", ""))).strip()
        expected_tool = str(row.get("tool", "")).strip()
        start_time = time.perf_counter()
        try:
            ranked_pairs = rank_fn(query)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            ranked_tools = [
                tool_name
                for tool_name, _score in ranked_pairs
                if tool_name in valid_tool_names
            ]
            if not ranked_tools:
                raise ValueError("Ranker returned no valid tools.")
            selected_tool = ranked_tools[0]
            signature_error = is_signature_error(tool_lookup, expected_tool, selected_tool)
            semantic_blur_case = expected_tool in blur_tools
            results.append(
                {
                    "adapter_id": run.adapter_id,
                    "provider": run.provider,
                    "mode": run.mode,
                    "model_name": run.model_name,
                    "display_name": run.display_name,
                    "example_id": str(row.get("id", "")),
                    "query": query,
                    "expected_tool": expected_tool,
                    "status": "ok",
                    "selected_tool": selected_tool,
                    "ranked_tools": ranked_tools,
                    "correct_top1": selected_tool == expected_tool,
                    "top_5_hit": expected_tool in ranked_tools[:5],
                    "semantic_blur_case": semantic_blur_case,
                    "semantic_blur_hit": (selected_tool == expected_tool) if semantic_blur_case else None,
                    "signature_error": signature_error,
                    "signature_compatible": not signature_error,
                    "latency_ms": round_float(latency_ms),
                    "score_candidates": [
                        {"tool": tool_name, "score": round_float(score)}
                        for tool_name, score in ranked_pairs
                    ],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "adapter_id": run.adapter_id,
                    "provider": run.provider,
                    "mode": run.mode,
                    "model_name": run.model_name,
                    "display_name": run.display_name,
                    "example_id": str(row.get("id", "")),
                    "query": query,
                    "expected_tool": expected_tool,
                    "status": "error",
                    "selected_tool": None,
                    "ranked_tools": [],
                    "correct_top1": None,
                    "top_5_hit": None,
                    "semantic_blur_case": expected_tool in blur_tools,
                    "semantic_blur_hit": None,
                    "signature_error": None,
                    "signature_compatible": None,
                    "latency_ms": round_float((time.perf_counter() - start_time) * 1000.0),
                    "error_message": str(exc),
                }
            )
    return results


def summarize_ablation_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    blur_rows = [row for row in ok_rows if row.get("semantic_blur_case")]

    def mean_bool(items: Sequence[dict[str, Any]], key: str) -> float | None:
        if not items:
            return None
        return round_float(sum(1.0 if item.get(key) else 0.0 for item in items) / len(items))

    return {
        "total_examples": len(rows),
        "successful_examples": len(ok_rows),
        "error_examples": len(rows) - len(ok_rows),
        "semantic_blur_examples": len(blur_rows),
        "top_1_accuracy": mean_bool(ok_rows, "correct_top1"),
        "top_5_accuracy": mean_bool(ok_rows, "top_5_hit"),
        "semantic_blur_accuracy": mean_bool(blur_rows, "semantic_blur_hit"),
        "signature_error_rate": mean_bool(ok_rows, "signature_error"),
        "mean_latency_ms": mean_or_none(row.get("latency_ms") for row in ok_rows),
    }


def mean_or_none(values: Any) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round_float(sum(cleaned) / len(cleaned))


def build_model_summary(
    run: RetrievalRun,
    results: Sequence[dict[str, Any]],
    *,
    status: str = "ok",
    error_message: str | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "adapter_id": run.adapter_id,
        "provider": run.provider,
        "mode": run.mode,
        "model_name": run.model_name,
        "display_name": run.display_name,
        "status": status,
        "error_message": error_message,
        "metrics": summarize_ablation_results(results) if status == "ok" else None,
        "results_path": str(results_path.resolve()) if results_path is not None else "",
        "metadata": run.metadata,
    }


def build_error_summary(run: RetrievalRun, message: str) -> dict[str, Any]:
    return build_model_summary(
        run,
        [],
        status="error",
        error_message=message,
        results_path=None,
    )


def evaluate_bm25(
    rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    tool_lookup: dict[str, dict[str, Any]],
    blur_tools: dict[str, list[dict[str, Any]]],
    *,
    ranking_limit: int,
    k1: float,
    b: float,
) -> tuple[RetrievalRun, list[dict[str, Any]]]:
    run = RetrievalRun(
        adapter_id="bm25/tool-schema",
        provider="bm25",
        mode="sparse_retrieval",
        model_name="BM25",
        display_name="BM25",
        metadata={"k1": k1, "b": b},
    )
    retriever = BM25Retriever(
        [tool["name"] for tool in tools],
        [render_tool_text(tool) for tool in tools],
        k1=k1,
        b=b,
    )
    results = evaluate_ranker(
        run,
        rows,
        tool_lookup,
        blur_tools,
        lambda query: retriever.rank(query, top_k=ranking_limit),
    )
    return run, results


def evaluate_embedding_checkpoint(
    checkpoint_path: Path,
    rows: Sequence[dict[str, Any]],
    tool_lookup: dict[str, dict[str, Any]],
    blur_tools: dict[str, list[dict[str, Any]]],
    *,
    architecture: str,
    loss_name: str,
    display_name: str,
    embedding_device: str,
    ranking_limit: int,
) -> tuple[RetrievalRun, list[dict[str, Any]]] | tuple[RetrievalRun, None]:
    variant_id = f"{architecture}/{loss_name}"
    run = RetrievalRun(
        adapter_id=f"embedding/{slugify(variant_id)}",
        provider="embedding",
        mode="embedding",
        model_name=variant_id,
        display_name=display_name,
        metadata={
            "architecture": architecture,
            "loss_name": loss_name,
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )
    if not checkpoint_path.is_file():
        return run, None

    from benchmark.adapters import EmbeddingSelectorEngine

    engine = EmbeddingSelectorEngine(
        EmbeddingVariantSpec(
            variant_id=variant_id,
            architecture=architecture,
            loss_name=loss_name,
            checkpoint_path=checkpoint_path,
        ),
        device=embedding_device,
        ranking_limit=ranking_limit,
    )

    def rank(query: str) -> list[tuple[str, float | None]]:
        payload = engine.select(query)
        candidates = payload.get("score_candidates", [])
        if candidates:
            return [
                (str(item.get("tool", "")).strip(), item.get("score"))
                for item in candidates
                if str(item.get("tool", "")).strip()
            ]
        return [(tool_name, None) for tool_name in payload["ranked_tools"]]

    return run, evaluate_ranker(run, rows, tool_lookup, blur_tools, rank)


def evaluate_dense_embeddings(
    rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    tool_lookup: dict[str, dict[str, Any]],
    blur_tools: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    display_name: str,
    model_root: Path,
    ranking_limit: int,
    device: str,
    dtype: str,
    batch_size: int,
    max_length: int,
    query_instruction: str,
    local_files_only: bool,
) -> tuple[RetrievalRun, list[dict[str, Any]]]:
    run = RetrievalRun(
        adapter_id=f"dense/{slugify(display_name)}",
        provider="qwen_embedding",
        mode="dense_retrieval",
        model_name=model_name,
        display_name=display_name,
        metadata={
            "embedding_model": model_name,
            "dense_device": device,
            "dense_dtype": dtype,
            "dense_max_length": max_length,
            "pooling": "last_token",
            "query_instruction": query_instruction,
            "local_files_only": local_files_only,
        },
    )

    import torch

    tool_names = [tool["name"] for tool in tools]
    tool_texts = [render_tool_text(tool) for tool in tools]
    dense_model_dir = model_root / "dense" / slugify(display_name)
    dense_model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dense_model_dir / "tool_embeddings.pt"
    text_hash = hashlib.sha256(
        json.dumps(
            {"model": model_name, "tool_names": tool_names, "tool_texts": tool_texts},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    tokenizer, model, input_device = load_dense_encoder(
        model_name=model_name,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    tool_embeddings = load_cached_tool_embeddings(cache_path, text_hash)
    if tool_embeddings is None:
        tool_embeddings = encode_with_dense_model(
            tokenizer,
            model,
            input_device,
            tool_texts,
            batch_size=batch_size,
            max_length=max_length,
            progress_desc=f"Embedding {display_name} tools",
        )
        torch.save(
            {
                "model_name": model_name,
                "tool_names": tool_names,
                "tool_text_hash": text_hash,
                "embeddings": tool_embeddings,
            },
            cache_path,
        )

    query_texts = [
        format_dense_query(
            str(row.get("query", row.get("text", ""))).strip(),
            instruction=query_instruction,
        )
        for row in rows
    ]
    start_time = time.perf_counter()
    query_embeddings = encode_with_dense_model(
        tokenizer,
        model,
        input_device,
        query_texts,
        batch_size=batch_size,
        max_length=max_length,
        progress_desc=f"Embedding {display_name} queries",
    )
    scores = query_embeddings @ tool_embeddings.T
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_row = total_latency_ms / max(1, len(rows))

    def rank_for_index(index: int) -> list[tuple[str, float | None]]:
        ranked_indices = torch.argsort(scores[index], descending=True).tolist()
        return [
            (tool_names[candidate_index], float(scores[index, candidate_index].item()))
            for candidate_index in ranked_indices[:ranking_limit]
        ]

    ranked_by_row = [rank_for_index(index) for index in range(len(rows))]
    result_rows: list[dict[str, Any]] = []
    for row, ranked_pairs in zip(rows, ranked_by_row, strict=True):
        result_rows.extend(
            evaluate_ranker(
                run,
                [row],
                tool_lookup,
                blur_tools,
                lambda _query, pairs=ranked_pairs: pairs,
            )
        )
        result_rows[-1]["latency_ms"] = round_float(latency_per_row)
    run.metadata["tool_embedding_cache_path"] = str(cache_path.resolve())
    return run, result_rows


def load_cached_tool_embeddings(cache_path: Path, text_hash: str) -> Any | None:
    if not cache_path.is_file():
        return None
    import torch

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    if payload.get("tool_text_hash") != text_hash:
        return None
    embeddings = payload.get("embeddings")
    return embeddings if embeddings is not None else None


def load_dense_encoder(
    *,
    model_name: str,
    device: str,
    dtype: str,
    local_files_only: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    torch_dtype = resolve_dense_dtype(dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    requested_device = str(device).strip().lower()
    if requested_device == "auto" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        input_device = next(model.parameters()).device
    else:
        resolved_device = torch.device(
            "cpu"
            if requested_device == "auto"
            else device
        )
        model = AutoModel.from_pretrained(model_name, **model_kwargs).to(resolved_device)
        input_device = resolved_device
    model.eval()
    return tokenizer, model, input_device


def resolve_dense_dtype(dtype: str) -> Any | None:
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype)


def encode_with_dense_model(
    tokenizer: Any,
    model: Any,
    input_device: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    progress_desc: str,
) -> Any:
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    chunks: list[Any] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=progress_desc,
            unit="batch",
            leave=False,
        ):
            batch_texts = list(texts[start:start + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            outputs = model(**encoded)
            last_hidden_state = getattr(outputs, "last_hidden_state", outputs[0])
            pooled = last_token_pool(last_hidden_state, encoded["attention_mask"])
            chunks.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def last_token_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def format_dense_query(query: str, *, instruction: str) -> str:
    stripped_instruction = str(instruction).strip()
    if not stripped_instruction:
        return query
    return f"Instruct: {stripped_instruction}\nQuery:{query}"


def train_missing_checkpoints(args: argparse.Namespace, model_root: Path) -> None:
    losses = ("circle", "functional_margin")
    for loss_name in losses:
        checkpoint_path = model_root / "normal" / loss_name / "best.pt"
        if checkpoint_path.is_file() and not args.force_train:
            print(f"Checkpoint exists, skipping training: {checkpoint_path}")
            continue
        if not args.train_missing and not args.force_train:
            continue
        command = [
            sys.executable,
            "-m",
            "training.train_embedding_space",
            "--dataset-path",
            str(args.dataset_path),
            "--train-dataset-path",
            str(args.train_dataset_path),
            "--test-dataset-path",
            str(args.test_dataset_path),
            "--tools-path",
            str(args.tools_path),
            "--output-dir",
            str(model_root),
            "--loss-type",
            loss_name,
            "--encoder-model",
            args.encoder_model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--max-length",
            str(args.max_length),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--device",
            args.train_device,
            "--circle-margin",
            str(args.circle_margin),
            "--circle-gamma",
            str(args.circle_gamma),
            "--compatibility-weight",
            str(args.compatibility_weight),
            "--compatibility-margin",
            str(args.compatibility_margin),
        ]
        print("Training checkpoint:")
        print(" ".join(command))
        if args.dry_run_training:
            continue
        subprocess.run(command, check=True, cwd=REPO_ROOT)


def persist_model(
    output_root: Path,
    run: RetrievalRun,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    results_dir = output_root / "results"
    summaries_dir = output_root / "model_summaries"
    results_path = results_dir / f"{slugify(run.adapter_id)}.jsonl"
    summary_path = summaries_dir / f"{slugify(run.adapter_id)}.json"
    write_jsonl(results_path, results)
    summary = build_model_summary(run, results, results_path=results_path)
    write_json(summary_path, summary)
    return summary


def persist_error_model(output_root: Path, run: RetrievalRun, message: str) -> dict[str, Any]:
    summaries_dir = output_root / "model_summaries"
    summary_path = summaries_dir / f"{slugify(run.adapter_id)}.json"
    summary = build_error_summary(run, message)
    write_json(summary_path, summary)
    return summary


def format_percent(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        text = r"\placeholder{X}\%"
    else:
        text = f"{value * 100.0:.1f}\\%"
    return rf"\textbf{{{text}}}" if bold else text


def build_table_rows(model_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_display_name = {
        str(summary.get("display_name", "")): summary
        for summary in model_summaries
    }
    dense_display_name = next(
        (
            str(summary.get("display_name", ""))
            for summary in model_summaries
            if summary.get("provider") == "qwen_embedding"
        ),
        DEFAULT_QWEN_DISPLAY_NAME,
    )
    order = [
        ("BM25", "BM25", False),
        (dense_display_name, dense_display_name, False),
        ("Circle Loss only", "Circle Loss only", False),
        (r"\textbf{NTILC ($\mathcal{L}_{FD}$)}", "NTILC ($\\mathcal{L}_{FD}$)", True),
    ]
    rows: list[dict[str, Any]] = []
    for latex_label, display_name, bold in order:
        summary = by_display_name.get(display_name)
        metrics = summary.get("metrics") if isinstance(summary, dict) else None
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "label": latex_label,
                "display_name": display_name,
                "bold": bold,
                "top_1_accuracy": metrics.get("top_1_accuracy"),
                "top_5_accuracy": metrics.get("top_5_accuracy"),
                "semantic_blur_accuracy": metrics.get("semantic_blur_accuracy"),
                "signature_error_rate": metrics.get("signature_error_rate"),
                "status": summary.get("status") if isinstance(summary, dict) else "missing",
            }
        )
    return rows


def write_table_outputs(output_root: Path, model_summaries: Sequence[dict[str, Any]]) -> None:
    table_rows = build_table_rows(model_summaries)
    csv_rows = [
        {
            "retriever_loss": row["display_name"],
            "status": row["status"],
            "top_1_accuracy_percent": value_to_percent(row["top_1_accuracy"]),
            "top_5_accuracy_percent": value_to_percent(row["top_5_accuracy"]),
            "semantic_blur_accuracy_percent": value_to_percent(row["semantic_blur_accuracy"]),
            "signature_error_rate_percent": value_to_percent(row["signature_error_rate"]),
        }
        for row in table_rows
    ]
    write_csv(
        output_root / "retrieval_ablation_table.csv",
        csv_rows,
        fieldnames=(
            "retriever_loss",
            "status",
            "top_1_accuracy_percent",
            "top_5_accuracy_percent",
            "semantic_blur_accuracy_percent",
            "signature_error_rate_percent",
        ),
    )

    lines = [
        r"\begin{table}[H]",
        r"\caption{Tool-retrieval ablation on MetaTool test split. Signature error rate measures the fraction of selected tools whose argument contract is incompatible with the gold tool.}",
        r"\label{tab:retrieval}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Retriever / Loss & Top-1 Acc. & Top-5 Acc. & Semantic-Blur Acc. & Signature Error Rate \\",
        r"\midrule",
    ]
    for row in table_rows:
        lines.append(
            " & ".join(
                [
                    row["label"],
                    format_percent(row["top_1_accuracy"], bold=row["bold"]),
                    format_percent(row["top_5_accuracy"], bold=row["bold"]),
                    format_percent(row["semantic_blur_accuracy"], bold=row["bold"]),
                    format_percent(row["signature_error_rate"], bold=row["bold"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output_root / "retrieval_ablation_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def value_to_percent(value: Any) -> float | None:
    if value is None:
        return None
    return round_float(float(value) * 100.0)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    model_root = (args.model_root or args.ablation_root / "models").expanduser()
    output_root = (args.output_root or args.ablation_root / "outputs").expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    train_missing_checkpoints(args, model_root)

    all_rows = clean_rows(load_dataset_rows(args.dataset_path))
    test_rows = clean_rows(load_dataset_rows(args.test_dataset_path))
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    tools = load_tool_records(args.tools_path, all_rows)
    tool_lookup = {str(tool["name"]): tool for tool in tools}
    missing_tools = sorted({row["tool"] for row in test_rows} - set(tool_lookup))
    if missing_tools:
        raise ValueError("Test rows reference tools missing from tools.json: " + ", ".join(missing_tools))

    blur_index = build_semantic_blur_index(
        tools,
        similarity_threshold=args.blur_similarity_threshold,
        top_n=args.blur_top_n,
    )
    blur_tools = blur_index["blur_tools"]
    write_json(output_root / "semantic_blur_index.json", blur_index)

    model_summaries: list[dict[str, Any]] = []

    bm25_run, bm25_results = evaluate_bm25(
        test_rows,
        tools,
        tool_lookup,
        blur_tools,
        ranking_limit=args.ranking_limit,
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    model_summaries.append(persist_model(output_root, bm25_run, bm25_results))

    if not args.skip_dense:
        try:
            dense_run, dense_results = evaluate_dense_embeddings(
                test_rows,
                tools,
                tool_lookup,
                blur_tools,
                model_name=args.qwen_embedding_model,
                display_name=args.qwen_display_name,
                model_root=model_root,
                ranking_limit=args.ranking_limit,
                device=args.dense_device,
                dtype=args.dense_dtype,
                batch_size=args.dense_batch_size,
                max_length=args.dense_max_length,
                query_instruction=args.dense_query_instruction,
                local_files_only=args.dense_local_files_only,
            )
            model_summaries.append(persist_model(output_root, dense_run, dense_results))
        except Exception as exc:
            dense_run = RetrievalRun(
                adapter_id=f"dense/{slugify(args.qwen_display_name)}",
                provider="qwen_embedding",
                mode="dense_retrieval",
                model_name=args.qwen_embedding_model,
                display_name=args.qwen_display_name,
                metadata={"embedding_model": args.qwen_embedding_model},
            )
            model_summaries.append(persist_error_model(output_root, dense_run, str(exc)))

    if not args.skip_checkpoints:
        checkpoint_specs = [
            ("circle", "Circle Loss only", "normal", model_root / "normal" / "circle" / "best.pt"),
            (
                "functional_margin",
                "NTILC ($\\mathcal{L}_{FD}$)",
                "normal",
                model_root / "normal" / "functional_margin" / "best.pt",
            ),
        ]
        for loss_name, display_name, architecture, checkpoint_path in checkpoint_specs:
            checkpoint_run, checkpoint_results = evaluate_embedding_checkpoint(
                checkpoint_path,
                test_rows,
                tool_lookup,
                blur_tools,
                architecture=architecture,
                loss_name=loss_name,
                display_name=display_name,
                embedding_device=args.embedding_device,
                ranking_limit=args.ranking_limit,
            )
            if checkpoint_results is None:
                model_summaries.append(
                    persist_error_model(
                        output_root,
                        checkpoint_run,
                        f"Checkpoint not found: {checkpoint_path}",
                    )
                )
            else:
                model_summaries.append(persist_model(output_root, checkpoint_run, checkpoint_results))

    summary = {
        "created_at": now_utc_iso(),
        "benchmark": "metatool_retrieval_ablation",
        "paths": {
            "dataset_path": str(args.dataset_path.resolve()),
            "train_dataset_path": str(args.train_dataset_path.resolve()),
            "test_dataset_path": str(args.test_dataset_path.resolve()),
            "tools_path": str(args.tools_path.resolve()),
            "model_root": str(model_root.resolve()),
            "output_root": str(output_root.resolve()),
        },
        "config": {
            "ranking_limit": args.ranking_limit,
            "limit": args.limit,
            "semantic_blur_similarity_threshold": args.blur_similarity_threshold,
            "semantic_blur_top_n": args.blur_top_n,
            "semantic_blur_tool_count": len(blur_tools),
            "semantic_blur_pair_count": len(blur_index["pairs"]),
            "test_examples": len(test_rows),
            "tool_count": len(tools),
        },
        "models": model_summaries,
        "table_rows": build_table_rows(model_summaries),
    }
    write_json(output_root / "summary.json", summary)
    write_table_outputs(output_root, model_summaries)

    print(f"Wrote summary: {output_root / 'summary.json'}")
    print(f"Wrote LaTeX table: {output_root / 'retrieval_ablation_table.tex'}")
    print(f"Wrote CSV table: {output_root / 'retrieval_ablation_table.csv'}")


if __name__ == "__main__":
    main()
