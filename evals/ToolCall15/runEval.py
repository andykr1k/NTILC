"""Run the ToolCall15 benchmark with the notebook-style planner/retriever/dispatcher."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import outlines
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, create_model
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import embed_texts, load_checkpoint_bundle


DATA_DIR = REPO_ROOT / "data" / "ToolCall15"
DEFAULT_CHECKPOINT_ROOT = DATA_DIR / "output"
DEFAULT_BENCHMARK_PATH = DATA_DIR / "benchmark.json"
DEFAULT_TOOLS_PATH = DATA_DIR / "tools.json"
DEFAULT_OUTPUT_PATH = DEFAULT_CHECKPOINT_ROOT / "eval" / "eval_summary.json"
DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen3.5-27B"
DEFAULT_QWEN_DTYPE = "bfloat16"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    architecture: str
    loss_name: str
    checkpoint_path: Path
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved ToolCall15 embedding-space variant using the same "
            "notebook-style flow as agent.ipynb: Qwen planning, embedding-based tool "
            "selection, and Qwen dispatch generation."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional single checkpoint to evaluate. When omitted, all embedding-space variants are discovered under --checkpoint-root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root directory containing saved embedding-space checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default="best.pt",
        help="Checkpoint filename to evaluate for each discovered variant.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the ToolCall15 benchmark.json file.",
    )
    parser.add_argument(
        "--tools-path",
        type=Path,
        default=DEFAULT_TOOLS_PATH,
        help="Path to the ToolCall15 tools.json file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the eval summary JSON.",
    )
    parser.add_argument(
        "--qwen-model-name",
        default=DEFAULT_QWEN_MODEL_NAME,
        help="Hugging Face model name or local path for the planner/dispatcher model.",
    )
    parser.add_argument(
        "--qwen-device",
        default="cuda:6",
        help='Torch device for Qwen. Uses a CUDA device when available if set to "auto".',
    )
    parser.add_argument(
        "--embed-device",
        default="cuda:6",
        help='Torch device for the embedding model. Uses a CUDA device when available if set to "auto".',
    )
    parser.add_argument(
        "--qwen-dtype",
        default=DEFAULT_QWEN_DTYPE,
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for loading the Qwen model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieved tools to keep per plan action.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=10,
        help="Maximum number of planner actions allowed per request.",
    )
    parser.add_argument(
        "--planner-max-new-tokens",
        type=int,
        default=96,
        help="Max new tokens for planner generation.",
    )
    parser.add_argument(
        "--dispatcher-max-new-tokens",
        type=int,
        default=160,
        help="Max new tokens for dispatcher generation.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size for embedding plan actions.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require model/tokenizer loads to come from local cache only.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_runtime_devices(qwen_device: str, embed_device: str) -> tuple[str, str]:
    if torch.cuda.is_available():
        if qwen_device == "auto":
            qwen_device = "cuda:0"
        if embed_device == "auto":
            embed_device = "cuda:1" if torch.cuda.device_count() > 1 else qwen_device
        return qwen_device, embed_device

    if qwen_device == "auto":
        qwen_device = "cpu"
    if embed_device == "auto":
        embed_device = "cpu"
    return qwen_device, embed_device


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round_float(sum(values) / len(values))


def first_numeric(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def max_numeric(history: list[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in history:
        value = first_numeric(row, *keys)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return max(values)


def safe_metric_sort_value(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


JSON_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"')


def normalize_outline_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def is_subsequence(sequence: list[str], subsequence: list[str]) -> bool:
    if not subsequence:
        return True
    next_index = 0
    for value in sequence:
        if value == subsequence[next_index]:
            next_index += 1
            if next_index == len(subsequence):
                return True
    return False


def reciprocal_rank(ranked_tools: list[str], expected_tools: list[str]) -> float:
    expected_set = set(expected_tools)
    for index, tool_name in enumerate(ranked_tools, start=1):
        if tool_name in expected_set:
            return 1.0 / float(index)
    return 0.0


def build_plan_output_model(max_actions: int) -> type[BaseModel]:
    class PlanOutput(BaseModel):
        actions: list[str] = Field(..., min_length=1, max_length=max_actions)

    return PlanOutput


def normalize_plan_actions(
    actions: list[str],
    user_request: str,
    max_actions: int,
) -> list[str]:
    deduped: list[str] = []
    for action in actions:
        cleaned = normalize_outline_text(action)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    fallback = normalize_outline_text(user_request)
    return deduped[:max_actions] or [fallback]


def build_plan_block(actions: list[str]) -> str:
    lines = ["<plan>"]
    for action in actions:
        lines.append(f"  <action><len:{len(action)}>{action}</len></action>")
    lines.append("</plan>")
    return "\n".join(lines)


def plan_actions_look_meaningful(actions: list[str]) -> bool:
    return any(re.search(r"[A-Za-z]", action) for action in actions)


def extract_complete_json_strings(raw_json: str) -> list[str]:
    values: list[str] = []
    for match in JSON_STRING_PATTERN.finditer(raw_json):
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, str):
            values.append(decoded)
    return values


def recover_plan_actions(raw_plan_json: str) -> list[str]:
    recovered = extract_complete_json_strings(raw_plan_json)
    if recovered and recovered[0] == "actions":
        recovered = recovered[1:]
    return recovered


def parse_plan_actions(
    runtime: "AgentRuntime",
    user_request: str,
    raw_plan_json: str,
) -> tuple[list[str], str | None]:
    try:
        plan_payload = runtime.plan_output_model.model_validate_json(raw_plan_json)
    except Exception:
        plan_payload = None
    else:
        parsed_actions = normalize_plan_actions(
            plan_payload.actions,
            user_request=user_request,
            max_actions=runtime.max_actions,
        )
        if plan_actions_look_meaningful(parsed_actions):
            return parsed_actions, None

    recovered_actions = normalize_plan_actions(
        recover_plan_actions(raw_plan_json),
        user_request=user_request,
        max_actions=runtime.max_actions,
    )
    if plan_actions_look_meaningful(recovered_actions):
        return recovered_actions, "Recovered planner actions from malformed JSON output."

    return [normalize_outline_text(user_request)], (
        "Planner output was malformed or degenerate; fell back to the user request."
    )


def build_planner_prompt(user_request: str, max_actions: int) -> str:
    return f"""You are the planner inside NTILC.
Return a plan with an actions field containing at most {max_actions} short atomic natural language actions.
Requirements:
- Actions must be plain English task steps, not tool calls.
- Do not use tool names, API names, function names, schema fields, or argument names.
- Each action must be a single line.
- Include the final user-facing step when needed.

Example:
User request: get me the top news today and email to akrik@umich.edu
actions:
- get today's date
- get news with today's date
- compose email with todays news and send to akrik@umich.edu

Do not include commentary, reasoning, numbering, or extra fields.

User request:
{user_request}
"""


def schema_type_to_annotation(spec: dict[str, Any]) -> Any:
    schema_type = spec.get("type", "string")
    enum_values = spec.get("enum")
    if enum_values:
        return Literal.__getitem__(tuple(enum_values))
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        item_spec = spec.get("items", {})
        item_annotation = schema_type_to_annotation(
            item_spec if isinstance(item_spec, dict) else {"type": "string"}
        )
        return list[item_annotation]
    return Any


def build_dispatch_output_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[Any, Field]] = {}

    for name, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {"type": "string"}
        annotation = schema_type_to_annotation(spec)
        if name in required:
            fields[name] = (annotation, Field(...))
        else:
            fields[name] = (annotation | None, Field(default=spec.get("default")))

    safe_name = "".join(
        part.capitalize() for part in re.split(r"[^0-9A-Za-z]+", tool_name) if part
    ) or "Tool"
    return create_model(f"{safe_name}Dispatch", **fields)


def normalize_dispatch_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, str):
            normalized[name] = normalize_outline_text(value)
        elif isinstance(value, list):
            normalized[name] = [
                normalize_outline_text(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            normalized[name] = value
    return normalized


def build_dispatch_block(tool_name: str, arguments: dict[str, Any]) -> str:
    lines = ["<dispatch>", f"  <tool><len:{len(tool_name)}>{tool_name}</len></tool>"]
    for name, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        lines.append(
            f'  <arg name="{name}"><len:{len(rendered)}>{rendered}</len></arg>'
        )
    lines.append("</dispatch>")
    return "\n".join(lines)


def build_dispatcher_prompt(
    user_request: str,
    action: str,
    tool_name: str,
    schema: dict[str, Any],
) -> str:
    return f"""You are the NTILC dispatcher.
Return only the arguments for the selected tool.
Use the provided schema exactly.
Do not include explanation or extra fields.

User request:
{user_request}

Current action:
{action}

Selected tool:
{tool_name}

Tool schema:
{json.dumps(schema, indent=2)}
"""


def build_generation_kwargs(
    qwen_tokenizer,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": qwen_tokenizer.pad_token_id,
        "eos_token_id": qwen_tokenizer.eos_token_id,
        "do_sample": False,
    }


@dataclass
class SharedRuntime:
    tool_by_name: dict[str, dict[str, Any]]
    registry_tool_names: list[str]
    qwen_model_name: str
    qwen_device: str
    qwen_dtype: str
    qwen_tokenizer: Any
    structured_qwen: Any
    plan_output_model: type[BaseModel]
    planner_generator: Any
    planner_max_new_tokens: int
    dispatcher_max_new_tokens: int
    max_actions: int
    top_k: int
    embedding_batch_size: int
    dispatch_generator_cache: dict[str, tuple[type[BaseModel], Any]] = field(default_factory=dict)


@dataclass
class AgentRuntime:
    embed_model: Any
    embed_tokenizer: Any
    embed_max_length: int
    embed_device: str
    tool_names: list[str]
    tool_centroids: torch.Tensor
    checkpoint_architecture: str | None
    checkpoint_loss_name: str | None
    encoder_model: str | None
    tool_by_name: dict[str, dict[str, Any]]
    qwen_model_name: str
    qwen_device: str
    qwen_dtype: str
    qwen_tokenizer: Any
    structured_qwen: Any
    plan_output_model: type[BaseModel]
    planner_generator: Any
    planner_max_new_tokens: int
    dispatcher_max_new_tokens: int
    max_actions: int
    top_k: int
    embedding_batch_size: int
    dispatch_generator_cache: dict[str, tuple[type[BaseModel], Any]] = field(default_factory=dict)


def build_shared_runtime(args: argparse.Namespace) -> SharedRuntime:
    qwen_device, embed_device = resolve_runtime_devices(args.qwen_device, args.embed_device)

    tools_payload = load_json(args.tools_path)
    tool_registry = tools_payload.get("tools", [])
    tool_by_name = {
        str(tool["name"]): tool
        for tool in tool_registry
        if isinstance(tool, dict) and "name" in tool
    }

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token or qwen_tokenizer.unk_token

    qwen_model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": qwen_device},
        "local_files_only": args.local_files_only,
    }
    qwen_dtype = resolve_torch_dtype(args.qwen_dtype)
    if qwen_dtype is not None:
        qwen_model_kwargs["dtype"] = qwen_dtype

    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_name,
        **qwen_model_kwargs,
    )
    qwen_model.eval()
    qwen_model.generation_config.pad_token_id = qwen_tokenizer.pad_token_id

    structured_qwen = outlines.from_transformers(qwen_model, qwen_tokenizer)
    plan_output_model = build_plan_output_model(args.max_actions)

    return SharedRuntime(
        tool_by_name=tool_by_name,
        registry_tool_names=sorted(tool_by_name),
        qwen_model_name=args.qwen_model_name,
        qwen_device=qwen_device,
        qwen_dtype=args.qwen_dtype,
        qwen_tokenizer=qwen_tokenizer,
        structured_qwen=structured_qwen,
        plan_output_model=plan_output_model,
        planner_generator=outlines.Generator(structured_qwen, plan_output_model),
        planner_max_new_tokens=args.planner_max_new_tokens,
        dispatcher_max_new_tokens=args.dispatcher_max_new_tokens,
        max_actions=args.max_actions,
        top_k=args.top_k,
        embedding_batch_size=args.embedding_batch_size,
    )


def build_variant_runtime(
    shared_runtime: SharedRuntime,
    variant: VariantSpec,
    embed_device: str,
) -> AgentRuntime:
    embedding_bundle = load_checkpoint_bundle(variant.checkpoint_path, device=embed_device)
    tool_centroids = F.normalize(embedding_bundle["centroids"].to(embed_device), dim=-1)

    return AgentRuntime(
        embed_model=embedding_bundle["model"],
        embed_tokenizer=embedding_bundle["tokenizer"],
        embed_max_length=int(embedding_bundle["max_length"]),
        embed_device=embed_device,
        tool_names=list(embedding_bundle["tool_names"]),
        tool_centroids=tool_centroids,
        checkpoint_architecture=embedding_bundle.get("architecture"),
        checkpoint_loss_name=embedding_bundle.get("loss_name"),
        encoder_model=getattr(embedding_bundle["model"], "encoder_model", None),
        tool_by_name=shared_runtime.tool_by_name,
        qwen_model_name=shared_runtime.qwen_model_name,
        qwen_device=shared_runtime.qwen_device,
        qwen_dtype=shared_runtime.qwen_dtype,
        qwen_tokenizer=shared_runtime.qwen_tokenizer,
        structured_qwen=shared_runtime.structured_qwen,
        plan_output_model=shared_runtime.plan_output_model,
        planner_generator=shared_runtime.planner_generator,
        planner_max_new_tokens=shared_runtime.planner_max_new_tokens,
        dispatcher_max_new_tokens=shared_runtime.dispatcher_max_new_tokens,
        max_actions=shared_runtime.max_actions,
        top_k=shared_runtime.top_k,
        embedding_batch_size=shared_runtime.embedding_batch_size,
        dispatch_generator_cache=shared_runtime.dispatch_generator_cache,
    )


def read_training_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        return []

    payload = load_json(metrics_path)
    if not isinstance(payload, list):
        return []

    return [row for row in payload if isinstance(row, dict)]


def summarize_training_history(
    history: list[dict[str, Any]],
    architecture: str,
) -> dict[str, Any] | None:
    if not history:
        return None

    last_row = history[-1]
    summary: dict[str, Any] = {
        "epochs": len(history),
        "best_epoch": (
            int(last_row["best_epoch"])
            if isinstance(last_row.get("best_epoch"), int)
            else None
        ),
    }

    if architecture == "hierarchical":
        summary.update(
            {
                "best_val_tool_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_tool_retrieval_accuracy",
                        "val_tool_retrieval_accuracy",
                    )
                ),
                "final_val_tool_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_tool_retrieval_accuracy")
                ),
                "best_val_parent_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_parent_retrieval_accuracy",
                        "val_parent_retrieval_accuracy",
                    )
                ),
                "final_val_parent_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_parent_retrieval_accuracy")
                ),
            }
        )
    else:
        summary.update(
            {
                "best_val_retrieval_accuracy": round_float(
                    max_numeric(
                        history,
                        "best_val_retrieval_accuracy",
                        "val_retrieval_accuracy",
                    )
                ),
                "final_val_retrieval_accuracy": round_float(
                    first_numeric(last_row, "val_retrieval_accuracy")
                ),
                "final_val_classification_accuracy": round_float(
                    first_numeric(last_row, "val_classification_accuracy")
                ),
            }
        )

    return summary


def variant_spec_from_checkpoint_path(
    checkpoint_path: Path,
    checkpoint_root: Path | None = None,
) -> VariantSpec:
    architecture = "legacy"
    loss_name = "single_checkpoint"

    if checkpoint_root is not None:
        try:
            relative_parts = checkpoint_path.resolve().relative_to(checkpoint_root.resolve()).parts
        except ValueError:
            relative_parts = ()
        if len(relative_parts) >= 3:
            architecture = relative_parts[0]
            loss_name = relative_parts[1]
    elif checkpoint_path.parent.name not in {"output", "eval"}:
        architecture_candidate = checkpoint_path.parent.parent.name
        loss_candidate = checkpoint_path.parent.name
        if architecture_candidate:
            architecture = architecture_candidate
            loss_name = loss_candidate

    variant_id = f"{architecture}/{loss_name}"
    return VariantSpec(
        variant_id=variant_id,
        architecture=architecture,
        loss_name=loss_name,
        checkpoint_path=checkpoint_path,
        metrics_path=checkpoint_path.parent / "metrics.json",
    )


def discover_variants(
    checkpoint_root: Path,
    checkpoint_filename: str,
) -> list[VariantSpec]:
    variants = [
        variant_spec_from_checkpoint_path(checkpoint_path, checkpoint_root=checkpoint_root)
        for checkpoint_path in sorted(checkpoint_root.glob(f"*/*/{checkpoint_filename}"))
        if checkpoint_path.is_file()
    ]
    if variants:
        return variants

    legacy_checkpoint_path = checkpoint_root / checkpoint_filename
    if legacy_checkpoint_path.is_file():
        return [variant_spec_from_checkpoint_path(legacy_checkpoint_path, checkpoint_root=checkpoint_root)]

    return []


def get_planner_generator(runtime: AgentRuntime):
    return runtime.planner_generator


def get_dispatch_generator(
    runtime: AgentRuntime,
    tool_name: str,
    schema: dict[str, Any],
) -> tuple[type[BaseModel], Any]:
    cache_key = f"{tool_name}:{json.dumps(schema, sort_keys=True)}"
    cached = runtime.dispatch_generator_cache.get(cache_key)
    if cached is None:
        output_model = build_dispatch_output_model(tool_name, schema)
        cached = (output_model, outlines.Generator(runtime.structured_qwen, output_model))
        runtime.dispatch_generator_cache[cache_key] = cached
    return cached


def retrieve_tools(runtime: AgentRuntime, actions: list[str]) -> list[dict[str, Any]]:
    if not actions:
        return []

    action_embeddings = embed_texts(
        model=runtime.embed_model,
        tokenizer=runtime.embed_tokenizer,
        texts=actions,
        device=runtime.embed_device,
        max_length=runtime.embed_max_length,
        batch_size=min(len(actions), runtime.embedding_batch_size),
        progress_desc="Embedding plan actions",
    )
    action_embeddings = F.normalize(action_embeddings.to(runtime.tool_centroids.device), dim=-1)
    score_matrix = action_embeddings @ runtime.tool_centroids.T

    rows: list[dict[str, Any]] = []
    k = min(runtime.top_k, len(runtime.tool_names))
    for action, scores in zip(actions, score_matrix, strict=True):
        top_scores, top_indices = torch.topk(scores, k=k)
        candidates = [
            {
                "tool": runtime.tool_names[index],
                "score": round_float(score),
            }
            for score, index in zip(top_scores.tolist(), top_indices.tolist(), strict=True)
        ]
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        ranked_tools = [runtime.tool_names[index] for index in ranked_indices]
        rows.append(
            {
                "action": action,
                "candidates": candidates,
                "ranked_tools": ranked_tools,
            }
        )
    return rows


def evaluate_dispatch_prediction(
    predicted_tools: list[str],
    expected_tools: list[str],
    ordering: list[str] | None,
    expected_calls: int | None,
) -> dict[str, Any]:
    if not expected_tools:
        return {
            "last_dispatch_hit": None,
            "first_dispatch_hit": None,
            "any_dispatch_hit": None,
            "all_expected_present": None,
            "expected_tool_recall": None,
            "expected_tool_precision": None,
            "expected_order_satisfied": None,
            "expected_call_count_satisfied": None,
            "exact_unique_tool_set_match": None,
            "exact_unique_tool_sequence_match": None,
        }

    expected_unique = unique_in_order(expected_tools)
    predicted_unique = unique_in_order(predicted_tools)
    expected_set = set(expected_unique)
    predicted_set = set(predicted_unique)

    matched_expected = [tool for tool in expected_unique if tool in predicted_set]
    recall = len(matched_expected) / len(expected_unique) if expected_unique else None

    if predicted_unique:
        precision = len([tool for tool in predicted_unique if tool in expected_set]) / len(predicted_unique)
    else:
        precision = 0.0

    last_dispatch_hit = bool(predicted_tools) and predicted_tools[-1] in expected_set
    first_dispatch_hit = bool(predicted_tools) and predicted_tools[0] in expected_set
    any_dispatch_hit = any(tool in expected_set for tool in predicted_tools)
    all_expected_present = set(expected_unique).issubset(predicted_set)

    if ordering:
        expected_order_satisfied = is_subsequence(predicted_tools, ordering)
    else:
        expected_order_satisfied = None

    if expected_calls is not None:
        expected_call_count_satisfied = (
            len(predicted_tools) == expected_calls
            and all(tool in expected_set for tool in predicted_tools)
        )
    else:
        expected_call_count_satisfied = None

    return {
        "last_dispatch_hit": last_dispatch_hit,
        "first_dispatch_hit": first_dispatch_hit,
        "any_dispatch_hit": any_dispatch_hit,
        "all_expected_present": all_expected_present,
        "expected_tool_recall": round_float(recall),
        "expected_tool_precision": round_float(precision),
        "expected_order_satisfied": expected_order_satisfied,
        "expected_call_count_satisfied": expected_call_count_satisfied,
        "exact_unique_tool_set_match": predicted_set == expected_set,
        "exact_unique_tool_sequence_match": predicted_unique == expected_unique,
    }


def run_scenario(runtime: AgentRuntime, scenario: dict[str, Any]) -> dict[str, Any]:
    user_request = str(scenario.get("input", ""))
    planner_generator = get_planner_generator(runtime)

    raw_plan_json: str | None = None
    plan_actions: list[str] = []
    plan_block: str | None = None
    retrieval_rows: list[dict[str, Any]] = []
    mapped_steps: list[dict[str, Any]] = []
    dispatches: list[dict[str, Any]] = []
    warnings: list[str] = []
    status = "ok"
    error_stage: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    try:
        planner_prompt = build_planner_prompt(user_request, runtime.max_actions)
        raw_plan_json = planner_generator(
            planner_prompt,
            **build_generation_kwargs(
                runtime.qwen_tokenizer,
                runtime.planner_max_new_tokens,
            ),
        )
        plan_actions, planner_warning = parse_plan_actions(
            runtime=runtime,
            user_request=user_request,
            raw_plan_json=raw_plan_json,
        )
        if planner_warning is not None:
            warnings.append(planner_warning)
        plan_block = build_plan_block(plan_actions)
    except Exception as exc:
        status = "error"
        error_stage = "planner"
        error_message = str(exc)
        error_traceback = traceback.format_exc(limit=5)

    if status == "ok":
        try:
            retrieval_rows = retrieve_tools(runtime, plan_actions)
            for row in retrieval_rows:
                candidates = row.get("candidates", [])
                if not candidates:
                    warnings.append(f"No tool candidates returned for action: {row['action']}")
                    continue
                selected_tool_name = str(candidates[0]["tool"])
                tool_spec = runtime.tool_by_name.get(selected_tool_name)
                if tool_spec is None:
                    raise KeyError(f"Selected tool {selected_tool_name!r} is missing from tools.json.")
                mapped_steps.append(
                    {
                        "action": row["action"],
                        "tool_name": selected_tool_name,
                        "schema": tool_spec["parameters"],
                        "candidates": candidates,
                    }
                )
        except Exception as exc:
            status = "error"
            error_stage = "retrieval"
            error_message = str(exc)
            error_traceback = traceback.format_exc(limit=5)

    if status == "ok":
        for step_index, step in enumerate(mapped_steps, start=1):
            try:
                dispatcher_prompt = build_dispatcher_prompt(
                    user_request=user_request,
                    action=step["action"],
                    tool_name=step["tool_name"],
                    schema=step["schema"],
                )
                output_model, dispatch_generator = get_dispatch_generator(
                    runtime,
                    step["tool_name"],
                    step["schema"],
                )
                raw_dispatch_json = dispatch_generator(
                    dispatcher_prompt,
                    **build_generation_kwargs(
                        runtime.qwen_tokenizer,
                        runtime.dispatcher_max_new_tokens,
                    ),
                )
                dispatch_payload = output_model.model_validate_json(raw_dispatch_json)
                dispatch_arguments = normalize_dispatch_arguments(
                    dispatch_payload.model_dump(exclude_none=True)
                )
                dispatch_block = build_dispatch_block(step["tool_name"], dispatch_arguments)
                dispatches.append(
                    {
                        "step_index": step_index,
                        "action": step["action"],
                        "tool_name": step["tool_name"],
                        "schema": step["schema"],
                        "raw_dispatch_json": raw_dispatch_json,
                        "arguments": dispatch_arguments,
                        "dispatch_block": dispatch_block,
                    }
                )
            except Exception as exc:
                status = "error"
                error_stage = f"dispatcher_step_{step_index}"
                error_message = str(exc)
                error_traceback = traceback.format_exc(limit=5)
                break

    predicted_tools = [dispatch["tool_name"] for dispatch in dispatches]
    dispatch_blocks = [dispatch["dispatch_block"] for dispatch in dispatches]
    last_dispatch_block = dispatch_blocks[-1] if dispatch_blocks else None
    last_dispatch_tool = predicted_tools[-1] if predicted_tools else None

    expected_tools = [str(tool) for tool in scenario.get("expected_tools", [])]
    ordering = scenario.get("ordering")
    ordering_list = [str(tool) for tool in ordering] if isinstance(ordering, list) else None
    expected_calls = scenario.get("expected_calls")
    evaluation = evaluate_dispatch_prediction(
        predicted_tools=predicted_tools,
        expected_tools=expected_tools,
        ordering=ordering_list,
        expected_calls=expected_calls if isinstance(expected_calls, int) else None,
    )

    if expected_tools and retrieval_rows:
        first_action_ranked_tools = retrieval_rows[0].get("ranked_tools", [])
        action_level_mrr = reciprocal_rank(first_action_ranked_tools, expected_tools)
    else:
        action_level_mrr = None

    return {
        "id": scenario.get("id"),
        "category": scenario.get("category"),
        "title": scenario.get("title"),
        "input": user_request,
        "expected_tools": expected_tools,
        "expected_behavior": scenario.get("expected_behavior"),
        "ordering": ordering_list,
        "expected_calls": expected_calls,
        "constraints": scenario.get("constraints"),
        "rules": scenario.get("rules"),
        "conditional": bool(scenario.get("conditional", False)),
        "status": status,
        "warnings": warnings,
        "error_stage": error_stage,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "raw_plan_json": raw_plan_json,
        "plan_actions": plan_actions,
        "plan_block": plan_block,
        "retrieval_rows": retrieval_rows,
        "mapped_steps": mapped_steps,
        "dispatches": dispatches,
        "dispatch_blocks": dispatch_blocks,
        "last_dispatch_block": last_dispatch_block,
        "predicted_tools": predicted_tools,
        "predicted_unique_tools": unique_in_order(predicted_tools),
        "last_dispatch_tool": last_dispatch_tool,
        "action_level_first_step_mrr": round_float(action_level_mrr),
        "evaluation": evaluation,
    }


def build_aggregate_metrics(scenario_reports: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in scenario_reports if row.get("expected_tools")]
    single_tool = [row for row in evaluable if len(row["expected_tools"]) == 1]
    multi_tool = [row for row in evaluable if len(row["expected_tools"]) > 1]
    order_cases = [row for row in evaluable if row.get("ordering")]
    call_count_cases = [row for row in evaluable if row.get("expected_calls") is not None]

    def collect_bool(rows: list[dict[str, Any]], field_name: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row["evaluation"].get(field_name)
            if value is not None:
                values.append(1.0 if value else 0.0)
        return values

    def collect_number(rows: list[dict[str, Any]], field_name: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row["evaluation"].get(field_name)
            if value is not None:
                values.append(float(value))
        return values

    return {
        "overall": {
            "last_dispatch_hit_rate": mean_or_none(collect_bool(evaluable, "last_dispatch_hit")),
            "first_dispatch_hit_rate": mean_or_none(collect_bool(evaluable, "first_dispatch_hit")),
            "any_dispatch_hit_rate": mean_or_none(collect_bool(evaluable, "any_dispatch_hit")),
            "all_expected_present_rate": mean_or_none(collect_bool(evaluable, "all_expected_present")),
            "exact_unique_tool_set_match_rate": mean_or_none(
                collect_bool(evaluable, "exact_unique_tool_set_match")
            ),
            "exact_unique_tool_sequence_match_rate": mean_or_none(
                collect_bool(evaluable, "exact_unique_tool_sequence_match")
            ),
            "mean_expected_tool_recall": mean_or_none(
                collect_number(evaluable, "expected_tool_recall")
            ),
            "mean_expected_tool_precision": mean_or_none(
                collect_number(evaluable, "expected_tool_precision")
            ),
        },
        "single_tool": {
            "last_dispatch_hit_rate": mean_or_none(collect_bool(single_tool, "last_dispatch_hit")),
            "mean_expected_tool_precision": mean_or_none(
                collect_number(single_tool, "expected_tool_precision")
            ),
        },
        "multi_tool": {
            "all_expected_present_rate": mean_or_none(collect_bool(multi_tool, "all_expected_present")),
            "mean_expected_tool_recall": mean_or_none(
                collect_number(multi_tool, "expected_tool_recall")
            ),
            "mean_expected_tool_precision": mean_or_none(
                collect_number(multi_tool, "expected_tool_precision")
            ),
            "exact_unique_tool_sequence_match_rate": mean_or_none(
                collect_bool(multi_tool, "exact_unique_tool_sequence_match")
            ),
        },
        "ordering": {
            "expected_order_satisfied_rate": mean_or_none(
                collect_bool(order_cases, "expected_order_satisfied")
            ),
        },
        "call_counts": {
            "expected_call_count_satisfied_rate": mean_or_none(
                collect_bool(call_count_cases, "expected_call_count_satisfied")
            ),
        },
    }


def evaluate_variant(
    variant: VariantSpec,
    shared_runtime: SharedRuntime,
    scenarios: list[dict[str, Any]],
    benchmark_tool_names: set[str],
    embed_device: str,
) -> dict[str, Any]:
    history = read_training_history(variant.metrics_path)
    training_summary = summarize_training_history(history, variant.architecture)

    warnings: list[str] = []
    try:
        runtime = build_variant_runtime(
            shared_runtime=shared_runtime,
            variant=variant,
            embed_device=embed_device,
        )
        resolved_architecture = runtime.checkpoint_architecture or variant.architecture
        resolved_loss_name = runtime.checkpoint_loss_name or variant.loss_name
        resolved_variant_id = f"{resolved_architecture}/{resolved_loss_name}"
        training_summary = summarize_training_history(history, resolved_architecture)
        tool_name_set = set(runtime.tool_names)
        missing_benchmark_tools = sorted(benchmark_tool_names - tool_name_set)
        if missing_benchmark_tools:
            warnings.append(
                "Benchmark tools missing from checkpoint centroids: "
                + ", ".join(missing_benchmark_tools)
            )

        tools_without_registry_entry = sorted(tool_name_set - set(shared_runtime.tool_by_name))
        if tools_without_registry_entry:
            warnings.append(
                "Checkpoint tools missing from tools.json: "
                + ", ".join(tools_without_registry_entry)
            )

        scenario_reports = [run_scenario(runtime, scenario) for scenario in scenarios]
        aggregate_metrics = build_aggregate_metrics(scenario_reports)

        success_count = sum(1 for row in scenario_reports if row["status"] == "ok")
        error_count = len(scenario_reports) - success_count

        return {
            "variant_id": resolved_variant_id,
            "architecture": resolved_architecture,
            "loss_name": resolved_loss_name,
            "status": "ok",
            "warnings": warnings,
            "error_message": None,
            "error_traceback": None,
            "paths": {
                "checkpoint_path": str(variant.checkpoint_path.resolve()),
                "metrics_path": str(variant.metrics_path.resolve()),
            },
            "runtime": {
                "qwen_model_name": runtime.qwen_model_name,
                "qwen_device": runtime.qwen_device,
                "qwen_dtype": runtime.qwen_dtype,
                "embed_device": runtime.embed_device,
                "top_k": runtime.top_k,
                "max_actions": runtime.max_actions,
                "planner_max_new_tokens": runtime.planner_max_new_tokens,
                "dispatcher_max_new_tokens": runtime.dispatcher_max_new_tokens,
                "embedding_batch_size": runtime.embedding_batch_size,
                "tool_count": len(runtime.tool_names),
                "embed_max_length": runtime.embed_max_length,
                "encoder_model": runtime.encoder_model,
                "checkpoint_architecture": runtime.checkpoint_architecture,
                "checkpoint_loss_name": runtime.checkpoint_loss_name,
            },
            "training": training_summary,
            "counts": {
                "total_scenarios": len(scenario_reports),
                "successful_scenarios": success_count,
                "error_scenarios": error_count,
                "evaluable_scenarios": sum(
                    1 for row in scenario_reports if row.get("expected_tools")
                ),
                "behavior_only_scenarios": sum(
                    1 for row in scenario_reports if row.get("expected_behavior") is not None
                ),
                "no_tool_scenarios": sum(
                    1
                    for row in scenario_reports
                    if "expected_tools" in row and not row.get("expected_tools")
                ),
                "single_tool_scenarios": sum(
                    1 for row in scenario_reports if len(row.get("expected_tools", [])) == 1
                ),
                "multi_tool_scenarios": sum(
                    1 for row in scenario_reports if len(row.get("expected_tools", [])) > 1
                ),
            },
            "metrics": aggregate_metrics,
            "tool_names": runtime.tool_names,
            "scenarios": scenario_reports,
        }
    except Exception as exc:
        return {
            "variant_id": variant.variant_id,
            "architecture": variant.architecture,
            "loss_name": variant.loss_name,
            "status": "error",
            "warnings": warnings,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=5),
            "paths": {
                "checkpoint_path": str(variant.checkpoint_path.resolve()),
                "metrics_path": str(variant.metrics_path.resolve()),
            },
            "runtime": {
                "qwen_model_name": shared_runtime.qwen_model_name,
                "qwen_device": shared_runtime.qwen_device,
                "qwen_dtype": shared_runtime.qwen_dtype,
                "embed_device": embed_device,
                "top_k": shared_runtime.top_k,
                "max_actions": shared_runtime.max_actions,
                "planner_max_new_tokens": shared_runtime.planner_max_new_tokens,
                "dispatcher_max_new_tokens": shared_runtime.dispatcher_max_new_tokens,
                "embedding_batch_size": shared_runtime.embedding_batch_size,
            },
            "training": training_summary,
            "counts": {
                "total_scenarios": len(scenarios),
                "successful_scenarios": 0,
                "error_scenarios": len(scenarios),
                "evaluable_scenarios": sum(
                    1 for scenario in scenarios if scenario.get("expected_tools")
                ),
                "behavior_only_scenarios": sum(
                    1 for scenario in scenarios if scenario.get("expected_behavior") is not None
                ),
                "no_tool_scenarios": sum(
                    1
                    for scenario in scenarios
                    if "expected_tools" in scenario and not scenario.get("expected_tools")
                ),
                "single_tool_scenarios": sum(
                    1 for scenario in scenarios if len(scenario.get("expected_tools", [])) == 1
                ),
                "multi_tool_scenarios": sum(
                    1 for scenario in scenarios if len(scenario.get("expected_tools", [])) > 1
                ),
            },
            "metrics": None,
            "tool_names": [],
            "scenarios": [],
        }


def build_leaderboard(variant_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for variant in variant_summaries:
        if variant.get("status") != "ok":
            continue

        overall = variant["metrics"]["overall"]
        training = variant.get("training") or {}
        entries.append(
            {
                "variant_id": variant["variant_id"],
                "architecture": variant["architecture"],
                "loss_name": variant["loss_name"],
                "last_dispatch_hit_rate": overall["last_dispatch_hit_rate"],
                "any_dispatch_hit_rate": overall["any_dispatch_hit_rate"],
                "all_expected_present_rate": overall["all_expected_present_rate"],
                "exact_unique_tool_set_match_rate": overall["exact_unique_tool_set_match_rate"],
                "exact_unique_tool_sequence_match_rate": overall["exact_unique_tool_sequence_match_rate"],
                "mean_expected_tool_recall": overall["mean_expected_tool_recall"],
                "mean_expected_tool_precision": overall["mean_expected_tool_precision"],
                "best_epoch": training.get("best_epoch"),
                "best_val_retrieval_accuracy": training.get(
                    "best_val_retrieval_accuracy",
                    training.get("best_val_tool_retrieval_accuracy"),
                ),
            }
        )

    entries.sort(
        key=lambda row: (
            -safe_metric_sort_value(row["exact_unique_tool_sequence_match_rate"]),
            -safe_metric_sort_value(row["all_expected_present_rate"]),
            -safe_metric_sort_value(row["any_dispatch_hit_rate"]),
            -safe_metric_sort_value(row["mean_expected_tool_recall"]),
            row["variant_id"],
        )
    )

    for index, row in enumerate(entries, start=1):
        row["rank"] = index
    return entries


def build_cross_variant_metrics(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    if not leaderboard:
        return {
            "best_variant_by_exact_sequence_match": None,
            "best_exact_unique_tool_sequence_match_rate": None,
            "best_variant_by_all_expected_present": None,
            "best_all_expected_present_rate": None,
            "best_variant_by_last_dispatch_hit": None,
            "best_last_dispatch_hit_rate": None,
        }

    best_by_exact_sequence = max(
        leaderboard,
        key=lambda row: (
            safe_metric_sort_value(row["exact_unique_tool_sequence_match_rate"]),
            safe_metric_sort_value(row["all_expected_present_rate"]),
            safe_metric_sort_value(row["any_dispatch_hit_rate"]),
        ),
    )
    best_by_all_present = max(
        leaderboard,
        key=lambda row: (
            safe_metric_sort_value(row["all_expected_present_rate"]),
            safe_metric_sort_value(row["exact_unique_tool_sequence_match_rate"]),
            safe_metric_sort_value(row["mean_expected_tool_recall"]),
        ),
    )
    best_by_last_hit = max(
        leaderboard,
        key=lambda row: (
            safe_metric_sort_value(row["last_dispatch_hit_rate"]),
            safe_metric_sort_value(row["any_dispatch_hit_rate"]),
            safe_metric_sort_value(row["exact_unique_tool_sequence_match_rate"]),
        ),
    )

    return {
        "best_variant_by_exact_sequence_match": best_by_exact_sequence["variant_id"],
        "best_exact_unique_tool_sequence_match_rate": best_by_exact_sequence[
            "exact_unique_tool_sequence_match_rate"
        ],
        "best_variant_by_all_expected_present": best_by_all_present["variant_id"],
        "best_all_expected_present_rate": best_by_all_present["all_expected_present_rate"],
        "best_variant_by_last_dispatch_hit": best_by_last_hit["variant_id"],
        "best_last_dispatch_hit_rate": best_by_last_hit["last_dispatch_hit_rate"],
    }


def evaluate_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_payload = load_json(args.benchmark_path)
    scenarios = benchmark_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError(f"Expected {args.benchmark_path} to contain a 'scenarios' list.")

    benchmark_tool_names = {
        str(tool_name)
        for scenario in scenarios
        for tool_name in scenario.get("expected_tools", [])
    }

    shared_runtime = build_shared_runtime(args)

    if args.checkpoint_path is not None:
        if not args.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        variants = [variant_spec_from_checkpoint_path(args.checkpoint_path)]
    else:
        variants = discover_variants(args.checkpoint_root, args.checkpoint_filename)
        if not variants:
            raise FileNotFoundError(
                f"No checkpoint variants matching */*/{args.checkpoint_filename} or "
                f"{args.checkpoint_filename} found under {args.checkpoint_root}."
            )

    _, resolved_embed_device = resolve_runtime_devices(args.qwen_device, args.embed_device)
    variant_summaries = [
        evaluate_variant(
            variant=variant,
            shared_runtime=shared_runtime,
            scenarios=scenarios,
            benchmark_tool_names=benchmark_tool_names,
            embed_device=resolved_embed_device,
        )
        for variant in variants
    ]
    leaderboard = build_leaderboard(variant_summaries)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ToolCall15",
        "mode": "embedding_space_variant_sweep",
        "description": (
            "Evaluates every discovered ToolCall15 embedding-space variant with a shared "
            "Qwen planner/dispatcher pipeline: Qwen planner -> embedding tool retrieval "
            "per action -> Qwen dispatcher. Metrics score the final dispatch outputs "
            "against the benchmark for each variant."
        ),
        "paths": {
            "checkpoint_path": (
                str(args.checkpoint_path.resolve())
                if args.checkpoint_path is not None
                else None
            ),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "benchmark_path": str(args.benchmark_path.resolve()),
            "tools_path": str(args.tools_path.resolve()),
            "output_path": str(args.output_path.resolve()),
        },
        "runtime": {
            "qwen_model_name": shared_runtime.qwen_model_name,
            "qwen_device": shared_runtime.qwen_device,
            "qwen_dtype": shared_runtime.qwen_dtype,
            "embed_device": resolved_embed_device,
            "top_k": shared_runtime.top_k,
            "max_actions": shared_runtime.max_actions,
            "planner_max_new_tokens": shared_runtime.planner_max_new_tokens,
            "dispatcher_max_new_tokens": shared_runtime.dispatcher_max_new_tokens,
            "embedding_batch_size": shared_runtime.embedding_batch_size,
            "checkpoint_filename": args.checkpoint_filename,
            "variant_count": len(variants),
            "benchmark_tool_count": len(benchmark_tool_names),
            "local_files_only": bool(args.local_files_only),
        },
        "counts": {
            "total_variants": len(variant_summaries),
            "successful_variants": sum(
                1 for variant in variant_summaries if variant.get("status") == "ok"
            ),
            "error_variants": sum(
                1 for variant in variant_summaries if variant.get("status") != "ok"
            ),
            "total_scenarios": len(scenarios),
            "evaluable_scenarios": sum(
                1 for scenario in scenarios if scenario.get("expected_tools")
            ),
            "behavior_only_scenarios": sum(
                1 for scenario in scenarios if scenario.get("expected_behavior") is not None
            ),
            "no_tool_scenarios": sum(
                1 for scenario in scenarios if "expected_tools" in scenario and not scenario.get("expected_tools")
            ),
            "single_tool_scenarios": sum(
                1 for scenario in scenarios if len(scenario.get("expected_tools", [])) == 1
            ),
            "multi_tool_scenarios": sum(
                1 for scenario in scenarios if len(scenario.get("expected_tools", [])) > 1
            ),
        },
        "metrics": build_cross_variant_metrics(leaderboard),
        "leaderboard": leaderboard,
        "tool_names": shared_runtime.registry_tool_names,
        "variants": variant_summaries,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_benchmark(args)
    best_variant = summary["metrics"]["best_variant_by_exact_sequence_match"]
    best_sequence = summary["metrics"]["best_exact_unique_tool_sequence_match_rate"]
    best_all_present = summary["metrics"]["best_all_expected_present_rate"]
    print(f"Wrote eval summary to {args.output_path}")
    print(
        "Best variant: "
        f"{best_variant}, "
        f"exact_sequence={best_sequence}, "
        f"all_present={best_all_present}"
    )


if __name__ == "__main__":
    main()
