from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from training.dataset_utils import load_dataset_rows, write_json, write_jsonl


DEFAULT_TOOL_SELECTION_SYSTEM_PROMPT = (
    "You are a tool selection model. Choose the single best tool for the user request "
    "from the provided catalog. Return only valid JSON."
)
DEFAULT_RANKING_LIMIT = 5
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class EmbeddingVariantSpec:
    variant_id: str
    architecture: str
    loss_name: str
    checkpoint_path: Path


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "model"


def round_float(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round_float(sum(clean) / len(clean))


def percentile_or_none(values: Iterable[float | None], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round_float(clean[0])
    rank = (len(clean) - 1) * percentile
    low_index = math.floor(rank)
    high_index = math.ceil(rank)
    if low_index == high_index:
        return round_float(clean[low_index])
    blend = rank - low_index
    return round_float(clean[low_index] * (1.0 - blend) + clean[high_index] * blend)


def load_tool_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tools = payload.get("tools", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_tools, list):
        raise ValueError(f"Expected a list of tools in {path}")
    return [tool for tool in raw_tools if isinstance(tool, dict)]


def load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_dataset_rows(path)
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        tool = str(row.get("tool", "")).strip()
        query = str(row.get("query", row.get("text", ""))).strip()
        if tool and query:
            normalized = dict(row)
            normalized["tool"] = tool
            normalized["query"] = query
            cleaned_rows.append(normalized)
    if not cleaned_rows:
        raise ValueError(f"No valid benchmark rows found in {path}")
    return cleaned_rows


def summarize_parameters(parameters: dict[str, Any]) -> str:
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = set(parameters.get("required", [])) if isinstance(parameters, dict) else set()
    if not isinstance(properties, dict) or not properties:
        return "no arguments"

    parts: list[str] = []
    for name in sorted(properties):
        spec = properties.get(name, {})
        if not isinstance(spec, dict):
            spec = {}
        field_type = str(spec.get("type", "any")).strip() or "any"
        marker = "*" if name in required else ""
        parts.append(f"{name}{marker}:{field_type}")
    return ", ".join(parts)


def render_tool_catalog(
    tools: Sequence[dict[str, Any]],
    *,
    limit_tools: Sequence[str] | None = None,
) -> str:
    tool_name_filter = set(limit_tools) if limit_tools is not None else None
    lines: list[str] = []
    for tool in tools:
        tool_name = str(tool.get("name", "")).strip()
        if not tool_name:
            continue
        if tool_name_filter is not None and tool_name not in tool_name_filter:
            continue
        description = str(tool.get("description", "")).strip()
        parameters = tool.get("parameters", {})
        argument_summary = summarize_parameters(parameters if isinstance(parameters, dict) else {})
        lines.append(
            f"- {tool_name}: {description}\n"
            f"  arguments: {argument_summary}"
        )
    return "\n".join(lines)


def build_selection_messages(
    query: str,
    tools: Sequence[dict[str, Any]],
    *,
    ranking_limit: int = DEFAULT_RANKING_LIMIT,
    system_prompt: str = DEFAULT_TOOL_SELECTION_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    catalog_block = render_tool_catalog(tools)
    user_prompt = (
        f"User request:\n{query}\n\n"
        f"Tool catalog:\n{catalog_block}\n\n"
        "Return JSON with exactly these fields:\n"
        '{\n'
        '  "selected_tool": "<tool_name>",\n'
        f'  "ranked_tools": ["<tool_name_1>", "<tool_name_2>", "... up to {ranking_limit} total"],\n'
        '  "reason": "<one short sentence>"\n'
        '}\n'
        "Rules:\n"
        "- selected_tool must be one of the listed tools.\n"
        "- ranked_tools must be unique, ordered best to worst, and contain selected_tool first.\n"
        "- Do not include any tools not in the catalog.\n"
        "- Do not include markdown fences or any text before or after the JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_PATTERN.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_ranked_tools(
    ranked_tools: Sequence[Any],
    *,
    valid_tool_names: Sequence[str],
    selected_tool: str | None,
    ranking_limit: int,
) -> list[str]:
    valid_names = set(valid_tool_names)
    normalized: list[str] = []
    if selected_tool and selected_tool in valid_names:
        normalized.append(selected_tool)
    for item in ranked_tools:
        tool_name = str(item).strip()
        if tool_name and tool_name in valid_names and tool_name not in normalized:
            normalized.append(tool_name)
        if len(normalized) >= ranking_limit:
            break
    return normalized


def parse_selection_response(
    raw_text: str,
    *,
    valid_tool_names: Sequence[str],
    ranking_limit: int = DEFAULT_RANKING_LIMIT,
) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    if payload is None:
        raise ValueError("Model output did not contain a valid JSON object.")

    selected_tool = str(
        payload.get("selected_tool", payload.get("tool", payload.get("prediction", "")))
    ).strip()
    if not selected_tool:
        ranked_candidate = payload.get("ranked_tools", payload.get("tools", []))
        if isinstance(ranked_candidate, list) and ranked_candidate:
            selected_tool = str(ranked_candidate[0]).strip()

    valid_names = set(valid_tool_names)
    if selected_tool not in valid_names:
        raise ValueError(f"Model selected invalid tool: {selected_tool!r}")

    ranked_candidate = payload.get("ranked_tools", payload.get("tools", []))
    ranked_tools = (
        ranked_candidate
        if isinstance(ranked_candidate, list)
        else [selected_tool]
    )
    normalized_ranked_tools = normalize_ranked_tools(
        ranked_tools,
        valid_tool_names=valid_tool_names,
        selected_tool=selected_tool,
        ranking_limit=ranking_limit,
    )
    if not normalized_ranked_tools:
        normalized_ranked_tools = [selected_tool]

    return {
        "selected_tool": selected_tool,
        "ranked_tools": normalized_ranked_tools,
        "reason": str(payload.get("reason", "")).strip(),
        "raw_payload": payload,
    }


def reciprocal_rank(ranked_tools: Sequence[str], expected_tool: str) -> float:
    for index, tool_name in enumerate(ranked_tools, start=1):
        if tool_name == expected_tool:
            return 1.0 / float(index)
    return 0.0


def safe_metric_sort_value(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def summarize_result_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    accuracy_values = [1.0 if row.get("correct_top1") else 0.0 for row in successful_rows]
    top_3_values = [1.0 if row.get("top_3_hit") else 0.0 for row in successful_rows]
    top_5_values = [1.0 if row.get("top_5_hit") else 0.0 for row in successful_rows]
    mrr_values = [row.get("reciprocal_rank") for row in successful_rows]
    latencies = [row.get("latency_ms") for row in successful_rows]
    input_tokens = [row.get("input_tokens") for row in successful_rows]
    output_tokens = [row.get("output_tokens") for row in successful_rows]
    total_tokens = [row.get("total_tokens") for row in successful_rows]
    costs = [row.get("cost_usd") for row in successful_rows]
    tool_totals: dict[str, int] = defaultdict(int)
    tool_correct: dict[str, int] = defaultdict(int)

    for row in successful_rows:
        expected_tool = str(row.get("expected_tool", "")).strip()
        if not expected_tool:
            continue
        tool_totals[expected_tool] += 1
        if row.get("correct_top1"):
            tool_correct[expected_tool] += 1

    return {
        "total_examples": len(rows),
        "successful_examples": len(successful_rows),
        "error_examples": len(rows) - len(successful_rows),
        "top_1_accuracy": mean_or_none(accuracy_values),
        "top_3_accuracy": mean_or_none(top_3_values),
        "top_5_accuracy": mean_or_none(top_5_values),
        "mean_reciprocal_rank": mean_or_none(mrr_values),
        "mean_latency_ms": mean_or_none(latencies),
        "p50_latency_ms": percentile_or_none(latencies, 0.50),
        "p95_latency_ms": percentile_or_none(latencies, 0.95),
        "mean_input_tokens": mean_or_none(input_tokens),
        "mean_output_tokens": mean_or_none(output_tokens),
        "mean_total_tokens": mean_or_none(total_tokens),
        "sum_input_tokens": round_float(sum(value for value in input_tokens if value is not None)),
        "sum_output_tokens": round_float(sum(value for value in output_tokens if value is not None)),
        "sum_total_tokens": round_float(sum(value for value in total_tokens if value is not None)),
        "mean_cost_usd": mean_or_none(costs),
        "sum_cost_usd": round_float(sum(value for value in costs if value is not None)),
        "per_tool_accuracy": {
            tool_name: round_float(tool_correct[tool_name] / total)
            for tool_name, total in sorted(tool_totals.items())
        },
    }


def estimate_cost_usd(
    pricing: dict[str, Any] | None,
    *,
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if pricing is None:
        return None
    model_pricing = pricing.get(model_name)
    if not isinstance(model_pricing, dict):
        return None
    input_rate = model_pricing.get("input_per_million_usd")
    output_rate = model_pricing.get("output_per_million_usd")
    if not isinstance(input_rate, (int, float)) or not isinstance(output_rate, (int, float)):
        return None
    if input_tokens is None or output_tokens is None:
        return None
    return round_float(
        (float(input_tokens) / 1_000_000.0) * float(input_rate)
        + (float(output_tokens) / 1_000_000.0) * float(output_rate)
    )


def discover_embedding_variants(
    checkpoint_root: Path,
    checkpoint_filename: str,
) -> list[EmbeddingVariantSpec]:
    candidate_paths: dict[Path, None] = {}
    for pattern in (
        checkpoint_filename,
        f"*/{checkpoint_filename}",
        f"*/*/{checkpoint_filename}",
    ):
        for checkpoint_path in checkpoint_root.glob(pattern):
            if checkpoint_path.is_file():
                candidate_paths[checkpoint_path.resolve()] = None

    variants: list[EmbeddingVariantSpec] = []
    for checkpoint_path in sorted(candidate_paths):
        architecture = checkpoint_path.parent.parent.name
        loss_name = checkpoint_path.parent.name
        variants.append(
            EmbeddingVariantSpec(
                variant_id=f"{architecture}/{loss_name}",
                architecture=architecture,
                loss_name=loss_name,
                checkpoint_path=checkpoint_path,
            )
        )
    return variants


def build_leaderboard(model_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [dict(summary) for summary in model_summaries if summary.get("status") == "ok"]
    entries.sort(
        key=lambda row: (
            -safe_metric_sort_value(row["metrics"].get("top_1_accuracy")),
            -safe_metric_sort_value(row["metrics"].get("mean_reciprocal_rank")),
            -safe_metric_sort_value(row["metrics"].get("top_3_accuracy")),
            row["adapter_id"],
        )
    )
    leaderboard: list[dict[str, Any]] = []
    for index, row in enumerate(entries, start=1):
        leaderboard.append(
            {
                "rank": index,
                "adapter_id": row["adapter_id"],
                "provider": row["provider"],
                "mode": row["mode"],
                "model_name": row["model_name"],
                "top_1_accuracy": row["metrics"].get("top_1_accuracy"),
                "top_3_accuracy": row["metrics"].get("top_3_accuracy"),
                "top_5_accuracy": row["metrics"].get("top_5_accuracy"),
                "mean_reciprocal_rank": row["metrics"].get("mean_reciprocal_rank"),
                "mean_latency_ms": row["metrics"].get("mean_latency_ms"),
                "mean_total_tokens": row["metrics"].get("mean_total_tokens"),
                "sum_cost_usd": row["metrics"].get("sum_cost_usd"),
            }
        )
    return leaderboard


def build_benchmark_summary(
    *,
    benchmark_name: str,
    dataset_path: Path,
    tools_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    model_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    leaderboard = build_leaderboard(model_summaries)
    return {
        "created_at": now_utc_iso(),
        "benchmark": benchmark_name,
        "paths": {
            "dataset_path": str(dataset_path.resolve()),
            "tools_path": str(tools_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "config": config,
        "counts": {
            "model_count": len(model_summaries),
            "successful_models": sum(1 for summary in model_summaries if summary.get("status") == "ok"),
            "error_models": sum(1 for summary in model_summaries if summary.get("status") != "ok"),
        },
        "leaderboard": leaderboard,
        "models": list(model_summaries),
    }


__all__ = [
    "DEFAULT_RANKING_LIMIT",
    "EmbeddingVariantSpec",
    "build_benchmark_summary",
    "build_leaderboard",
    "build_selection_messages",
    "discover_embedding_variants",
    "estimate_cost_usd",
    "load_benchmark_rows",
    "load_tool_catalog",
    "now_utc_iso",
    "parse_selection_response",
    "reciprocal_rank",
    "render_tool_catalog",
    "round_float",
    "slugify",
    "summarize_result_rows",
    "write_json",
    "write_jsonl",
]
