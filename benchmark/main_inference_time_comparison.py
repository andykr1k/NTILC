from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters import (
    AnthropicSelectionAdapter,
    EmbeddingSelectionAdapter,
    GeminiSelectionAdapter,
    LocalHFSelectionAdapter,
    OpenAISelectionAdapter,
    build_model_summary,
)
from benchmark.common import (
    DEFAULT_RANKING_LIMIT,
    EmbeddingVariantSpec,
    build_benchmark_summary,
    load_benchmark_rows,
    load_tool_catalog,
    now_utc_iso,
    render_tool_catalog,
    round_float,
    slugify,
    write_json,
    write_jsonl,
)
from benchmark.env import load_env_file


DEFAULT_DATASETS = ("ToolEyes", "MetaTool", "API-Bank", "BFCL", "ToolBench")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "output"
DEFAULT_RUN_NAME = "main-inference-time-comparison"
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_NTILC_CHECKPOINT_GLOB = "output/normal/functional_margin/**/best.pt"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_OPENAI_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
API_METHOD_KEYS = frozenset({"openai_ict", "gemini_ict", "anthropic_ict"})
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FAILURE_ROW_FIELDS = [
    "dataset",
    "method_key",
    "method",
    "provider",
    "model",
    "example_id",
    "query",
    "expected_tool",
    "selected_tool",
    "ranked_tools",
    "status",
    "failure_type",
    "error_message",
    "reason",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]
FAILURE_MATRIX_BASE_FIELDS = ["dataset", "example_id", "query", "expected_tool"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_path: Path
    tools_path: Path
    checkpoint_path: Path | None


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    provider: str
    mode: str
    model_name: str


@dataclass
class MethodOutput:
    summary: dict[str, Any]
    results: list[dict[str, Any]]
    loaded_existing: bool = False
    dry_run: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper main inference-time comparison table on each dataset test split. "
            "Outputs per-example JSONL, aggregate JSON, CSV table rows, and a LaTeX table."
        )
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Cap examples per dataset. Use 0 to run the full test split.",
    )
    parser.add_argument("--ranking-limit", type=int, default=DEFAULT_RANKING_LIMIT)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["qwen_ict", "openai_ict", "gemini_ict", "anthropic_ict", "ntilc"],
        choices=["qwen_ict", "openai_ict", "gemini_ict", "anthropic_ict", "ntilc"],
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-label", default="Qwen3-27B (ICT)")
    parser.add_argument("--hf-device", default="cuda:7")
    parser.add_argument("--hf-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--hf-max-new-tokens", type=int, default=160)
    parser.add_argument("--hf-local-files-only", action="store_true")

    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--openai-label", default="ChatGPT 5 (ICT)")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--gemini-label", default="Gemini 2.5 Flash (ICT)")
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--anthropic-label", default="Claude Sonnet 4.6 (ICT)")
    parser.add_argument(
        "--api-max-output-tokens",
        type=int,
        default=0,
        help=(
            "Optional output token cap for API models. "
            "Use 0 to omit provider caps where supported."
        ),
    )
    parser.add_argument("--api-timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--api-parallel-workers",
        type=int,
        default=3,
        help="Maximum API providers evaluated concurrently within each dataset.",
    )

    parser.add_argument("--ntilc-label", default="NTILC")
    parser.add_argument(
        "--ntilc-checkpoint-glob",
        default=DEFAULT_NTILC_CHECKPOINT_GLOB,
        help=(
            "Dataset-relative glob used to find the NTILC checkpoint. "
            "The first sorted match is used for each dataset."
        ),
    )
    parser.add_argument("--embedding-device", default="cuda:7")

    parser.add_argument(
        "--registry-tokenizer-model",
        default=DEFAULT_QWEN_MODEL,
        help="HF tokenizer used to count registry prompt tokens. Falls back to regex counting if unavailable.",
    )
    parser.add_argument("--registry-tokenizer-local-files-only", action="store_true")
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def build_token_counter(
    tokenizer_model: str,
    *,
    local_files_only: bool,
) -> tuple[Callable[[str], int], dict[str, Any]]:
    if tokenizer_model:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )

            def count_with_tokenizer(text: str) -> int:
                encoded = tokenizer(str(text), add_special_tokens=False, verbose=False)
                return int(len(encoded["input_ids"]))

            return count_with_tokenizer, {
                "method": "transformers",
                "tokenizer_model": tokenizer_model,
                "local_files_only": local_files_only,
            }
        except Exception as exc:
            return (
                lambda text: len(TOKEN_PATTERN.findall(str(text))),
                {
                    "method": "regex_fallback",
                    "requested_tokenizer_model": tokenizer_model,
                    "local_files_only": local_files_only,
                    "error_message": str(exc),
                },
            )

    return (
        lambda text: len(TOKEN_PATTERN.findall(str(text))),
        {"method": "regex_fallback", "requested_tokenizer_model": ""},
    )


def resolve_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for dataset_name in args.datasets:
        dataset_dir = args.data_root / dataset_name
        dataset_path = dataset_dir / "tool_embedding_dataset_test.jsonl"
        tools_path = dataset_dir / "tools.json"
        checkpoint_matches = sorted(dataset_dir.glob(args.ntilc_checkpoint_glob))
        specs.append(
            DatasetSpec(
                name=dataset_name,
                dataset_path=dataset_path,
                tools_path=tools_path,
                checkpoint_path=checkpoint_matches[0] if checkpoint_matches else None,
            )
        )
    return specs


def build_method_specs(args: argparse.Namespace) -> dict[str, MethodSpec]:
    return {
        "qwen_ict": MethodSpec(
            key="qwen_ict",
            label=args.qwen_label,
            provider="huggingface",
            mode="llm_local",
            model_name=args.qwen_model,
        ),
        "openai_ict": MethodSpec(
            key="openai_ict",
            label=args.openai_label,
            provider="openai",
            mode="llm_api",
            model_name=args.openai_model,
        ),
        "gemini_ict": MethodSpec(
            key="gemini_ict",
            label=args.gemini_label,
            provider="gemini",
            mode="llm_api",
            model_name=args.gemini_model,
        ),
        "anthropic_ict": MethodSpec(
            key="anthropic_ict",
            label=args.anthropic_label,
            provider="anthropic",
            mode="llm_api",
            model_name=args.anthropic_model,
        ),
        "ntilc": MethodSpec(
            key="ntilc",
            label=args.ntilc_label,
            provider="embedding",
            mode="embedding",
            model_name="ntilc",
        ),
    }


def partition_methods(methods: Sequence[MethodSpec]) -> tuple[list[MethodSpec], list[MethodSpec]]:
    api_methods: list[MethodSpec] = []
    local_methods: list[MethodSpec] = []
    for method in methods:
        if method.key in API_METHOD_KEYS or method.mode == "llm_api":
            api_methods.append(method)
        else:
            local_methods.append(method)
    return api_methods, local_methods


def infer_embedding_variant(dataset_dir: Path, checkpoint_path: Path) -> EmbeddingVariantSpec:
    relative_parts = checkpoint_path.resolve().relative_to(dataset_dir.resolve()).parts
    if "output" in relative_parts:
        output_index = relative_parts.index("output")
        variant_parts = list(relative_parts[output_index + 1 : -1])
    else:
        variant_parts = list(relative_parts[:-1])

    architecture = variant_parts[0] if len(variant_parts) >= 1 else checkpoint_path.parent.parent.name
    loss_name = variant_parts[1] if len(variant_parts) >= 2 else checkpoint_path.parent.name
    variant_id = "/".join(variant_parts[:]) if variant_parts else f"{architecture}/{loss_name}"
    return EmbeddingVariantSpec(
        variant_id=variant_id,
        architecture=architecture,
        loss_name=loss_name,
        checkpoint_path=checkpoint_path,
    )


def persist_results(
    *,
    dataset_dir: Path,
    method_key: str,
    summary: dict[str, Any],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    results_dir = dataset_dir / "results"
    summaries_dir = dataset_dir / "model_summaries"
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{method_key}.jsonl"
    summary_path = summaries_dir / f"{method_key}.json"
    write_jsonl(results_path, results)

    updated_summary = dict(summary)
    updated_summary["results_path"] = str(results_path.resolve())
    updated_summary["method_key"] = method_key
    write_json(summary_path, updated_summary)
    return updated_summary


def load_existing_summary(dataset_dir: Path, method_key: str) -> dict[str, Any] | None:
    summary_path = dataset_dir / "model_summaries" / f"{method_key}.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {summary_path} to contain a JSON object.")
    payload["method_key"] = method_key
    return payload


def load_jsonl_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number} of {path}.")
        rows.append(payload)
    return rows


def load_existing_results(
    dataset_dir: Path,
    method_key: str,
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_paths: list[Path] = []
    if summary is not None:
        results_path = str(summary.get("results_path", "")).strip()
        if results_path:
            candidate_paths.append(Path(results_path))
    candidate_paths.append(dataset_dir / "results" / f"{method_key}.jsonl")

    seen_paths: set[Path] = set()
    for path in candidate_paths:
        resolved_path = path if path.is_absolute() else (dataset_dir / path)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        if resolved_path.is_file():
            return load_jsonl_results(resolved_path)
    return []


def error_summary(
    *,
    method: MethodSpec,
    message: str,
    results: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_model_summary(
        adapter_id=f"{method.provider}/{slugify(method.model_name or method.key)}",
        provider=method.provider,
        mode=method.mode,
        model_name=method.model_name,
        results=list(results or []),
        status="error",
        error_message=message,
    )


def build_method_error_results(
    *,
    method: MethodSpec,
    rows: Sequence[dict[str, Any]],
    message: str,
) -> list[dict[str, Any]]:
    adapter_id = f"{method.provider}/{slugify(method.model_name or method.key)}"
    return [
        {
            "adapter_id": adapter_id,
            "provider": method.provider,
            "mode": method.mode,
            "model_name": method.model_name,
            "example_id": str(row.get("id", "")),
            "query": str(row.get("query", "")),
            "expected_tool": str(row.get("tool", "")),
            "status": "error",
            "selected_tool": None,
            "ranked_tools": [],
            "correct_top1": None,
            "top_3_hit": None,
            "top_5_hit": None,
            "reciprocal_rank": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "error_message": message,
            "reason": "",
            "raw_response": None,
        }
        for row in rows
    ]


def build_dry_run_output(method: MethodSpec, example_count: int) -> MethodOutput:
    summary = error_summary(method=method, message="Dry run: method was not evaluated.")
    summary["status"] = "dry_run"
    summary["metrics"] = {
        "total_examples": example_count,
        "successful_examples": 0,
        "error_examples": 0,
        "mean_total_tokens": None,
        "top_1_accuracy": None,
        "top_5_accuracy": None,
        "mean_latency_ms": None,
    }
    summary["method_key"] = method.key
    return MethodOutput(summary=summary, results=[], dry_run=True)


def evaluate_method(
    *,
    args: argparse.Namespace,
    dataset: DatasetSpec,
    method: MethodSpec,
    rows: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        if method.key == "qwen_ict":
            adapter = LocalHFSelectionAdapter(
                method.model_name,
                device=args.hf_device,
                dtype=args.hf_dtype,
                ranking_limit=args.ranking_limit,
                max_new_tokens=args.hf_max_new_tokens,
                local_files_only=args.hf_local_files_only,
                pricing=None,
            )
            return adapter.evaluate(rows, tools)

        if method.key == "openai_ict":
            adapter = OpenAISelectionAdapter(
                method.model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=None,
            )
            return adapter.evaluate(rows, tools)

        if method.key == "gemini_ict":
            adapter = GeminiSelectionAdapter(
                method.model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=None,
            )
            return adapter.evaluate(rows, tools)

        if method.key == "anthropic_ict":
            adapter = AnthropicSelectionAdapter(
                method.model_name,
                ranking_limit=args.ranking_limit,
                max_output_tokens=args.api_max_output_tokens,
                timeout_seconds=args.api_timeout_seconds,
                pricing=None,
            )
            return adapter.evaluate(rows, tools)

        if method.key == "ntilc":
            if dataset.checkpoint_path is None:
                message = (
                    "NTILC checkpoint not found. "
                    f"Looked for {args.ntilc_checkpoint_glob!r} under {dataset.dataset_path.parent}."
                )
                results = build_method_error_results(method=method, rows=rows, message=message)
                return (
                    error_summary(
                        method=method,
                        message=message,
                        results=results,
                    ),
                    results,
                )
            variant = infer_embedding_variant(dataset.dataset_path.parent, dataset.checkpoint_path)
            adapter = EmbeddingSelectionAdapter(
                variant,
                device=args.embedding_device,
                ranking_limit=args.ranking_limit,
            )
            return adapter.evaluate(rows, tools)

        raise ValueError(f"Unsupported method: {method.key}")
    except Exception as exc:
        message = str(exc)
        results = build_method_error_results(method=method, rows=rows, message=message)
        return error_summary(method=method, message=message, results=results), results


def build_table_row(
    *,
    dataset_name: str,
    method: MethodSpec,
    summary: dict[str, Any],
    registry_tokens: int,
) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "dataset": dataset_name,
        "method_key": method.key,
        "method": method.label,
        "provider": summary.get("provider", method.provider),
        "mode": summary.get("mode", method.mode),
        "model_name": summary.get("model_name", method.model_name),
        "adapter_id": summary.get("adapter_id", ""),
        "status": summary.get("status", "error"),
        "error_message": summary.get("error_message"),
        "registry_tokens": 0 if method.key == "ntilc" else registry_tokens,
        "total_tokens": metrics.get("mean_total_tokens"),
        "sum_total_tokens": metrics.get("sum_total_tokens"),
        "top_1_accuracy": metrics.get("top_1_accuracy"),
        "top_5_accuracy": metrics.get("top_5_accuracy"),
        "latency_ms": metrics.get("mean_latency_ms"),
        "successful_examples": metrics.get("successful_examples"),
        "total_examples": metrics.get("total_examples"),
        "results_path": summary.get("results_path", ""),
    }


def is_failure_result(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).strip() != "ok" or row.get("correct_top1") is not True


def failure_type(row: dict[str, Any]) -> str:
    if str(row.get("status", "")).strip() != "ok":
        return "error"
    if row.get("correct_top1") is not True:
        return "incorrect_top1"
    return ""


def build_failure_rows(
    *,
    dataset_name: str,
    method: MethodSpec,
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    failure_rows: list[dict[str, Any]] = []
    for row in results:
        current_failure_type = failure_type(row)
        if not current_failure_type:
            continue
        failure_rows.append(
            {
                "dataset": dataset_name,
                "method_key": method.key,
                "method": method.label,
                "provider": row.get("provider", method.provider),
                "model": row.get("model_name", method.model_name),
                "example_id": row.get("example_id", ""),
                "query": row.get("query", ""),
                "expected_tool": row.get("expected_tool", ""),
                "selected_tool": row.get("selected_tool") or "",
                "ranked_tools": list(row.get("ranked_tools", []) or []),
                "status": row.get("status", ""),
                "failure_type": current_failure_type,
                "error_message": row.get("error_message", ""),
                "reason": row.get("reason", ""),
                "latency_ms": row.get("latency_ms"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "total_tokens": row.get("total_tokens"),
            }
        )
    return failure_rows


def task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("example_id", "")),
        str(row.get("query", "")),
        str(row.get("expected_tool", "")),
    )


def failure_matrix_cell(row: dict[str, Any]) -> str:
    status = str(row.get("status", "")).strip()
    if status != "ok":
        error_message = str(row.get("error_message", "")).strip()
        return f"error:{error_message or status or 'unknown'}"
    if row.get("correct_top1") is True:
        return "ok"
    selected_tool = str(row.get("selected_tool", "") or "").strip()
    return f"wrong:{selected_tool or '<none>'}"


def build_failure_matrix_rows(
    *,
    dataset_name: str,
    methods: Sequence[MethodSpec],
    results_by_method: dict[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    task_order: list[tuple[str, str, str]] = []
    task_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    cells_by_task: dict[tuple[str, str, str], dict[str, str]] = {}

    for method in methods:
        for row in results_by_method.get(method.key, []):
            key = task_key(row)
            if key not in task_metadata:
                task_order.append(key)
                task_metadata[key] = {
                    "dataset": dataset_name,
                    "example_id": row.get("example_id", ""),
                    "query": row.get("query", ""),
                    "expected_tool": row.get("expected_tool", ""),
                }
                cells_by_task[key] = {}
            cells_by_task[key][method.key] = failure_matrix_cell(row)

    matrix_rows: list[dict[str, Any]] = []
    for key in task_order:
        row = dict(task_metadata[key])
        has_failure = False
        for method in methods:
            cell = cells_by_task[key].get(method.key, "missing")
            row[method.key] = cell
            if cell != "ok":
                has_failure = True
        if has_failure:
            matrix_rows.append(row)
    return matrix_rows


def csv_safe_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_safe_value(row.get(field, "")) for field in fieldnames})


def write_failure_reports(
    *,
    output_dir: Path,
    failure_rows: Sequence[dict[str, Any]],
    failure_matrix_rows: Sequence[dict[str, Any]],
    methods: Sequence[MethodSpec],
    failure_rows_stem: str = "failures",
) -> dict[str, str]:
    failures_csv_path = output_dir / f"{failure_rows_stem}.csv"
    failures_json_path = output_dir / f"{failure_rows_stem}.json"
    matrix_csv_path = output_dir / "failure_matrix.csv"
    write_csv_rows(failures_csv_path, failure_rows, FAILURE_ROW_FIELDS)
    write_json(failures_json_path, list(failure_rows))
    write_csv_rows(
        matrix_csv_path,
        failure_matrix_rows,
        [*FAILURE_MATRIX_BASE_FIELDS, *[method.key for method in methods]],
    )
    return {
        f"{failure_rows_stem}_csv": str(failures_csv_path.resolve()),
        f"{failure_rows_stem}_json": str(failures_json_path.resolve()),
        "failure_matrix_csv": str(matrix_csv_path.resolve()),
    }


def write_table_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "registry_tokens",
        "total_tokens",
        "top_1_accuracy",
        "top_5_accuracy",
        "latency_ms",
        "status",
        "model_name",
        "adapter_id",
        "successful_examples",
        "total_examples",
        "results_path",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def format_number(value: Any, *, digits: int = 0, missing: str = r"\placeholder{N}") -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_percent(value: Any, *, missing: str = r"\placeholder{N}") -> str:
    if value is None:
        return f"{missing}\\%"
    return f"{float(value) * 100.0:.2f}\\%"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_latex_table(rows: Sequence[dict[str, Any]], dataset_order: Sequence[str], method_order: Sequence[str]) -> str:
    by_key = {
        (str(row["dataset"]), str(row["method_key"])): row
        for row in rows
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{Main inference-time comparison.}",
        r"\label{tab:model_comparison}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Dataset & Method & Registry Tokens & Total Tokens & Top-1 Acc. & Top-5 Acc. & Latency (ms) \\",
        r"\midrule",
    ]

    for dataset_index, dataset_name in enumerate(dataset_order):
        for method_index, method_key in enumerate(method_order):
            row = by_key.get((dataset_name, method_key))
            dataset_cell = latex_escape(dataset_name) if method_index == 0 else ""
            if row is None:
                method_cell = latex_escape(method_key)
                registry_tokens = total_tokens = latency = r"\placeholder{N}"
                top1 = top5 = r"\placeholder{N}\%"
            else:
                method_cell = latex_escape(row["method"])
                registry_tokens = format_number(row.get("registry_tokens"), digits=0)
                total_tokens = format_number(row.get("total_tokens"), digits=0)
                top1 = format_percent(row.get("top_1_accuracy"))
                top5 = format_percent(row.get("top_5_accuracy"))
                latency = format_number(row.get("latency_ms"), digits=0)

            if method_key == "ntilc":
                method_cell = rf"\textbf{{{method_cell}}}"
                registry_tokens = rf"\textbf{{{registry_tokens}}}"
                total_tokens = rf"\textbf{{{total_tokens}}}"
                top1 = rf"\textbf{{{top1}}}"
                top5 = rf"\textbf{{{top5}}}"
                latency = rf"\textbf{{{latency}}}"

            suffix = r"\\"
            if method_index == len(method_order) - 1 and dataset_index != len(dataset_order) - 1:
                suffix = r"\\\midrule"
            lines.append(
                f"{dataset_cell} & {method_cell} & {registry_tokens} & {total_tokens} & "
                f"{top1} & {top5} & {latency} {suffix}"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def print_dry_run_plan(
    *,
    dataset_specs: Sequence[DatasetSpec],
    methods: Sequence[MethodSpec],
) -> None:
    print("Dry run: no model calls will be made.")
    for dataset in dataset_specs:
        print(f"\nDataset: {dataset.name}")
        print(f"  test split: {dataset.dataset_path}")
        print(f"  tools:      {dataset.tools_path}")
        print(f"  checkpoint: {dataset.checkpoint_path or 'MISSING'}")
        for method in methods:
            print(f"  method:     {method.label} [{method.key}] -> {method.model_name}")


def main() -> None:
    args = parse_args()
    loaded_env_keys = load_env_file(args.dotenv_path)
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = resolve_dataset_specs(args)
    method_specs_by_key = build_method_specs(args)
    method_specs = [method_specs_by_key[key] for key in args.methods]

    if args.dry_run:
        print_dry_run_plan(dataset_specs=dataset_specs, methods=method_specs)

    count_tokens, token_count_metadata = build_token_counter(
        args.registry_tokenizer_model,
        local_files_only=args.registry_tokenizer_local_files_only,
    )

    all_table_rows: list[dict[str, Any]] = []
    all_failure_rows: list[dict[str, Any]] = []
    all_failure_matrix_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    for dataset in dataset_specs:
        dataset_dir = run_dir / slugify(dataset.name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Dataset: {dataset.name} ===")
        print(f"Test split: {dataset.dataset_path}")
        print(f"Tools: {dataset.tools_path}")
        if dataset.checkpoint_path is not None:
            print(f"NTILC checkpoint: {dataset.checkpoint_path}")
        else:
            print("NTILC checkpoint: MISSING")

        if not dataset.dataset_path.is_file():
            raise FileNotFoundError(f"Dataset test split not found: {dataset.dataset_path}")
        if not dataset.tools_path.is_file():
            raise FileNotFoundError(f"Tool catalog not found: {dataset.tools_path}")

        benchmark_rows = load_benchmark_rows(dataset.dataset_path)
        if args.limit > 0:
            benchmark_rows = benchmark_rows[: args.limit]
        tools = load_tool_catalog(dataset.tools_path)
        registry_tokens = count_tokens(render_tool_catalog(tools))

        model_summaries: list[dict[str, Any]] = []
        method_outputs: dict[str, MethodOutput] = {}
        pending_methods: list[MethodSpec] = []
        for method in method_specs:
            if args.dry_run:
                method_outputs[method.key] = build_dry_run_output(method, len(benchmark_rows))
                continue

            if args.skip_existing:
                existing_summary = load_existing_summary(dataset_dir, method.key)
                if existing_summary is not None:
                    existing_results = load_existing_results(dataset_dir, method.key, existing_summary)
                    print(
                        f"Loaded existing summary for {dataset.name}/{method.key} "
                        f"({len(existing_results)} result rows)"
                    )
                    method_outputs[method.key] = MethodOutput(
                        summary=existing_summary,
                        results=existing_results,
                        loaded_existing=True,
                    )
                    continue

            pending_methods.append(method)

        api_methods, local_methods = partition_methods(pending_methods)
        if api_methods:
            worker_count = min(len(api_methods), max(1, int(args.api_parallel_workers)))
            print(f"Starting {len(api_methods)} API method(s) with {worker_count} worker(s).")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_method = {}
                for method in api_methods:
                    print(f"\n--- Method: {method.label} [API queued] ---")
                    future = executor.submit(
                        evaluate_method,
                        args=args,
                        dataset=dataset,
                        method=method,
                        rows=benchmark_rows,
                        tools=tools,
                    )
                    future_to_method[future] = method

                for method in local_methods:
                    print(f"\n--- Method: {method.label} [local] ---")
                    summary, results = evaluate_method(
                        args=args,
                        dataset=dataset,
                        method=method,
                        rows=benchmark_rows,
                        tools=tools,
                    )
                    method_outputs[method.key] = MethodOutput(summary=summary, results=results)

                for future in as_completed(future_to_method):
                    method = future_to_method[future]
                    try:
                        summary, results = future.result()
                    except Exception as exc:
                        message = str(exc)
                        results = build_method_error_results(method=method, rows=benchmark_rows, message=message)
                        summary = error_summary(method=method, message=message, results=results)
                    method_outputs[method.key] = MethodOutput(summary=summary, results=results)
                    print(f"Completed API method: {method.label}")
        else:
            for method in local_methods:
                print(f"\n--- Method: {method.label} [local] ---")
                summary, results = evaluate_method(
                    args=args,
                    dataset=dataset,
                    method=method,
                    rows=benchmark_rows,
                    tools=tools,
                )
                method_outputs[method.key] = MethodOutput(summary=summary, results=results)

        results_by_method: dict[str, Sequence[dict[str, Any]]] = {}
        dataset_failure_rows: list[dict[str, Any]] = []
        for method in method_specs:
            output = method_outputs.get(method.key)
            if output is None:
                message = f"Method {method.key} was not evaluated."
                results = build_method_error_results(method=method, rows=benchmark_rows, message=message)
                output = MethodOutput(
                    summary=error_summary(method=method, message=message, results=results),
                    results=results,
                )

            summary = output.summary
            results = output.results
            if not output.dry_run and not output.loaded_existing:
                summary = persist_results(
                    dataset_dir=dataset_dir,
                    method_key=method.key,
                    summary=summary,
                    results=results,
                )
                output.summary = summary
            else:
                summary["method_key"] = method.key

            model_summaries.append(summary)
            results_by_method[method.key] = results
            dataset_failure_rows.extend(
                build_failure_rows(
                    dataset_name=dataset.name,
                    method=method,
                    results=results,
                )
            )
            table_row = build_table_row(
                dataset_name=dataset.name,
                method=method,
                summary=summary,
                registry_tokens=registry_tokens,
            )
            all_table_rows.append(table_row)
            print(
                "Result: "
                f"status={table_row['status']} "
                f"top1={round_float(table_row['top_1_accuracy'])} "
                f"top5={round_float(table_row['top_5_accuracy'])} "
                f"latency_ms={round_float(table_row['latency_ms'])}"
            )

        dataset_failure_matrix_rows = build_failure_matrix_rows(
            dataset_name=dataset.name,
            methods=method_specs,
            results_by_method=results_by_method,
        )
        failure_report_paths = write_failure_reports(
            output_dir=dataset_dir,
            failure_rows=dataset_failure_rows,
            failure_matrix_rows=dataset_failure_matrix_rows,
            methods=method_specs,
        )
        all_failure_rows.extend(dataset_failure_rows)
        all_failure_matrix_rows.extend(dataset_failure_matrix_rows)
        print(
            f"Failure rows: {len(dataset_failure_rows)}; "
            f"failure matrix tasks: {len(dataset_failure_matrix_rows)}"
        )

        dataset_summary = build_benchmark_summary(
            benchmark_name="main_inference_time_comparison_dataset",
            dataset_path=dataset.dataset_path,
            tools_path=dataset.tools_path,
            output_dir=dataset_dir,
            config={
                "dataset": dataset.name,
                "ranking_limit": args.ranking_limit,
                "example_count": len(benchmark_rows),
                "tool_count": len(tools),
                "registry_tokens": registry_tokens,
                "ntilc_checkpoint_path": str(dataset.checkpoint_path.resolve()) if dataset.checkpoint_path else "",
                "methods": [method.key for method in method_specs],
            },
            model_summaries=model_summaries,
        )
        dataset_summary["paths"].update(failure_report_paths)
        dataset_summary["failure_counts"] = {
            "failure_rows": len(dataset_failure_rows),
            "failure_matrix_tasks": len(dataset_failure_matrix_rows),
        }
        write_json(dataset_dir / "summary.json", dataset_summary)
        dataset_summaries.append(dataset_summary)

    table_rows_path = run_dir / "table_rows.csv"
    latex_path = run_dir / "table.tex"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    aggregate_failure_paths = write_failure_reports(
        output_dir=run_dir,
        failure_rows=all_failure_rows,
        failure_matrix_rows=all_failure_matrix_rows,
        methods=method_specs,
        failure_rows_stem="failure_rows",
    )

    write_table_rows_csv(table_rows_path, all_table_rows)
    latex_table = build_latex_table(
        all_table_rows,
        dataset_order=[dataset.name for dataset in dataset_specs],
        method_order=[method.key for method in method_specs],
    )
    latex_path.write_text(latex_table, encoding="utf-8")

    write_json(
        summary_path,
        {
            "created_at": now_utc_iso(),
            "benchmark": "main_inference_time_comparison",
            "paths": {
                "output_dir": str(run_dir.resolve()),
                "table_rows_csv": str(table_rows_path.resolve()),
                "latex_table": str(latex_path.resolve()),
                **aggregate_failure_paths,
            },
            "config": {
                "args": make_json_safe(vars(args)),
                "loaded_env_keys": loaded_env_keys,
                "token_count": token_count_metadata,
            },
            "datasets": dataset_summaries,
            "table_rows": all_table_rows,
            "failure_counts": {
                "failure_rows": len(all_failure_rows),
                "failure_matrix_tasks": len(all_failure_matrix_rows),
            },
        },
    )
    write_json(
        config_path,
        {
            "created_at": now_utc_iso(),
            "args": make_json_safe(vars(args)),
            "summary_path": str(summary_path.resolve()),
        },
    )

    print(f"\nWrote aggregate summary to {summary_path}")
    print(f"Wrote table rows to {table_rows_path}")
    print(f"Wrote LaTeX table to {latex_path}")
    print(f"Wrote aggregate failure rows to {run_dir / 'failure_rows.csv'}")
    print(f"Wrote aggregate failure matrix to {run_dir / 'failure_matrix.csv'}")
    print("\nLaTeX table:")
    print(latex_table)


if __name__ == "__main__":
    main()
