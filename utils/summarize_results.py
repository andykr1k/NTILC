from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import (
    DEFAULT_RANKING_LIMIT,
    build_benchmark_summary,
    load_benchmark_rows,
    now_utc_iso,
    slugify,
    summarize_result_rows,
    write_json,
)

DEFAULT_DATASET_NAME = "OSS"
DEFAULT_CHECKPOINT_FILENAME = "best.pt"
DEFAULT_RESULT_GLOB = "*.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild benchmark summary.json and config.json from an existing "
            "benchmark output results directory."
        )
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to a benchmark results directory, for example benchmark/output/run-name/results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary.json and config.json are written. Defaults to the parent of results_path.",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--tools-path", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--checkpoint-filename", default="")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--result-glob", default=DEFAULT_RESULT_GLOB)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid JSONL lines instead of skipping them and recording warnings.",
    )
    return parser.parse_args()


def load_json_if_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def first_str(*values: Any) -> str:
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def run_arg(existing_config: dict[str, Any], key: str) -> Any:
    args = existing_config.get("args")
    if not isinstance(args, dict):
        return None
    return args.get(key)


def normalize_for_match(value: str) -> str:
    return slugify(value).replace("-", "")


def infer_dataset_name(output_dir: Path) -> str:
    data_dir = REPO_ROOT / "data"
    haystack = normalize_for_match(" ".join(output_dir.parts))
    if data_dir.is_dir():
        for dataset_dir in sorted(data_dir.iterdir()):
            if dataset_dir.is_dir() and normalize_for_match(dataset_dir.name) in haystack:
                return dataset_dir.name
    return DEFAULT_DATASET_NAME


def default_dataset_path(dataset_name: str) -> Path:
    return REPO_ROOT / "data" / dataset_name / "tool_embedding_dataset_test.jsonl"


def default_tools_path(dataset_name: str) -> Path:
    return REPO_ROOT / "data" / dataset_name / "tools.json"


def default_checkpoint_root(dataset_name: str) -> Path:
    return REPO_ROOT / "data" / dataset_name / "output"


def resolve_dataset_path(
    requested_path: Path | None,
    *,
    output_dir: Path,
    source_summary: dict[str, Any],
    source_config: dict[str, Any],
) -> Path:
    if requested_path is not None:
        return requested_path.expanduser()

    from_summary = nested_get(source_summary, "paths", "dataset_path")
    from_config = run_arg(source_config, "dataset_path")
    inferred = first_str(from_summary, from_config)
    if inferred:
        return Path(inferred).expanduser()

    return default_dataset_path(infer_dataset_name(output_dir))


def resolve_tools_path(
    requested_path: Path | None,
    *,
    output_dir: Path,
    source_summary: dict[str, Any],
    source_config: dict[str, Any],
) -> Path:
    if requested_path is not None:
        return requested_path.expanduser()

    from_summary = nested_get(source_summary, "paths", "tools_path")
    from_config = run_arg(source_config, "tools_path")
    inferred = first_str(from_summary, from_config)
    if inferred:
        return Path(inferred).expanduser()

    return default_tools_path(infer_dataset_name(output_dir))


def resolve_checkpoint_root(
    requested_path: Path | None,
    *,
    output_dir: Path,
    source_summary: dict[str, Any],
    source_config: dict[str, Any],
) -> Path:
    if requested_path is not None:
        return requested_path.expanduser()

    from_summary = nested_get(source_summary, "config", "checkpoint_root")
    from_config = run_arg(source_config, "embedding_root")
    inferred = first_str(from_summary, from_config)
    if inferred:
        return Path(inferred).expanduser()

    return default_checkpoint_root(infer_dataset_name(output_dir))


def load_result_rows(
    path: Path,
    *,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid_lines: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                invalid_lines.append({"line": line_number, "error": exc.msg})
                continue

            if not isinstance(payload, dict):
                message = "JSONL result rows must be JSON objects"
                if strict:
                    raise ValueError(f"{path}:{line_number}: {message}")
                invalid_lines.append({"line": line_number, "error": message})
                continue

            rows.append(payload)

    return rows, invalid_lines


def infer_adapter_id_from_path(path: Path) -> str:
    stem = path.stem
    for prefix in ("embedding", "hybrid", "hf", "openai", "anthropic", "gemini"):
        marker = f"{prefix}-"
        if stem.startswith(marker):
            return f"{prefix}/{stem[len(marker):]}"
    return stem


def infer_provider(adapter_id: str) -> str:
    prefix = adapter_id.split("/", 1)[0]
    if prefix == "hf":
        return "huggingface"
    return prefix


def infer_mode(provider: str) -> str:
    if provider == "embedding":
        return "embedding"
    if provider == "hybrid":
        return "embedding_rerank"
    if provider == "huggingface":
        return "llm_local"
    if provider in {"openai", "anthropic", "gemini"}:
        return "llm_api"
    return ""


def infer_model_name(adapter_id: str, provider: str) -> str:
    _, _, name = adapter_id.partition("/")
    if provider == "huggingface":
        return name
    return name or adapter_id


def first_row_value(rows: Sequence[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def build_model_summary(
    *,
    adapter_id: str,
    provider: str,
    mode: str,
    model_name: str,
    results: Sequence[dict[str, Any]],
    status: str = "ok",
    error_message: str | None = None,
    results_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_status = status
    effective_error_message = error_message
    metrics = summarize_result_rows(results) if status == "ok" else None
    if (
        effective_status == "ok"
        and metrics is not None
        and metrics.get("total_examples", 0) > 0
        and metrics.get("successful_examples", 0) == 0
    ):
        effective_status = "error"
        if effective_error_message is None:
            first_error = next(
                (
                    str(row.get("error_message", "")).strip()
                    for row in results
                    if str(row.get("error_message", "")).strip()
                ),
                "",
            )
            effective_error_message = first_error or "All benchmark examples failed."

    payload = {
        "adapter_id": adapter_id,
        "provider": provider,
        "mode": mode,
        "model_name": model_name,
        "status": effective_status,
        "error_message": effective_error_message,
        "metrics": metrics,
        "results_path": str(results_path.resolve()) if results_path is not None else "",
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def infer_variant_metadata(
    *,
    provider: str,
    model_name: str,
    rows: Sequence[dict[str, Any]],
    checkpoint_root: Path,
    checkpoint_filename: str,
) -> dict[str, Any]:
    if provider not in {"embedding", "hybrid"}:
        return {}

    variant_id = model_name.split("+", 1)[0].strip()
    parts = [part for part in variant_id.split("/") if part]
    if len(parts) < 2:
        return {}

    metadata: dict[str, Any] = {
        "checkpoint_path": str((checkpoint_root / Path(*parts) / checkpoint_filename).resolve()),
        "architecture": parts[0],
        "loss_name": parts[1],
    }
    if len(parts) > 2:
        metadata["variant_name"] = "/".join(parts[2:])

    if provider == "hybrid":
        _, separator, reranker_model = model_name.partition("+")
        if separator and reranker_model.strip():
            metadata["reranker_model"] = reranker_model.strip()
        embedding_top_k = first_row_value(rows, "embedding_top_k")
        if isinstance(embedding_top_k, int):
            metadata["embedding_top_k"] = embedding_top_k

    return metadata


def summarize_result_file(
    path: Path,
    *,
    strict: bool,
    checkpoint_root: Path,
    checkpoint_filename: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, invalid_lines = load_result_rows(path, strict=strict)
    adapter_id = first_str(first_row_value(rows, "adapter_id"), infer_adapter_id_from_path(path))
    provider = first_str(first_row_value(rows, "provider"), infer_provider(adapter_id))
    mode = first_str(first_row_value(rows, "mode"), infer_mode(provider))
    model_name = first_str(first_row_value(rows, "model_name"), infer_model_name(adapter_id, provider))

    metadata = infer_variant_metadata(
        provider=provider,
        model_name=model_name,
        rows=rows,
        checkpoint_root=checkpoint_root,
        checkpoint_filename=checkpoint_filename,
    )
    if invalid_lines:
        metadata["invalid_json_lines"] = invalid_lines

    status = "ok" if rows else "error"
    error_message = None if rows else "No result rows found."
    summary = build_model_summary(
        adapter_id=adapter_id,
        provider=provider,
        mode=mode,
        model_name=model_name,
        results=rows,
        status=status,
        error_message=error_message,
        results_path=path,
        metadata=metadata,
    )
    file_info = {
        "path": str(path.resolve()),
        "row_count": len(rows),
        "invalid_json_line_count": len(invalid_lines),
        "max_ranked_tools": max(
            (
                len(row.get("ranked_tools", []))
                for row in rows
                if isinstance(row.get("ranked_tools"), list)
            ),
            default=0,
        ),
        "adapter_id": adapter_id,
        "status": summary["status"],
    }
    return summary, file_info


def count_dataset_examples(dataset_path: Path, limit: Any, fallback: int) -> int:
    try:
        rows = load_benchmark_rows(dataset_path)
    except Exception:
        return fallback

    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = 0

    if limit_value > 0:
        return min(len(rows), limit_value)
    return len(rows)


def list_model_names(model_summaries: Sequence[dict[str, Any]], provider: str) -> list[str]:
    names = {
        str(summary.get("model_name", "")).strip()
        for summary in model_summaries
        if summary.get("provider") == provider and str(summary.get("model_name", "")).strip()
    }
    return sorted(names)


def infer_hybrid_reranker_model(
    model_summaries: Sequence[dict[str, Any]],
    existing_value: Any,
) -> str:
    value = first_str(existing_value)
    if value:
        return value

    reranker_names = set()
    for summary in model_summaries:
        if summary.get("provider") != "hybrid":
            continue
        model_name = str(summary.get("model_name", ""))
        _, separator, reranker_model = model_name.partition("+")
        if separator and reranker_model.strip():
            reranker_names.add(reranker_model.strip())
    return sorted(reranker_names)[0] if len(reranker_names) == 1 else ""


def build_summary_config(
    *,
    source_summary: dict[str, Any],
    source_config: dict[str, Any],
    results_dir: Path,
    checkpoint_root: Path,
    checkpoint_filename: str,
    model_summaries: Sequence[dict[str, Any]],
    result_file_info: Sequence[dict[str, Any]],
    dataset_path: Path,
) -> dict[str, Any]:
    existing_config = source_summary.get("config")
    config = dict(existing_config) if isinstance(existing_config, dict) else {}

    max_row_count = max((int(info["row_count"]) for info in result_file_info), default=0)
    example_count = count_dataset_examples(
        dataset_path,
        run_arg(source_config, "limit"),
        fallback=max_row_count,
    )
    ranking_limit = run_arg(source_config, "ranking_limit")
    if not isinstance(ranking_limit, int):
        inferred_ranking_limit = max((int(info["max_ranked_tools"]) for info in result_file_info), default=0)
        ranking_limit = int(config.get("ranking_limit") or inferred_ranking_limit or DEFAULT_RANKING_LIMIT)

    config.update(
        {
            "checkpoint_root": str(checkpoint_root.resolve()),
            "checkpoint_filename": checkpoint_filename,
            "ranking_limit": ranking_limit,
            "example_count": example_count,
            "embedding_top_k": config.get("embedding_top_k", run_arg(source_config, "embedding_top_k") or 5),
            "hybrid_top_k": config.get("hybrid_top_k", run_arg(source_config, "hybrid_top_k") or 5),
            "hybrid_reranker_model": infer_hybrid_reranker_model(
                model_summaries,
                config.get("hybrid_reranker_model", run_arg(source_config, "hybrid_reranker_model")),
            ),
            "hf_models": config.get("hf_models", run_arg(source_config, "hf_model") or list_model_names(model_summaries, "huggingface")),
            "openai_models": config.get("openai_models", run_arg(source_config, "openai_model") or list_model_names(model_summaries, "openai")),
            "anthropic_models": config.get(
                "anthropic_models",
                run_arg(source_config, "anthropic_model") or list_model_names(model_summaries, "anthropic"),
            ),
            "gemini_models": config.get("gemini_models", run_arg(source_config, "gemini_model") or list_model_names(model_summaries, "gemini")),
            "pricing_path": config.get("pricing_path", first_str(run_arg(source_config, "pricing_path"))),
            "dotenv_path": config.get("dotenv_path", first_str(run_arg(source_config, "dotenv_path"))),
            "loaded_env_keys": config.get("loaded_env_keys", []),
            "results_dir": str(results_dir.resolve()),
            "generated_from_results": True,
            "result_file_count": len(result_file_info),
        }
    )

    invalid_line_count = sum(int(info["invalid_json_line_count"]) for info in result_file_info)
    if invalid_line_count:
        config["invalid_json_line_count"] = invalid_line_count

    empty_result_files = [info["path"] for info in result_file_info if int(info["row_count"]) == 0]
    if empty_result_files:
        config["empty_result_files"] = empty_result_files

    return config


def main() -> None:
    args = parse_args()
    results_dir = args.results_path.expanduser()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    source_output_dir = results_dir.parent
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else source_output_dir
    summary_path = args.summary_path.expanduser() if args.summary_path is not None else output_dir / "summary.json"
    config_path = args.config_path.expanduser() if args.config_path is not None else output_dir / "config.json"

    source_summary = load_json_if_file(source_output_dir / "summary.json")
    source_config = load_json_if_file(source_output_dir / "config.json")

    dataset_path = resolve_dataset_path(
        args.dataset_path,
        output_dir=source_output_dir,
        source_summary=source_summary,
        source_config=source_config,
    )
    tools_path = resolve_tools_path(
        args.tools_path,
        output_dir=source_output_dir,
        source_summary=source_summary,
        source_config=source_config,
    )
    checkpoint_root = resolve_checkpoint_root(
        args.checkpoint_root,
        output_dir=source_output_dir,
        source_summary=source_summary,
        source_config=source_config,
    )
    checkpoint_filename = first_str(
        args.checkpoint_filename,
        nested_get(source_summary, "config", "checkpoint_filename"),
        run_arg(source_config, "checkpoint_filename"),
        DEFAULT_CHECKPOINT_FILENAME,
    )

    result_paths = sorted(path for path in results_dir.glob(args.result_glob) if path.is_file())
    if not result_paths:
        raise FileNotFoundError(f"No result files matching {args.result_glob!r} found in {results_dir}")

    model_summaries: list[dict[str, Any]] = []
    result_file_info: list[dict[str, Any]] = []
    for result_path in result_paths:
        summary, file_info = summarize_result_file(
            result_path,
            strict=args.strict,
            checkpoint_root=checkpoint_root,
            checkpoint_filename=checkpoint_filename,
        )
        model_summaries.append(summary)
        result_file_info.append(file_info)

    benchmark_name = first_str(args.benchmark_name, source_summary.get("benchmark"), "tool_selection")
    summary_config = build_summary_config(
        source_summary=source_summary,
        source_config=source_config,
        results_dir=results_dir,
        checkpoint_root=checkpoint_root,
        checkpoint_filename=checkpoint_filename,
        model_summaries=model_summaries,
        result_file_info=result_file_info,
        dataset_path=dataset_path,
    )
    summary = build_benchmark_summary(
        benchmark_name=benchmark_name,
        dataset_path=dataset_path,
        tools_path=tools_path,
        output_dir=output_dir,
        config=summary_config,
        model_summaries=model_summaries,
    )
    write_json(summary_path, summary)

    run_name = first_str(args.run_name, source_config.get("run_name"), source_output_dir.name)
    write_json(
        config_path,
        {
            "created_at": now_utc_iso(),
            "args": make_json_safe(
                {
                    "results_path": results_dir,
                    "output_dir": output_dir,
                    "summary_path": summary_path,
                    "config_path": config_path,
                    "dataset_path": dataset_path,
                    "tools_path": tools_path,
                    "checkpoint_root": checkpoint_root,
                    "checkpoint_filename": checkpoint_filename,
                    "benchmark_name": benchmark_name,
                    "run_name": run_name,
                    "result_glob": args.result_glob,
                    "strict": args.strict,
                }
            ),
            "run_name": run_name,
            "summary_path": str(summary_path.resolve()),
        },
    )

    print(f"Wrote benchmark summary to {summary_path}")
    print(f"Wrote benchmark config to {config_path}")
    print("Leaderboard:")
    for row in summary["leaderboard"]:
        print(
            f"  #{row['rank']} {row['adapter_id']}: "
            f"top1={row['top_1_accuracy']}, "
            f"mrr={row['mean_reciprocal_rank']}, "
            f"latency_ms={row['mean_latency_ms']}"
        )


if __name__ == "__main__":
    main()
