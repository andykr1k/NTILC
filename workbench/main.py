import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


st.set_page_config(
    page_title="Agentic Tool-Calling Eval Workbench",
    page_icon=":bar_chart:",
    layout="wide",
)


WORKBENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKBENCH_DIR.parent
DEFAULT_EVAL_JSON = WORKBENCH_DIR / "eval_runs.json"
DEFAULT_WANDB_DIR = PROJECT_ROOT / "wandb"
MAX_WANDB_SUMMARY_SIZE_BYTES = 2_000_000


RUN_TEMPLATE: Dict[str, Any] = {
    "run_id": "run_20260223_120000_stabletoolbench_gpt4o",
    "created_at": "2026-02-23T12:00:00Z",
    "benchmark": "stabletoolbench",
    "split": "test",
    "model": "gpt-4o",
    "provider": "openai",
    "prompt_version": "v3.2",
    "toolset_version": "market_tools_v1",
    "notes": "Optional notes",
    "tags": ["baseline", "tool-calling"],
    "config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "concurrency": 4},
    "metrics": {
        "success_rate": 0.68,
        "tokens_per_second": 145.2,
        "energy_joules": 3200.0,
        "cost_usd": 3.25,
    },
    "tasks": [
        {
            "task_id": "stabletoolbench_001",
            "category": "finance",
            "difficulty": "medium",
            "success": True,
            "expected_tool": "search_market",
            "predicted_tool": "search_market",
            "args_correct": True,
            "latency_ms": 1320,
            "input_tokens": 640,
            "output_tokens": 98,
            "energy_joules": 10.4,
            "cost_usd": 0.012,
        },
        {
            "task_id": "stabletoolbench_002",
            "category": "travel",
            "difficulty": "hard",
            "success": False,
            "expected_tool": "book_trip",
            "predicted_tool": "search_hotels",
            "args_correct": False,
            "error_type": "wrong_tool",
            "latency_ms": 4200,
            "input_tokens": 811,
            "output_tokens": 152,
            "energy_joules": 16.1,
            "cost_usd": 0.021,
        },
    ],
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
            if math.isnan(parsed) or math.isinf(parsed):
                return None
            return parsed
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "pass", "passed", "success"}:
            return True
        if normalized in {"false", "0", "no", "n", "fail", "failed", "error"}:
            return False
    return None


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _safe_percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(pd.Series(clean).quantile(q))


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _value_from_aliases(payload: Dict[str, Any], aliases: List[str]) -> Optional[float]:
    for alias in aliases:
        if alias in payload:
            parsed = _to_float(payload[alias])
            if parsed is not None:
                return parsed
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower in lowered:
            parsed = _to_float(lowered[alias_lower])
            if parsed is not None:
                return parsed
    return None


def _normalize_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(item).strip() for item in tags if str(item).strip()]
    if isinstance(tags, str):
        return [part.strip() for part in tags.split(",") if part.strip()]
    return []


def _parse_created_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    if text.startswith("run-"):
        match = re.match(r"^run-(\d{8})_(\d{6})-[A-Za-z0-9]+$", text)
        if match:
            try:
                parsed = datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _pretty_timestamp(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_json_file(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _save_json_file(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _ensure_eval_store(path: Path) -> None:
    if path.exists():
        return
    _save_json_file(path, {"runs": []})


@st.cache_data(show_spinner=False)
def load_eval_runs(path_str: str, mtime: float) -> List[Dict[str, Any]]:
    del mtime
    path = Path(path_str)
    payload = _load_json_file(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        runs = payload.get("runs", [])
        if isinstance(runs, list):
            return [item for item in runs if isinstance(item, dict)]
    return []


def save_eval_runs(path: Path, runs: List[Dict[str, Any]]) -> None:
    _save_json_file(path, {"runs": runs})
    load_eval_runs.clear()


def _normalize_tasks(tasks: Any) -> List[Dict[str, Any]]:
    if not isinstance(tasks, list):
        return []
    return [item for item in tasks if isinstance(item, dict)]


def _task_success(task: Dict[str, Any]) -> Optional[bool]:
    for key in ("success", "passed", "task_success"):
        if key in task:
            parsed = _to_bool(task.get(key))
            if parsed is not None:
                return parsed
    status = task.get("status")
    if status:
        parsed = _to_bool(status)
        if parsed is not None:
            return parsed
    return None


def _task_error_type(task: Dict[str, Any]) -> str:
    for key in ("error_type", "failure_reason", "error", "status"):
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def _normalize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(run)
    normalized["run_id"] = str(
        run.get("run_id") or run.get("id") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    normalized["benchmark"] = str(run.get("benchmark", "unknown"))
    normalized["model"] = str(run.get("model", run.get("model_name", "unknown")))
    normalized["provider"] = str(run.get("provider", "unknown"))
    normalized["split"] = str(run.get("split", "test"))
    normalized["prompt_version"] = str(run.get("prompt_version", ""))
    normalized["toolset_version"] = str(run.get("toolset_version", ""))
    normalized["notes"] = str(run.get("notes", ""))
    normalized["tags"] = _normalize_tags(run.get("tags", []))
    normalized["tasks"] = _normalize_tasks(run.get("tasks", []))
    normalized["metrics"] = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    normalized["config"] = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    normalized["created_at"] = run.get("created_at")
    return normalized


def _compute_run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    metrics = run.get("metrics", {})
    tasks = run.get("tasks", [])

    task_success_values: List[Optional[bool]] = [_task_success(task) for task in tasks]
    success_values = [value for value in task_success_values if value is not None]
    success_count = sum(1 for value in success_values if value)
    task_count = len(tasks)
    success_rate_from_tasks = _safe_div(float(success_count), float(len(success_values))) if success_values else None

    expected_pred_pairs = [
        (str(task.get("expected_tool", "")).strip(), str(task.get("predicted_tool", "")).strip())
        for task in tasks
        if str(task.get("expected_tool", "")).strip() and str(task.get("predicted_tool", "")).strip()
    ]
    tool_accuracy_from_tasks = (
        sum(1 for expected, predicted in expected_pred_pairs if expected == predicted) / len(expected_pred_pairs)
        if expected_pred_pairs
        else None
    )

    args_correct_values = [
        _to_bool(task.get("args_correct"))
        for task in tasks
        if task.get("args_correct") is not None
    ]
    args_accuracy_from_tasks = (
        sum(1 for value in args_correct_values if value) / len(args_correct_values)
        if args_correct_values
        else None
    )

    hallucination_values = [
        _to_bool(task.get("tool_hallucination", task.get("hallucinated_tool")))
        for task in tasks
        if task.get("tool_hallucination") is not None or task.get("hallucinated_tool") is not None
    ]
    hallucination_rate_from_tasks = (
        sum(1 for value in hallucination_values if value) / len(hallucination_values)
        if hallucination_values
        else None
    )

    latency_values = [
        _to_float(task.get("latency_ms", task.get("duration_ms")))
        for task in tasks
        if task.get("latency_ms") is not None or task.get("duration_ms") is not None
    ]
    input_tokens = [_to_float(task.get("input_tokens")) for task in tasks if task.get("input_tokens") is not None]
    output_tokens = [_to_float(task.get("output_tokens")) for task in tasks if task.get("output_tokens") is not None]
    total_tokens_task_level = [
        _to_float(task.get("total_tokens"))
        for task in tasks
        if task.get("total_tokens") is not None
    ]
    energy_values = [_to_float(task.get("energy_joules")) for task in tasks if task.get("energy_joules") is not None]
    cost_values = [_to_float(task.get("cost_usd")) for task in tasks if task.get("cost_usd") is not None]

    total_input_tokens = sum(value for value in input_tokens if value is not None) if input_tokens else None
    total_output_tokens = sum(value for value in output_tokens if value is not None) if output_tokens else None
    total_tokens_from_tasks: Optional[float]
    if total_tokens_task_level:
        total_tokens_from_tasks = sum(value for value in total_tokens_task_level if value is not None)
    elif total_input_tokens is not None or total_output_tokens is not None:
        total_tokens_from_tasks = (total_input_tokens or 0.0) + (total_output_tokens or 0.0)
    else:
        total_tokens_from_tasks = None

    duration_s = _value_from_aliases(
        {**run, **metrics},
        ["duration_s", "runtime_s", "wall_time_s", "elapsed_s", "total_time_s"],
    )
    if duration_s is None and latency_values:
        latency_sum_ms = sum(value for value in latency_values if value is not None)
        duration_s = latency_sum_ms / 1000.0

    summary = {
        "run_id": run["run_id"],
        "benchmark": run.get("benchmark", "unknown"),
        "model": run.get("model", "unknown"),
        "provider": run.get("provider", "unknown"),
        "split": run.get("split", "test"),
        "prompt_version": run.get("prompt_version", ""),
        "toolset_version": run.get("toolset_version", ""),
        "tags": run.get("tags", []),
        "notes": run.get("notes", ""),
        "created_at": run.get("created_at"),
        "total_tasks": task_count if task_count else int(_value_from_aliases(metrics, ["total_tasks", "num_tasks"]) or 0),
        "success_count": success_count if success_values else _value_from_aliases(metrics, ["success_count"]),
        "success_rate": success_rate_from_tasks
        if success_rate_from_tasks is not None
        else _value_from_aliases(metrics, ["success_rate", "task_success_rate", "accuracy", "overall_accuracy"]),
        "tool_accuracy": tool_accuracy_from_tasks
        if tool_accuracy_from_tasks is not None
        else _value_from_aliases(metrics, ["tool_accuracy", "tool_selection_accuracy"]),
        "args_accuracy": args_accuracy_from_tasks
        if args_accuracy_from_tasks is not None
        else _value_from_aliases(metrics, ["args_accuracy", "argument_accuracy", "parameter_accuracy"]),
        "hallucination_rate": hallucination_rate_from_tasks
        if hallucination_rate_from_tasks is not None
        else _value_from_aliases(metrics, ["hallucination_rate", "tool_hallucination_rate"]),
        "avg_latency_ms": _safe_mean(latency_values)
        if latency_values
        else _value_from_aliases(metrics, ["avg_latency_ms", "latency_ms"]),
        "p95_latency_ms": _safe_percentile(latency_values, 0.95)
        if latency_values
        else _value_from_aliases(metrics, ["p95_latency_ms"]),
        "duration_s": duration_s,
        "input_tokens": total_input_tokens
        if total_input_tokens is not None
        else _value_from_aliases(metrics, ["input_tokens", "prompt_tokens"]),
        "output_tokens": total_output_tokens
        if total_output_tokens is not None
        else _value_from_aliases(metrics, ["output_tokens", "completion_tokens"]),
        "total_tokens": total_tokens_from_tasks
        if total_tokens_from_tasks is not None
        else _value_from_aliases(metrics, ["total_tokens", "tokens"]),
        "tokens_per_second": _safe_div(total_tokens_from_tasks, duration_s)
        if total_tokens_from_tasks is not None and duration_s
        else _value_from_aliases(metrics, ["tokens_per_second", "throughput_tps"]),
        "energy_joules": sum(value for value in energy_values if value is not None)
        if energy_values
        else _value_from_aliases(metrics, ["energy_joules", "total_energy_joules", "energy"]),
        "cost_usd": sum(value for value in cost_values if value is not None)
        if cost_values
        else _value_from_aliases(metrics, ["cost_usd", "total_cost_usd", "cost"]),
        "tasks": tasks,
    }

    summary["energy_per_task"] = _safe_div(summary.get("energy_joules"), float(summary["total_tasks"] or 0))
    summary["cost_per_success"] = _safe_div(summary.get("cost_usd"), float(summary.get("success_count") or 0))

    return summary


def build_eval_frames(runs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized_runs = [_normalize_run(run) for run in runs]
    run_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    for run in normalized_runs:
        summary = _compute_run_summary(run)
        created_at = _parse_created_at(summary.get("created_at"))
        summary["created_at_dt"] = created_at
        summary["created_at_display"] = _pretty_timestamp(created_at)
        run_rows.append(summary)

        for idx, task in enumerate(run.get("tasks", []), start=1):
            task_row = dict(task)
            task_row["run_id"] = run["run_id"]
            task_row["benchmark"] = run.get("benchmark")
            task_row["model"] = run.get("model")
            task_row["provider"] = run.get("provider")
            task_row["task_index"] = idx
            task_row["task_success"] = _task_success(task)
            task_row["error_type"] = _task_error_type(task) if _task_success(task) is False else ""
            task_rows.append(task_row)

    run_df = pd.DataFrame(run_rows)
    task_df = pd.DataFrame(task_rows)

    if not run_df.empty:
        for numeric_col in [
            "success_rate",
            "tool_accuracy",
            "args_accuracy",
            "hallucination_rate",
            "avg_latency_ms",
            "p95_latency_ms",
            "duration_s",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tokens_per_second",
            "energy_joules",
            "energy_per_task",
            "cost_usd",
            "cost_per_success",
        ]:
            if numeric_col in run_df.columns:
                run_df[numeric_col] = pd.to_numeric(run_df[numeric_col], errors="coerce")
        run_df["total_tasks"] = pd.to_numeric(run_df.get("total_tasks"), errors="coerce").fillna(0).astype(int)
        run_df["success_count"] = pd.to_numeric(run_df.get("success_count"), errors="coerce").fillna(0).astype(int)
        run_df["tags_display"] = run_df["tags"].apply(lambda values: ", ".join(values) if isinstance(values, list) else "")
        run_df = run_df.sort_values(by=["created_at_dt", "run_id"], ascending=[False, True], na_position="last")

    if not task_df.empty:
        for numeric_col in ["latency_ms", "duration_ms", "input_tokens", "output_tokens", "total_tokens", "energy_joules", "cost_usd"]:
            if numeric_col in task_df.columns:
                task_df[numeric_col] = pd.to_numeric(task_df[numeric_col], errors="coerce")

    return run_df, task_df


def _format_pct(value: Any) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "-"
    return f"{parsed * 100:.2f}%"


def _parse_json_payload(raw_payload: Optional[str], uploaded_file: Any) -> List[Dict[str, Any]]:
    if uploaded_file is not None:
        try:
            data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
    elif raw_payload:
        try:
            data = json.loads(raw_payload)
        except json.JSONDecodeError:
            return []
    else:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            return [item for item in tasks if isinstance(item, dict)]
    return []


def render_eval_tab(run_df: pd.DataFrame, task_df: pd.DataFrame) -> None:
    st.subheader("Eval Analytics")
    st.caption("JSON-backed benchmarking with analytics computed at app start.")

    if run_df.empty:
        st.info("No eval runs yet. Use the 'Create Eval Run' tab or place runs in workbench/eval_runs.json.")
        st.code(json.dumps({"runs": [RUN_TEMPLATE]}, indent=2), language="json")
        return

    col1, col2, col3, col4 = st.columns(4)
    all_benchmarks = sorted(value for value in run_df["benchmark"].dropna().unique().tolist() if str(value).strip())
    all_models = sorted(value for value in run_df["model"].dropna().unique().tolist() if str(value).strip())
    all_providers = sorted(value for value in run_df["provider"].dropna().unique().tolist() if str(value).strip())
    all_tags = sorted({tag for tags in run_df["tags"] for tag in (tags if isinstance(tags, list) else [])})

    selected_benchmarks = col1.multiselect("Benchmark", all_benchmarks, default=all_benchmarks)
    selected_models = col2.multiselect("Model", all_models, default=all_models)
    selected_providers = col3.multiselect("Provider", all_providers, default=all_providers)
    selected_tags = col4.multiselect("Tags", all_tags, default=[])

    filtered = run_df[
        run_df["benchmark"].isin(selected_benchmarks)
        & run_df["model"].isin(selected_models)
        & run_df["provider"].isin(selected_providers)
    ].copy()

    if selected_tags:
        filtered = filtered[
            filtered["tags"].apply(
                lambda tags: bool(set(selected_tags).intersection(set(tags))) if isinstance(tags, list) else False
            )
        ]

    if filtered.empty:
        st.warning("No runs match current filters.")
        return

    weighted_success_rate = _safe_div(float(filtered["success_count"].sum()), float(filtered["total_tasks"].sum()))
    weighted_success_rate = weighted_success_rate if weighted_success_rate is not None else filtered["success_rate"].mean()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Runs", f"{len(filtered)}")
    kpi_cols[1].metric("Success Rate", _format_pct(weighted_success_rate))
    kpi_cols[2].metric("Avg Tokens/sec", f"{filtered['tokens_per_second'].mean():.2f}" if filtered["tokens_per_second"].notna().any() else "-")
    kpi_cols[3].metric("Avg Energy/Task (J)", f"{filtered['energy_per_task'].mean():.2f}" if filtered["energy_per_task"].notna().any() else "-")
    kpi_cols[4].metric("Avg Cost/Success ($)", f"{filtered['cost_per_success'].mean():.4f}" if filtered["cost_per_success"].notna().any() else "-")

    st.markdown("#### Leaderboard")
    leaderboard_cols = [
        "run_id",
        "created_at_display",
        "benchmark",
        "model",
        "provider",
        "total_tasks",
        "success_count",
        "success_rate",
        "tool_accuracy",
        "tokens_per_second",
        "energy_per_task",
        "cost_per_success",
        "tags_display",
    ]
    leaderboard = filtered[leaderboard_cols].sort_values(
        by=["success_rate", "tool_accuracy", "tokens_per_second"],
        ascending=[False, False, False],
        na_position="last",
    )
    st.dataframe(
        leaderboard,
        use_container_width=True,
        hide_index=True,
        column_config={
            "success_rate": st.column_config.NumberColumn("success_rate", format="%.4f"),
            "tool_accuracy": st.column_config.NumberColumn("tool_accuracy", format="%.4f"),
            "tokens_per_second": st.column_config.NumberColumn("tokens_per_second", format="%.2f"),
            "energy_per_task": st.column_config.NumberColumn("energy_per_task", format="%.2f"),
            "cost_per_success": st.column_config.NumberColumn("cost_per_success", format="%.4f"),
        },
    )

    st.markdown("#### Compare Runs")
    compare_ids = st.multiselect(
        "Select runs to compare",
        options=filtered["run_id"].tolist(),
        default=filtered["run_id"].head(3).tolist(),
    )
    if compare_ids:
        compare_df = filtered[filtered["run_id"].isin(compare_ids)][
            ["run_id", "success_rate", "tool_accuracy", "tokens_per_second", "energy_per_task", "cost_per_success"]
        ]
        melted = compare_df.melt(id_vars="run_id", var_name="metric", value_name="value")
        chart = px.bar(
            melted,
            x="run_id",
            y="value",
            color="metric",
            barmode="group",
            title="Selected Run Comparison",
            height=380,
        )
        st.plotly_chart(chart, use_container_width=True)

    st.markdown("#### Performance Views")
    perf_col1, perf_col2 = st.columns(2)
    scatter_df = filtered.dropna(subset=["success_rate", "tokens_per_second"])
    if not scatter_df.empty:
        perf_col1.plotly_chart(
            px.scatter(
                scatter_df,
                x="tokens_per_second",
                y="success_rate",
                color="model",
                symbol="benchmark",
                hover_data=["run_id", "provider", "total_tasks"],
                title="Accuracy vs Throughput",
                height=360,
            ),
            use_container_width=True,
        )
    else:
        perf_col1.info("Need both success rate and tokens/sec for scatter plot.")

    energy_df = filtered.dropna(subset=["success_rate", "energy_per_task"])
    if not energy_df.empty:
        perf_col2.plotly_chart(
            px.scatter(
                energy_df,
                x="energy_per_task",
                y="success_rate",
                color="model",
                symbol="provider",
                hover_data=["run_id", "benchmark"],
                title="Accuracy vs Energy per Task",
                height=360,
            ),
            use_container_width=True,
        )
    else:
        perf_col2.info("Need both success rate and energy/task for energy plot.")

    trend_df = filtered.dropna(subset=["created_at_dt", "success_rate"]).copy()
    if len(trend_df) >= 2:
        st.plotly_chart(
            px.line(
                trend_df.sort_values("created_at_dt"),
                x="created_at_dt",
                y="success_rate",
                color="model",
                markers=True,
                hover_data=["run_id", "benchmark"],
                title="Success Rate Trend",
                height=350,
            ),
            use_container_width=True,
        )

    st.markdown("#### Failure Analysis")
    if task_df.empty:
        st.info("No task-level data available for failure analysis.")
    else:
        task_filtered = task_df[task_df["run_id"].isin(filtered["run_id"])].copy()
        failed_tasks = task_filtered[task_filtered["task_success"] == False]  # noqa: E712
        if failed_tasks.empty:
            st.success("No failed tasks in the current filter selection.")
        else:
            fail_col1, fail_col2 = st.columns(2)
            error_breakdown = (
                failed_tasks.groupby("error_type", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
            )
            fail_col1.dataframe(error_breakdown.head(20), use_container_width=True, hide_index=True)

            if "expected_tool" in failed_tasks.columns and "predicted_tool" in failed_tasks.columns:
                mismatch = (
                    failed_tasks.assign(
                        expected_tool=failed_tasks["expected_tool"].fillna(""),
                        predicted_tool=failed_tasks["predicted_tool"].fillna(""),
                    )
                    .groupby(["expected_tool", "predicted_tool"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                fail_col2.dataframe(mismatch.head(20), use_container_width=True, hide_index=True)
            else:
                fail_col2.info("expected_tool / predicted_tool not present in task rows.")

            st.markdown("#### Task Drill-Down")
            drill_run = st.selectbox("Run", filtered["run_id"].tolist(), key="drill_run_id")
            drill_tasks = task_filtered[task_filtered["run_id"] == drill_run].copy()
            display_columns = [
                col
                for col in [
                    "task_index",
                    "task_id",
                    "category",
                    "difficulty",
                    "task_success",
                    "expected_tool",
                    "predicted_tool",
                    "args_correct",
                    "latency_ms",
                    "total_tokens",
                    "energy_joules",
                    "cost_usd",
                    "error_type",
                ]
                if col in drill_tasks.columns
            ]
            st.dataframe(drill_tasks[display_columns], use_container_width=True, hide_index=True)
            if not drill_tasks.empty:
                drill_task_index = st.selectbox(
                    "Task row",
                    drill_tasks["task_index"].tolist(),
                    key="drill_task_index",
                )
                selected_task = drill_tasks[drill_tasks["task_index"] == drill_task_index].iloc[0].to_dict()
                st.json(selected_task)


def _build_manual_metrics(
    success_rate: str,
    tool_accuracy: str,
    tokens_per_second: str,
    energy_joules: str,
    cost_usd: str,
    duration_s: str,
) -> Dict[str, float]:
    values = {
        "success_rate": _to_float(success_rate),
        "tool_accuracy": _to_float(tool_accuracy),
        "tokens_per_second": _to_float(tokens_per_second),
        "energy_joules": _to_float(energy_joules),
        "cost_usd": _to_float(cost_usd),
        "duration_s": _to_float(duration_s),
    }
    return {key: value for key, value in values.items() if value is not None}


def render_create_tab(eval_json_path: Path, existing_runs: List[Dict[str, Any]]) -> None:
    st.subheader("Create Eval Run")
    st.caption(f"Runs are stored in `{eval_json_path}` (JSON only).")

    with st.expander("JSON Template"):
        st.code(json.dumps({"runs": [RUN_TEMPLATE]}, indent=2), language="json")

    with st.form("create_eval_run_form", clear_on_submit=False):
        top_col1, top_col2, top_col3 = st.columns(3)
        benchmark = top_col1.selectbox(
            "Benchmark",
            options=["stabletoolbench", "toolbench", "custom"],
            index=0,
        )
        split = top_col2.selectbox("Split", options=["test", "val", "train", "custom"], index=0)
        provider = top_col3.text_input("Provider", value="openai")

        model_col1, model_col2, model_col3 = st.columns(3)
        model = model_col1.text_input("Model", value="gpt-4o")
        prompt_version = model_col2.text_input("Prompt Version", value="v1")
        toolset_version = model_col3.text_input("Toolset Version", value="tools_v1")

        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        temperature = cfg_col1.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        top_p = cfg_col2.number_input("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        max_tokens = cfg_col3.number_input("Max Tokens", min_value=1, value=512, step=32)
        concurrency = cfg_col4.number_input("Concurrency", min_value=1, value=4, step=1)

        notes = st.text_area("Notes", value="", height=80)
        tags = st.text_input("Tags (comma-separated)", value="")

        st.markdown("Task Data (optional)")
        ingest_mode = st.radio(
            "Provide task-level rows",
            options=["None", "Paste JSON", "Upload JSON file"],
            horizontal=True,
        )
        task_json_input = ""
        uploaded_tasks = None
        if ingest_mode == "Paste JSON":
            task_json_input = st.text_area(
                "Tasks JSON (list of task objects OR {'tasks': [...]})",
                value="",
                height=160,
            )
        elif ingest_mode == "Upload JSON file":
            uploaded_tasks = st.file_uploader("Upload task JSON", type=["json"])

        st.markdown("Manual Run Metrics (optional)")
        m1, m2, m3 = st.columns(3)
        manual_success_rate = m1.text_input("Success Rate (0-1)", value="")
        manual_tool_accuracy = m2.text_input("Tool Accuracy (0-1)", value="")
        manual_tokens_per_sec = m3.text_input("Tokens/sec", value="")

        m4, m5, m6 = st.columns(3)
        manual_energy = m4.text_input("Energy Joules", value="")
        manual_cost = m5.text_input("Cost USD", value="")
        manual_duration = m6.text_input("Duration Seconds", value="")

        submit = st.form_submit_button("Create Run")

    if not submit:
        return

    tasks = _parse_json_payload(task_json_input, uploaded_tasks)
    if ingest_mode != "None" and not tasks and (task_json_input.strip() or uploaded_tasks is not None):
        st.error("Could not parse task JSON. Use a JSON list or {'tasks': [...]} format.")
        return

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{benchmark}_{model.replace('/', '_').replace(' ', '_')}"
    manual_metrics = _build_manual_metrics(
        manual_success_rate,
        manual_tool_accuracy,
        manual_tokens_per_sec,
        manual_energy,
        manual_cost,
        manual_duration,
    )

    new_run: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "split": split,
        "model": model,
        "provider": provider,
        "prompt_version": prompt_version,
        "toolset_version": toolset_version,
        "notes": notes,
        "tags": _normalize_tags(tags),
        "config": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "concurrency": int(concurrency),
        },
        "tasks": tasks,
        "metrics": manual_metrics,
    }

    # If tasks exist, compute derived metrics once and persist for easier external use.
    derived = _compute_run_summary(_normalize_run(new_run))
    for key in [
        "success_rate",
        "tool_accuracy",
        "args_accuracy",
        "hallucination_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "duration_s",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tokens_per_second",
        "energy_joules",
        "energy_per_task",
        "cost_usd",
        "cost_per_success",
        "total_tasks",
        "success_count",
    ]:
        value = derived.get(key)
        if value is not None:
            new_run["metrics"][key] = value

    existing_runs.append(new_run)
    save_eval_runs(eval_json_path, existing_runs)

    st.success(f"Created run `{run_id}` with {len(tasks)} task rows.")
    st.json(new_run)


def _parse_wandb_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    if yaml is None:
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, Exception):
        return {}

    parsed: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        return parsed

    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            parsed[key] = value.get("value")
        else:
            parsed[key] = value
    return parsed


def _extract_scalar_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key, value in summary.items():
        parsed = _to_float(value)
        if parsed is not None:
            scalars[key] = parsed
    return scalars


def _extract_metric_by_keywords(metrics: Dict[str, float], keywords: List[str]) -> Optional[float]:
    lower_map = {key.lower(): value for key, value in metrics.items()}
    for key, value in lower_map.items():
        if all(word in key for word in keywords):
            return value
    return None


@st.cache_data(show_spinner=False)
def load_wandb_runs(base_dir_str: str, include_large: bool) -> pd.DataFrame:
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    run_dirs = sorted(base_dir.glob("run-*"), reverse=True)
    for run_dir in run_dirs:
        files_dir = run_dir / "files"
        if not files_dir.exists():
            continue

        metadata = _load_json_file(files_dir / "wandb-metadata.json")
        summary_path = files_dir / "wandb-summary.json"
        summary: Dict[str, Any] = {}
        summary_skipped = False
        if summary_path.exists():
            if summary_path.stat().st_size > MAX_WANDB_SUMMARY_SIZE_BYTES and not include_large:
                summary_skipped = True
            else:
                parsed = _load_json_file(summary_path)
                if isinstance(parsed, dict):
                    summary = parsed

        config = _parse_wandb_config(files_dir / "config.yaml")
        scalar_metrics = _extract_scalar_metrics(summary)

        started_at = _parse_created_at(
            (metadata or {}).get("startedAt") if isinstance(metadata, dict) else None
        ) or _parse_created_at(run_dir.name)

        row = {
            "run_dir": run_dir.name,
            "run_id": run_dir.name.split("-")[-1],
            "started_at_dt": started_at,
            "started_at": _pretty_timestamp(started_at),
            "project": config.get("wandb_project") or config.get("project") or "",
            "run_name": config.get("wandb_run_name") or config.get("run_name") or "",
            "model": config.get("encoder_model") or config.get("model_name") or "",
            "program": (metadata or {}).get("program") if isinstance(metadata, dict) else "",
            "host": (metadata or {}).get("host") if isinstance(metadata, dict) else "",
            "gpu": (metadata or {}).get("gpu") if isinstance(metadata, dict) else "",
            "gpu_count": (metadata or {}).get("gpu_count") if isinstance(metadata, dict) else "",
            "runtime_sec": _to_float(summary.get("_runtime")),
            "step": _to_float(summary.get("_step")),
            "summary_skipped": summary_skipped,
            "scalar_metrics_count": len(scalar_metrics),
            "loss": _extract_metric_by_keywords(scalar_metrics, ["loss"]),
            "accuracy": _extract_metric_by_keywords(scalar_metrics, ["acc"])
            or _extract_metric_by_keywords(scalar_metrics, ["accuracy"])
            or _extract_metric_by_keywords(scalar_metrics, ["success"]),
            "learning_rate": _extract_metric_by_keywords(scalar_metrics, ["learning", "rate"]),
        }
        row["scalar_metrics"] = scalar_metrics
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(by="started_at_dt", ascending=False, na_position="last")
    return frame


def render_wandb_tab() -> None:
    st.subheader("W&B Runs")
    st.caption("Reads local `wandb/run-*/files` artifacts and summarizes scalar metrics.")

    controls_col1, controls_col2 = st.columns([4, 1])
    wandb_dir = controls_col1.text_input("W&B directory", value=str(DEFAULT_WANDB_DIR))
    include_large = controls_col2.checkbox("Parse large summaries", value=False)

    if st.button("Refresh W&B Data"):
        load_wandb_runs.clear()

    wandb_df = load_wandb_runs(wandb_dir, include_large)
    if wandb_df.empty:
        st.info("No W&B runs found in the selected directory.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    projects = sorted(value for value in wandb_df["project"].dropna().unique().tolist() if str(value).strip())
    hosts = sorted(value for value in wandb_df["host"].dropna().unique().tolist() if str(value).strip())
    project_filter = filter_col1.multiselect("Project", projects, default=projects)
    host_filter = filter_col2.multiselect("Host", hosts, default=hosts)
    run_name_query = filter_col3.text_input("Run name contains", value="")

    filtered = wandb_df.copy()
    if project_filter:
        filtered = filtered[filtered["project"].isin(project_filter)]
    if host_filter:
        filtered = filtered[filtered["host"].isin(host_filter)]
    if run_name_query.strip():
        filtered = filtered[
            filtered["run_name"].fillna("").str.contains(run_name_query.strip(), case=False, regex=False)
        ]

    if filtered.empty:
        st.warning("No W&B runs match current filters.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Runs", str(len(filtered)))
    metric_cols[1].metric("Median Runtime (s)", f"{filtered['runtime_sec'].median():.2f}" if filtered["runtime_sec"].notna().any() else "-")
    metric_cols[2].metric("Distinct Hosts", str(filtered["host"].nunique()))
    metric_cols[3].metric("Large Summaries Skipped", str(int(filtered["summary_skipped"].sum())))

    table_cols = [
        "run_dir",
        "started_at",
        "project",
        "run_name",
        "model",
        "runtime_sec",
        "step",
        "loss",
        "accuracy",
        "learning_rate",
        "host",
        "gpu",
        "gpu_count",
        "scalar_metrics_count",
        "summary_skipped",
    ]
    st.dataframe(filtered[table_cols], use_container_width=True, hide_index=True)

    chart_candidates = filtered.dropna(subset=["runtime_sec", "loss"])
    if not chart_candidates.empty:
        st.plotly_chart(
            px.scatter(
                chart_candidates,
                x="runtime_sec",
                y="loss",
                color="project",
                hover_data=["run_dir", "run_name", "model", "step"],
                title="Runtime vs Loss",
                height=360,
            ),
            use_container_width=True,
        )

    selected_run = st.selectbox("Inspect scalar summary metrics", filtered["run_dir"].tolist())
    selected_row = filtered[filtered["run_dir"] == selected_run].iloc[0]
    st.json(selected_row["scalar_metrics"])


def main() -> None:
    st.title("Agentic Tool-Calling Eval Workbench")

    _ensure_eval_store(DEFAULT_EVAL_JSON)
    eval_mtime = DEFAULT_EVAL_JSON.stat().st_mtime if DEFAULT_EVAL_JSON.exists() else 0.0
    eval_runs = load_eval_runs(str(DEFAULT_EVAL_JSON), eval_mtime)
    run_df, task_df = build_eval_frames(eval_runs)

    tab_eval, tab_create, tab_wandb = st.tabs(
        ["Eval Analytics", "Create Eval Run", "W&B"]
    )
    with tab_eval:
        render_eval_tab(run_df, task_df)
    with tab_create:
        render_create_tab(DEFAULT_EVAL_JSON, eval_runs)
    with tab_wandb:
        render_wandb_tab()


if __name__ == "__main__":
    main()
