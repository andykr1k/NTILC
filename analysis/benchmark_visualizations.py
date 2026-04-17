from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROVIDER_COLORS = {
    "embedding": "#1f77b4",
    "hybrid": "#ff7f0e",
    "huggingface": "#2ca02c",
    "openai": "#d62728",
    "anthropic": "#9467bd",
    "gemini": "#8c564b",
}

MODE_ORDER = ["embedding", "embedding_rerank", "llm_local", "llm_api"]
PROVIDER_ORDER = ["embedding", "hybrid", "huggingface", "openai", "anthropic", "gemini"]
ARCHITECTURE_ORDER = ["normal", "hierarchical"]
LOSS_ORDER = ["prototype_ce", "contrastive", "circle", "functional_margin"]


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette=list(PROVIDER_COLORS.values()),
        font_scale=0.95,
    )
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _humanize_provider(provider: str) -> str:
    mapping = {
        "embedding": "Embedding",
        "hybrid": "Hybrid",
        "huggingface": "Local OSS LLM",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
    }
    return mapping.get(provider, provider.replace("_", " ").title())


def _display_name(model_row: dict[str, Any]) -> str:
    provider = str(model_row.get("provider", "")).strip()
    architecture = str(model_row.get("architecture", "")).strip()
    loss_name = str(model_row.get("loss_name", "")).strip()
    model_name = str(model_row.get("model_name", "")).strip()
    reranker_model = str(model_row.get("reranker_model", "")).strip()

    if provider == "embedding" and architecture and loss_name:
        return f"{architecture}/{loss_name}"
    if provider == "hybrid" and architecture and loss_name and reranker_model:
        return f"{architecture}/{loss_name} + {reranker_model}"
    return model_name or str(model_row.get("adapter_id", "")).strip()


def _variant_key(model_row: dict[str, Any]) -> str:
    architecture = str(model_row.get("architecture", "")).strip()
    loss_name = str(model_row.get("loss_name", "")).strip()
    if architecture and loss_name:
        return f"{architecture}/{loss_name}"
    return ""


def load_benchmark_frames(
    summary_path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if results_dir is None:
        results_dir = summary_path.parent / "results"
    results_dir = Path(results_dir)

    model_rows: list[dict[str, Any]] = []
    results_rows: list[dict[str, Any]] = []

    for model in summary.get("models", []):
        metrics = model.get("metrics") or {}
        metadata = model.get("metadata") or {}
        result_row = {
            "adapter_id": model.get("adapter_id"),
            "provider": model.get("provider"),
            "mode": model.get("mode"),
            "model_name": model.get("model_name"),
            "status": model.get("status"),
            "error_message": model.get("error_message"),
            "results_path": model.get("results_path"),
            "architecture": metadata.get("architecture"),
            "loss_name": metadata.get("loss_name"),
            "reranker_model": metadata.get("reranker_model"),
            "embedding_top_k": metadata.get("embedding_top_k"),
            "checkpoint_path": metadata.get("checkpoint_path"),
            "top_1_accuracy": metrics.get("top_1_accuracy"),
            "top_3_accuracy": metrics.get("top_3_accuracy"),
            "top_5_accuracy": metrics.get("top_5_accuracy"),
            "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
            "mean_latency_ms": metrics.get("mean_latency_ms"),
            "p50_latency_ms": metrics.get("p50_latency_ms"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "mean_input_tokens": metrics.get("mean_input_tokens"),
            "mean_output_tokens": metrics.get("mean_output_tokens"),
            "mean_total_tokens": metrics.get("mean_total_tokens"),
            "sum_input_tokens": metrics.get("sum_input_tokens"),
            "sum_output_tokens": metrics.get("sum_output_tokens"),
            "sum_total_tokens": metrics.get("sum_total_tokens"),
            "mean_cost_usd": metrics.get("mean_cost_usd"),
            "sum_cost_usd": metrics.get("sum_cost_usd"),
            "total_examples": metrics.get("total_examples"),
            "successful_examples": metrics.get("successful_examples"),
            "error_examples": metrics.get("error_examples"),
            "per_tool_accuracy": metrics.get("per_tool_accuracy"),
        }
        result_row["provider_label"] = _humanize_provider(str(result_row.get("provider", "")))
        result_row["display_name"] = _display_name(result_row)
        result_row["variant_key"] = _variant_key(result_row)
        total_examples = _safe_float(result_row.get("total_examples")) or 0.0
        error_examples = _safe_float(result_row.get("error_examples")) or 0.0
        result_row["error_rate"] = (error_examples / total_examples) if total_examples else 0.0
        model_rows.append(result_row)

        result_path_value = result_row.get("results_path")
        result_path = Path(result_path_value) if result_path_value else results_dir / f"{result_row['adapter_id'].replace('/', '-')}.jsonl"
        for row in _load_jsonl(result_path):
            merged = dict(row)
            merged.update(
                {
                    "display_name": result_row["display_name"],
                    "provider_label": result_row["provider_label"],
                    "architecture": result_row["architecture"],
                    "loss_name": result_row["loss_name"],
                    "reranker_model": result_row["reranker_model"],
                    "variant_key": result_row["variant_key"],
                }
            )
            results_rows.append(merged)

    models_df = pd.DataFrame(model_rows)
    results_df = pd.DataFrame(results_rows)

    if not models_df.empty:
        for column in (
            "top_1_accuracy",
            "top_3_accuracy",
            "top_5_accuracy",
            "mean_reciprocal_rank",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "mean_input_tokens",
            "mean_output_tokens",
            "mean_total_tokens",
            "sum_input_tokens",
            "sum_output_tokens",
            "sum_total_tokens",
            "mean_cost_usd",
            "sum_cost_usd",
            "total_examples",
            "successful_examples",
            "error_examples",
            "error_rate",
        ):
            if column in models_df:
                models_df[column] = pd.to_numeric(models_df[column], errors="coerce")

    if not results_df.empty:
        for column in (
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "reciprocal_rank",
        ):
            if column in results_df:
                results_df[column] = pd.to_numeric(results_df[column], errors="coerce")
        for column in ("correct_top1", "top_3_hit", "top_5_hit"):
            if column in results_df:
                results_df[column] = results_df[column].astype("boolean")

    return {
        "summary": summary,
        "models": models_df,
        "results": results_df,
        "successful_models": models_df.loc[models_df["status"] == "ok"].copy() if not models_df.empty else models_df,
        "successful_results": results_df.loc[results_df["status"] == "ok"].copy() if not results_df.empty else results_df,
    }


def save_figure(fig: plt.Figure, output_dir: str | Path, name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def top_model_table(models_df: pd.DataFrame, metric: str = "top_1_accuracy", top_n: int = 10) -> pd.DataFrame:
    columns = [
        "display_name",
        "provider_label",
        "mode",
        "top_1_accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "mean_reciprocal_rank",
        "mean_latency_ms",
        "mean_total_tokens",
        "sum_cost_usd",
    ]
    available_columns = [column for column in columns if column in models_df.columns]
    return (
        models_df.loc[models_df["status"] == "ok", available_columns]
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_top_metric_bars(
    models_df: pd.DataFrame,
    *,
    metric: str = "top_1_accuracy",
    top_n: int = 15,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    subset = (
        models_df.loc[models_df["status"] == "ok", ["display_name", "provider", metric]]
        .dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(top_n)
        .sort_values(metric, ascending=True)
    )
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(subset))))
    sns.barplot(
        data=subset,
        x=metric,
        y="display_name",
        hue="provider",
        dodge=False,
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_title(title or f"Top {top_n} Models by {metric.replace('_', ' ').title()}")
    ax.legend(title="Provider", loc="lower right")
    return fig, ax


def plot_metric_triptych(models_df: pd.DataFrame, top_n: int = 12) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    metrics = ["top_1_accuracy", "top_3_accuracy", "mean_reciprocal_rank"]
    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, 0.38 * top_n)), sharey=True)
    ranking = (
        models_df.loc[models_df["status"] == "ok", ["display_name", "provider", "top_1_accuracy"]]
        .dropna(subset=["top_1_accuracy"])
        .sort_values("top_1_accuracy", ascending=False)
        .head(top_n)["display_name"]
        .tolist()
    )
    subset = models_df.loc[models_df["display_name"].isin(ranking)].copy()
    for axis, metric in zip(axes, metrics, strict=True):
        metric_subset = subset[["display_name", "provider", metric]].dropna(subset=[metric]).copy()
        metric_subset["display_name"] = pd.Categorical(metric_subset["display_name"], categories=ranking[::-1], ordered=True)
        metric_subset = metric_subset.sort_values("display_name")
        sns.barplot(
            data=metric_subset,
            x=metric,
            y="display_name",
            hue="provider",
            dodge=False,
            palette=PROVIDER_COLORS,
            ax=axis,
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)))
    fig.suptitle("Leaderboards Across Core Benchmark Metrics", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig, axes


def plot_accuracy_latency_tradeoff(models_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.loc[models_df["status"] == "ok"].dropna(subset=["top_1_accuracy", "mean_latency_ms"]).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=subset,
        x="mean_latency_ms",
        y="top_1_accuracy",
        hue="provider",
        style="mode",
        s=180,
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    for _, row in subset.iterrows():
        ax.annotate(
            row["display_name"],
            (row["mean_latency_ms"], row["top_1_accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Latency (ms, log scale)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy-Latency Tradeoff")
    return fig, ax


def plot_accuracy_token_tradeoff(models_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.loc[models_df["status"] == "ok"].dropna(subset=["top_1_accuracy", "mean_total_tokens"]).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=subset,
        x="mean_total_tokens",
        y="top_1_accuracy",
        hue="provider",
        style="mode",
        s=180,
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    for _, row in subset.iterrows():
        ax.annotate(
            row["display_name"],
            (row["mean_total_tokens"], row["top_1_accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Total Tokens (log scale)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy-Token Tradeoff")
    return fig, ax


def plot_cost_accuracy_tradeoff(models_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.loc[models_df["status"] == "ok"].dropna(subset=["top_1_accuracy", "sum_cost_usd"]).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=subset,
        x="sum_cost_usd",
        y="top_1_accuracy",
        hue="provider",
        style="mode",
        s=180,
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    for _, row in subset.iterrows():
        ax.annotate(
            row["display_name"],
            (row["sum_cost_usd"], row["top_1_accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.85,
        )
    ax.set_xlabel("Total Benchmark Cost (USD)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Accuracy-Cost Tradeoff")
    return fig, ax


def plot_metric_by_provider(models_df: pd.DataFrame, metric: str = "top_1_accuracy") -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.loc[models_df["status"] == "ok"].dropna(subset=[metric]).copy()
    subset["provider_label"] = pd.Categorical(
        subset["provider_label"],
        categories=[_humanize_provider(provider) for provider in PROVIDER_ORDER],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=subset,
        x="provider_label",
        y=metric,
        hue="provider",
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="provider_label",
        y=metric,
        hue="provider",
        palette=PROVIDER_COLORS,
        dodge=False,
        linewidth=0.5,
        edgecolor="white",
        size=8,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(PROVIDER_COLORS)], labels[: len(PROVIDER_COLORS)], title="Provider", loc="best")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Provider")
    ax.tick_params(axis="x", rotation=20)
    return fig, ax


def plot_heatmap_by_variant(
    models_df: pd.DataFrame,
    *,
    mode: str,
    metric: str = "top_1_accuracy",
) -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.loc[(models_df["status"] == "ok") & (models_df["mode"] == mode)].copy()
    if subset.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.set_title(f"No rows available for mode={mode}")
        return fig, ax

    subset["architecture"] = pd.Categorical(subset["architecture"], categories=ARCHITECTURE_ORDER, ordered=True)
    subset["loss_name"] = pd.Categorical(subset["loss_name"], categories=LOSS_ORDER, ordered=True)
    matrix = subset.pivot_table(
        index="architecture",
        columns="loss_name",
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    matrix = matrix.astype(float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Architecture")
    ax.set_title(f"{metric.replace('_', ' ').title()} Heatmap for {mode}")
    return fig, ax


def compute_hybrid_lift(models_df: pd.DataFrame, metric: str = "top_1_accuracy") -> pd.DataFrame:
    embedding_df = (
        models_df.loc[(models_df["status"] == "ok") & (models_df["provider"] == "embedding"), ["variant_key", metric, "display_name"]]
        .dropna(subset=[metric])
        .rename(columns={metric: f"embedding_{metric}", "display_name": "embedding_name"})
    )
    hybrid_df = (
        models_df.loc[(models_df["status"] == "ok") & (models_df["provider"] == "hybrid"), ["variant_key", metric, "display_name", "reranker_model"]]
        .dropna(subset=[metric])
        .rename(columns={metric: f"hybrid_{metric}", "display_name": "hybrid_name"})
    )
    merged = hybrid_df.merge(embedding_df, on="variant_key", how="inner")
    merged["absolute_lift"] = merged[f"hybrid_{metric}"] - merged[f"embedding_{metric}"]
    return merged.sort_values("absolute_lift", ascending=False).reset_index(drop=True)


def plot_hybrid_lift(models_df: pd.DataFrame, metric: str = "top_1_accuracy") -> tuple[plt.Figure, plt.Axes]:
    lift_df = compute_hybrid_lift(models_df, metric=metric)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.7 * len(lift_df))))
    sns.barplot(
        data=lift_df,
        x="absolute_lift",
        y="variant_key",
        color=PROVIDER_COLORS["hybrid"],
        ax=ax,
    )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(f"Hybrid - Embedding {metric.replace('_', ' ').title()}")
    ax.set_ylabel("Embedding Variant")
    ax.set_title(f"Hybrid Lift over Base Embedding for {metric.replace('_', ' ').title()}")
    return fig, ax


def hardest_tools_table(results_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    subset = results_df.loc[results_df["status"] == "ok"].copy()
    grouped = (
        subset.groupby("expected_tool")
        .agg(
            mean_accuracy=("correct_top1", lambda values: pd.Series(values).astype(float).mean()),
            model_count=("adapter_id", "nunique"),
            example_count=("example_id", "count"),
        )
        .sort_values("mean_accuracy", ascending=True)
        .head(top_n)
        .reset_index()
    )
    return grouped


def plot_hardest_tools(results_df: pd.DataFrame, top_n: int = 15) -> tuple[plt.Figure, plt.Axes]:
    table = hardest_tools_table(results_df, top_n=top_n)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(table))))
    sns.barplot(data=table, x="mean_accuracy", y="expected_tool", color="#c44e52", ax=ax)
    ax.set_xlabel("Mean Top-1 Accuracy Across Models")
    ax.set_ylabel("Tool")
    ax.set_title("Hardest Tools in the Benchmark")
    return fig, ax


def plot_per_tool_heatmap(
    results_df: pd.DataFrame,
    models_df: pd.DataFrame,
    *,
    top_n_models: int = 10,
) -> tuple[plt.Figure, plt.Axes]:
    top_models = (
        models_df.loc[models_df["status"] == "ok", ["adapter_id", "display_name", "top_1_accuracy"]]
        .dropna(subset=["top_1_accuracy"])
        .sort_values("top_1_accuracy", ascending=False)
        .head(top_n_models)
    )
    subset = results_df.loc[
        (results_df["status"] == "ok") & (results_df["adapter_id"].isin(top_models["adapter_id"]))
    ].copy()
    matrix = (
        subset.assign(correct_top1_float=subset["correct_top1"].astype(float))
        .groupby(["expected_tool", "display_name"])["correct_top1_float"]
        .mean()
        .reset_index()
        .pivot(index="expected_tool", columns="display_name", values="correct_top1_float")
    )
    matrix = matrix.astype(float)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index, top_models["display_name"]]
    fig, ax = plt.subplots(figsize=(1.4 * max(6, len(matrix.columns)), max(8, 0.35 * len(matrix.index))))
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, linewidths=0.3, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Tool")
    ax.set_title("Per-Tool Accuracy Heatmap for Top Models")
    return fig, ax


def plot_latency_distribution(
    results_df: pd.DataFrame,
    models_df: pd.DataFrame,
    *,
    top_n_models: int = 8,
) -> tuple[plt.Figure, plt.Axes]:
    top_models = (
        models_df.loc[models_df["status"] == "ok", ["adapter_id", "display_name", "top_1_accuracy"]]
        .dropna(subset=["top_1_accuracy"])
        .sort_values("top_1_accuracy", ascending=False)
        .head(top_n_models)
    )
    subset = results_df.loc[
        (results_df["status"] == "ok") & (results_df["adapter_id"].isin(top_models["adapter_id"]))
    ].copy()
    subset["display_name"] = pd.Categorical(subset["display_name"], categories=top_models["display_name"][::-1], ordered=True)
    fig, ax = plt.subplots(figsize=(13, max(6, 0.6 * len(top_models))))
    sns.violinplot(
        data=subset,
        x="latency_ms",
        y="display_name",
        hue="provider",
        palette=PROVIDER_COLORS,
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Per-Query Latency (ms, log scale)")
    ax.set_ylabel("Model")
    ax.set_title("Latency Distribution for Top Models")
    ax.legend(title="Provider", loc="lower right")
    return fig, ax


def plot_confusion_matrix(
    results_df: pd.DataFrame,
    *,
    adapter_id: str,
    top_n_tools: int = 12,
) -> tuple[plt.Figure, plt.Axes]:
    subset = results_df.loc[
        (results_df["status"] == "ok") & (results_df["adapter_id"] == adapter_id)
    ].copy()
    if subset.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.set_title(f"No rows available for adapter_id={adapter_id}")
        return fig, ax

    tool_counts = subset["expected_tool"].value_counts().head(top_n_tools).index.tolist()
    subset = subset.loc[
        subset["expected_tool"].isin(tool_counts) & subset["selected_tool"].isin(tool_counts)
    ]
    matrix = pd.crosstab(subset["expected_tool"], subset["selected_tool"])
    matrix = matrix.reindex(index=tool_counts, columns=tool_counts, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="Reds", annot=True, fmt="d", linewidths=0.4, ax=ax)
    title_name = subset["display_name"].iloc[0]
    ax.set_xlabel("Predicted Tool")
    ax.set_ylabel("Gold Tool")
    ax.set_title(f"Confusion Matrix for {title_name}")
    return fig, ax


def plot_common_confusions(
    results_df: pd.DataFrame,
    *,
    adapter_id: str,
    top_n: int = 15,
) -> tuple[plt.Figure, plt.Axes]:
    subset = results_df.loc[
        (results_df["status"] == "ok")
        & (results_df["adapter_id"] == adapter_id)
        & (results_df["expected_tool"] != results_df["selected_tool"])
    ].copy()
    subset["confusion_pair"] = subset["expected_tool"] + " -> " + subset["selected_tool"]
    table = subset["confusion_pair"].value_counts().head(top_n).reset_index()
    table.columns = ["confusion_pair", "count"]
    fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * len(table))))
    sns.barplot(data=table.sort_values("count"), x="count", y="confusion_pair", color="#8172b2", ax=ax)
    title_name = subset["display_name"].iloc[0] if not subset.empty else adapter_id
    ax.set_xlabel("Count")
    ax.set_ylabel("Confusion Pair")
    ax.set_title(f"Most Common Confusions for {title_name}")
    return fig, ax


def plot_error_rates(models_df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    subset = models_df.copy().sort_values("error_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(subset))))
    sns.barplot(
        data=subset,
        x="error_rate",
        y="display_name",
        hue="provider",
        dodge=False,
        palette=PROVIDER_COLORS,
        ax=ax,
    )
    ax.set_xlabel("Error Rate")
    ax.set_ylabel("Model")
    ax.set_title("Benchmark Failure Rate by Model")
    ax.legend(title="Provider", loc="lower right")
    return fig, ax


def best_model_ids(models_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    return (
        models_df.loc[models_df["status"] == "ok", ["adapter_id", "top_1_accuracy"]]
        .dropna(subset=["top_1_accuracy"])
        .sort_values("top_1_accuracy", ascending=False)
        .head(top_n)["adapter_id"]
        .tolist()
    )


__all__ = [
    "PROVIDER_COLORS",
    "best_model_ids",
    "compute_hybrid_lift",
    "hardest_tools_table",
    "load_benchmark_frames",
    "plot_accuracy_latency_tradeoff",
    "plot_accuracy_token_tradeoff",
    "plot_common_confusions",
    "plot_confusion_matrix",
    "plot_cost_accuracy_tradeoff",
    "plot_error_rates",
    "plot_hardest_tools",
    "plot_heatmap_by_variant",
    "plot_hybrid_lift",
    "plot_latency_distribution",
    "plot_metric_by_provider",
    "plot_metric_triptych",
    "plot_per_tool_heatmap",
    "plot_top_metric_bars",
    "save_figure",
    "set_plot_style",
    "top_model_table",
]
