from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name. Falls back to WANDB_PROJECT.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity name. Falls back to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group name. Falls back to WANDB_GROUP.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name. Falls back to WANDB_NAME.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated Weights & Biases tags. Falls back to WANDB_TAGS.",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="Weights & Biases notes. Falls back to WANDB_NOTES.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        help="Weights & Biases mode such as online, offline, or disabled. Falls back to WANDB_MODE.",
    )


def add_diagnostic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--diagnostic-max-samples",
        type=int,
        default=2000,
        help="Maximum number of validation rows used for expensive diagnostics per epoch.",
    )
    parser.add_argument(
        "--diagnostic-top-k-tools",
        type=int,
        default=100,
        help="Maximum number of labels shown in projection legends.",
    )
    parser.add_argument(
        "--diagnostic-overlap-margin",
        type=float,
        default=0.05,
        help="Cosine-margin threshold used for near-overlap diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-min-tool-samples",
        type=int,
        default=5,
        help="Minimum per-label sample count required for pair-overlap reporting.",
    )
    parser.add_argument(
        "--diagnostic-heatmap-tools",
        type=int,
        default=20,
        help="Maximum number of labels shown on overlap and confusion heatmaps.",
    )
    parser.add_argument(
        "--diagnostic-top-overlap-rows",
        type=int,
        default=15,
        help="Maximum number of top overlap rows surfaced in summaries and preview tables.",
    )


def default_run_date() -> str:
    return datetime.now().astimezone().date().isoformat()


def default_wandb_run_name(
    embedding_type: str,
    loss_type: str,
    run_date: str | None = None,
) -> str:
    return f"{embedding_type}-{loss_type}-{run_date or default_run_date()}"


def _env_or_default(value: str | None, env_name: str) -> str | None:
    if value is not None:
        stripped = str(value).strip()
        return stripped or None
    env_value = os.getenv(env_name)
    if env_value is None:
        return None
    stripped = env_value.strip()
    return stripped or None


def _split_tags(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tag.strip() for tag in str(raw).split(",") if tag.strip()]


def init_wandb_run(
    *,
    enabled: bool,
    args: argparse.Namespace,
    embedding_type: str,
    loss_type: str,
    config: Dict[str, Any],
):
    if not enabled:
        return None, None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging was requested, but the wandb package is not installed."
        ) from exc

    run_date = default_run_date()
    run_name = _env_or_default(args.wandb_run_name, "WANDB_NAME")
    if run_name is None:
        run_name = default_wandb_run_name(
            embedding_type=embedding_type,
            loss_type=loss_type,
            run_date=run_date,
        )

    project = _env_or_default(args.wandb_project, "WANDB_PROJECT")
    entity = _env_or_default(args.wandb_entity, "WANDB_ENTITY")
    group = _env_or_default(args.wandb_group, "WANDB_GROUP")
    notes = _env_or_default(args.wandb_notes, "WANDB_NOTES")
    mode = _env_or_default(args.wandb_mode, "WANDB_MODE")
    raw_tags = _env_or_default(args.wandb_tags, "WANDB_TAGS")

    wandb_config = dict(config)
    wandb_config.update(
        {
            "embedding_type": embedding_type,
            "loss_type": loss_type,
            "run_date": run_date,
        }
    )

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        tags=_split_tags(raw_tags),
        notes=notes,
        mode=mode,
        config=wandb_config,
    )
    if run is not None:
        run.summary["embedding_type"] = embedding_type
        run.summary["loss_type"] = loss_type
        run.summary["run_date"] = run_date
    return wandb, run


def log_output_artifact(
    *,
    wandb_module,
    run,
    output_dir: Path,
    artifact_type: str = "model",
) -> None:
    if run is None:
        return

    artifact = wandb_module.Artifact(
        name=f"{run.id}-checkpoints",
        type=artifact_type,
        metadata={"output_dir": str(output_dir)},
    )
    for filename in ("best.pt", "last.pt", "metrics.json"):
        path = output_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=filename)
    run.log_artifact(artifact, aliases=["latest", "final"])


def select_diagnostic_indices(
    row_count: int,
    max_samples: int,
    seed: int,
) -> list[int]:
    if row_count <= 0:
        return []
    if max_samples <= 0 or row_count <= max_samples:
        return list(range(row_count))

    rng = random.Random(seed)
    indices = rng.sample(range(row_count), max_samples)
    indices.sort()
    return indices


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _to_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    return np.asarray(matrix, dtype=np.float32)


def _project_with_padding(values: np.ndarray, components: int, seed: int) -> np.ndarray:
    row_count = values.shape[0]
    if row_count == 0:
        return np.zeros((0, components), dtype=np.float32)
    if row_count == 1:
        return np.zeros((1, components), dtype=np.float32)

    max_components = min(components, values.shape[0], values.shape[1])
    if max_components <= 0:
        return np.zeros((row_count, components), dtype=np.float32)

    projected = PCA(n_components=max_components, random_state=seed).fit_transform(values)
    if max_components == components:
        return projected.astype(np.float32, copy=False)

    padded = np.zeros((row_count, components), dtype=np.float32)
    padded[:, :max_components] = projected
    return padded


def _umap_projection(values: np.ndarray, components: int, seed: int) -> np.ndarray:
    row_count = values.shape[0]
    if row_count == 0:
        return np.zeros((0, components), dtype=np.float32)
    if row_count <= components:
        return _project_with_padding(values, components=components, seed=seed)

    try:
        n_neighbors = min(15, row_count - 1)
        if n_neighbors < 2:
            return _project_with_padding(values, components=components, seed=seed)
        projected = UMAP(
            n_components=components,
            random_state=seed,
            n_neighbors=n_neighbors,
        ).fit_transform(values)
        return projected.astype(np.float32, copy=False)
    except Exception:
        return _project_with_padding(values, components=components, seed=seed)


def _to_plotly_rgba(color: Sequence[float], alpha: float = 0.7) -> str:
    red, green, blue = np.round(np.asarray(color[:3]) * 255).astype(int)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def plot_projection_2d(
    values: np.ndarray,
    row_labels: Sequence[str],
    title: str,
    *,
    top_k_labels: int,
    x_label: str,
    y_label: str,
):
    counts = Counter(row_labels)
    top_labels = [label for label, _ in counts.most_common(top_k_labels)]
    label_indices = {
        label: np.flatnonzero(np.asarray(row_labels) == label)
        for label in top_labels
    }
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(top_labels))))

    fig, ax = plt.subplots(figsize=(10, 8))
    for color, label in zip(colors, top_labels):
        idx = label_indices[label]
        ax.scatter(
            values[idx, 0],
            values[idx, 1],
            s=14,
            alpha=0.7,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if top_labels:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_projection_3d(
    values: np.ndarray,
    row_labels: Sequence[str],
    title: str,
    *,
    top_k_labels: int,
    labels: tuple[str, str, str],
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Install plotly to use the interactive 3D projections."
        ) from exc

    counts = Counter(row_labels)
    top_labels = [label for label, _ in counts.most_common(top_k_labels)]
    label_indices = {
        label: np.flatnonzero(np.asarray(row_labels) == label)
        for label in top_labels
    }
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(top_labels))))

    figure = go.Figure()
    for color, label in zip(colors, top_labels):
        idx = label_indices[label]
        figure.add_trace(
            go.Scatter3d(
                x=values[idx, 0],
                y=values[idx, 1],
                z=values[idx, 2],
                mode="markers",
                name=label,
                marker={"size": 3, "color": _to_plotly_rgba(color, alpha=0.75)},
            )
        )
    figure.update_layout(
        title=title,
        legend={"itemsizing": "constant"},
        scene={
            "xaxis_title": labels[0],
            "yaxis_title": labels[1],
            "zaxis_title": labels[2],
            "aspectmode": "data",
        },
    )
    return figure


def compute_overlap_tables(
    *,
    rows: Sequence[Dict[str, str]],
    row_label_names: Sequence[str],
    scores: np.ndarray,
    label_indices: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
    normalized_centroids: np.ndarray,
    overlap_margin: float,
    min_label_samples: int,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    true_scores = scores[np.arange(len(label_indices)), label_indices]
    competing_scores = scores.copy()
    competing_scores[np.arange(len(label_indices)), label_indices] = -np.inf
    closest_rival_idx = competing_scores.argmax(axis=1)
    closest_rival_scores = competing_scores[np.arange(len(label_indices)), closest_rival_idx]
    margins = true_scores - closest_rival_scores

    sample_df = pd.DataFrame(
        {
            "query": [row.get("query", row.get("text", "")) for row in rows],
            label_column: list(row_label_names),
            "predicted_label": [label_names[idx] for idx in predictions],
            "closest_rival_label": [label_names[idx] for idx in closest_rival_idx],
            "true_score": true_scores,
            "closest_rival_score": closest_rival_scores,
            "margin": margins,
            "misclassified": predictions != label_indices,
            "near_overlap": margins <= overlap_margin,
        }
    )

    label_count = len(label_names)
    overlap_matrix = np.full((label_count, label_count), np.nan, dtype=np.float32)
    confusion_matrix = np.full((label_count, label_count), np.nan, dtype=np.float32)
    label_counts = np.bincount(label_indices, minlength=label_count)
    label_rows: list[Dict[str, Any]] = []

    for index, label_name in enumerate(label_names):
        mask = label_indices == index
        count = int(mask.sum())
        if count == 0:
            continue

        label_scores = scores[mask]
        self_scores = label_scores[:, index]
        overlap_matrix[index, :] = (
            label_scores >= (self_scores[:, None] - overlap_margin)
        ).mean(axis=0)
        overlap_matrix[index, index] = np.nan

        prediction_counts = np.bincount(predictions[mask], minlength=label_count)
        confusion_matrix[index, :] = prediction_counts / count
        confusion_matrix[index, index] = np.nan

        label_samples = sample_df.loc[mask]
        rival_counts = label_samples["closest_rival_label"].value_counts()
        label_rows.append(
            {
                "label": label_name,
                "count": count,
                "accuracy": 1.0 - float(label_samples["misclassified"].mean()),
                "near_overlap_rate": float(label_samples["near_overlap"].mean()),
                "mean_margin": float(label_samples["margin"].mean()),
                "p10_margin": float(label_samples["margin"].quantile(0.10)),
                "mean_true_score": float(label_samples["true_score"].mean()),
                "mean_rival_score": float(label_samples["closest_rival_score"].mean()),
                "most_common_rival": rival_counts.index[0],
                "most_common_rival_share": float(rival_counts.iloc[0] / count),
            }
        )

    pair_rows: list[Dict[str, Any]] = []
    for first_index, first_name in enumerate(label_names):
        for second_index in range(first_index + 1, len(label_names)):
            second_name = label_names[second_index]
            if (
                label_counts[first_index] < min_label_samples
                or label_counts[second_index] < min_label_samples
            ):
                continue
            pair_rows.append(
                {
                    "label_a": first_name,
                    "label_b": second_name,
                    "samples_a": int(label_counts[first_index]),
                    "samples_b": int(label_counts[second_index]),
                    "a_to_b_overlap": float(overlap_matrix[first_index, second_index]),
                    "b_to_a_overlap": float(overlap_matrix[second_index, first_index]),
                    "mutual_overlap": float(
                        np.nanmean(
                            [
                                overlap_matrix[first_index, second_index],
                                overlap_matrix[second_index, first_index],
                            ]
                        )
                    ),
                    "a_to_b_confusion": float(confusion_matrix[first_index, second_index]),
                    "b_to_a_confusion": float(confusion_matrix[second_index, first_index]),
                    "centroid_cosine": float(
                        normalized_centroids[first_index] @ normalized_centroids[second_index]
                    ),
                }
            )

    label_df = pd.DataFrame(
        label_rows,
        columns=[
            "label",
            "count",
            "accuracy",
            "near_overlap_rate",
            "mean_margin",
            "p10_margin",
            "mean_true_score",
            "mean_rival_score",
            "most_common_rival",
            "most_common_rival_share",
        ],
    )
    if not label_df.empty:
        label_df = label_df.sort_values(
            ["near_overlap_rate", "mean_margin"],
            ascending=[False, True],
        ).reset_index(drop=True)

    pair_df = pd.DataFrame(
        pair_rows,
        columns=[
            "label_a",
            "label_b",
            "samples_a",
            "samples_b",
            "a_to_b_overlap",
            "b_to_a_overlap",
            "mutual_overlap",
            "a_to_b_confusion",
            "b_to_a_confusion",
            "centroid_cosine",
        ],
    )
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            ["mutual_overlap", "centroid_cosine"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return sample_df, label_df, pair_df, overlap_matrix, confusion_matrix


def _plot_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    colorbar_label: str,
    *,
    cmap: str,
):
    plot_values = np.nan_to_num(matrix, nan=0.0)
    vmax = float(plot_values.max()) if plot_values.size else 0.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * len(labels)), max(8, 0.45 * len(labels)))
    )
    image = ax.imshow(plot_values, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def compute_embedding_diagnostics(
    *,
    rows: Sequence[Dict[str, str]],
    embeddings: Any,
    label_names: Sequence[str],
    row_label_names: Sequence[str],
    centroids: Any,
    seed: int,
    top_k_labels: int,
    overlap_margin: float,
    min_label_samples: int,
    heatmap_labels: int,
    label_column: str,
    label_namespace: str,
) -> Dict[str, Any]:
    embedding_values = _to_numpy(embeddings)
    centroid_values = normalize_matrix(_to_numpy(centroids))
    if embedding_values.shape[0] == 0 or not rows:
        return {
            "scalars": {
                f"{label_namespace}/diagnostic_sample_size": 0,
                f"{label_namespace}/label_count": 0,
            },
            "figures": {},
            "plotly": {},
            "tables": {},
            "summary": {},
        }

    label_to_index = {name: index for index, name in enumerate(label_names)}
    labels = np.asarray([label_to_index[name] for name in row_label_names], dtype=np.int64)
    normalized_embeddings = normalize_matrix(embedding_values)
    scores = normalized_embeddings @ centroid_values.T
    predictions = scores.argmax(axis=1)
    subset_accuracy = float((predictions == labels).mean()) if len(labels) else float("nan")

    unique_labels = np.unique(labels)
    silhouette = float("nan")
    if len(labels) > 1 and len(unique_labels) > 1:
        silhouette = float(silhouette_score(embedding_values, labels, metric="cosine"))

    sample_df, label_df, pair_df, overlap_matrix, confusion_matrix = compute_overlap_tables(
        rows=rows,
        row_label_names=row_label_names,
        scores=scores,
        label_indices=labels,
        predictions=predictions,
        label_names=label_names,
        normalized_centroids=centroid_values,
        overlap_margin=overlap_margin,
        min_label_samples=min_label_samples,
        label_column=label_column,
    )

    figures: Dict[str, Any] = {}
    plotly_figures: Dict[str, Any] = {}
    pca_2d = _project_with_padding(embedding_values, components=2, seed=seed)
    pca_3d = _project_with_padding(embedding_values, components=3, seed=seed)
    umap_2d = _umap_projection(embedding_values, components=2, seed=seed)
    umap_3d = _umap_projection(embedding_values, components=3, seed=seed)

    figures[f"{label_namespace}/pca_2d"] = plot_projection_2d(
        pca_2d,
        row_label_names,
        f"PCA of {label_column} validation embeddings",
        top_k_labels=top_k_labels,
        x_label="PC1",
        y_label="PC2",
    )
    figures[f"{label_namespace}/umap_2d"] = plot_projection_2d(
        umap_2d,
        row_label_names,
        f"UMAP of {label_column} validation embeddings",
        top_k_labels=top_k_labels,
        x_label="UMAP1",
        y_label="UMAP2",
    )
    plotly_figures[f"{label_namespace}/pca_3d"] = plot_projection_3d(
        pca_3d,
        row_labels=row_label_names,
        title=f"3D PCA of {label_column} validation embeddings",
        top_k_labels=top_k_labels,
        labels=("PC1", "PC2", "PC3"),
    )
    plotly_figures[f"{label_namespace}/umap_3d"] = plot_projection_3d(
        umap_3d,
        row_labels=row_label_names,
        title=f"3D UMAP of {label_column} validation embeddings",
        top_k_labels=top_k_labels,
        labels=("UMAP1", "UMAP2", "UMAP3"),
    )

    count_by_label = Counter(row_label_names)
    heatmap_label_names = [name for name, _ in count_by_label.most_common(heatmap_labels)]
    heatmap_indices = [label_to_index[name] for name in heatmap_label_names]
    if heatmap_indices:
        figures[f"{label_namespace}/overlap_heatmap"] = _plot_heatmap(
            overlap_matrix[np.ix_(heatmap_indices, heatmap_indices)],
            heatmap_label_names,
            f"Near-overlap rates by {label_column} (margin <= {overlap_margin:.2f})",
            "share of samples near a competing centroid",
            cmap="magma",
        )
        figures[f"{label_namespace}/confusion_heatmap"] = _plot_heatmap(
            confusion_matrix[np.ix_(heatmap_indices, heatmap_indices)],
            heatmap_label_names,
            f"Cross-{label_column} confusion rates",
            "share of samples predicted as another label",
            cmap="viridis",
        )

    tables: Dict[str, pd.DataFrame] = {
        f"{label_namespace}/samples_table": sample_df.reset_index(drop=True),
        f"{label_namespace}/overlap_by_label_table": label_df.reset_index(drop=True),
        f"{label_namespace}/overlap_pairs_table": pair_df.reset_index(drop=True),
    }

    summary: Dict[str, Any] = {
        f"{label_namespace}/diagnostic_sample_size": int(len(rows)),
        f"{label_namespace}/label_count": int(len(unique_labels)),
        f"{label_namespace}/subset_retrieval_accuracy": subset_accuracy,
        f"{label_namespace}/silhouette_score": silhouette,
        f"{label_namespace}/global_near_overlap_rate": float(sample_df["near_overlap"].mean()),
    }

    if not pair_df.empty:
        top_pair = pair_df.iloc[0]
        top_pair_examples = (
            sample_df.loc[
                (
                    (sample_df[label_column] == top_pair["label_a"])
                    & (sample_df["closest_rival_label"] == top_pair["label_b"])
                )
                | (
                    (sample_df[label_column] == top_pair["label_b"])
                    & (sample_df["closest_rival_label"] == top_pair["label_a"])
                ),
                [
                    label_column,
                    "predicted_label",
                    "closest_rival_label",
                    "true_score",
                    "closest_rival_score",
                    "margin",
                    "query",
                ],
            ]
            .sort_values(["margin", "true_score"], ascending=[True, False])
            .head(10)
            .reset_index(drop=True)
        )
        tables[f"{label_namespace}/top_pair_examples_table"] = top_pair_examples
        summary.update(
            {
                f"{label_namespace}/top_pair/label_a": str(top_pair["label_a"]),
                f"{label_namespace}/top_pair/label_b": str(top_pair["label_b"]),
                f"{label_namespace}/top_pair/mutual_overlap": float(top_pair["mutual_overlap"]),
                f"{label_namespace}/top_pair/centroid_cosine": float(top_pair["centroid_cosine"]),
            }
        )

    return {
        "scalars": summary,
        "figures": figures,
        "plotly": plotly_figures,
        "tables": tables,
        "summary": summary,
    }


def build_wandb_log_payload(
    *,
    wandb_module,
    scalars: Dict[str, Any],
    diagnostics: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(scalars)
    diagnostics = diagnostics or []

    for bundle in diagnostics:
        payload.update(bundle.get("scalars", {}))

        for key, figure in bundle.get("figures", {}).items():
            payload[key] = wandb_module.Image(figure)
            plt.close(figure)

        plotly_media = getattr(wandb_module, "Plotly", None)
        for key, figure in bundle.get("plotly", {}).items():
            if plotly_media is not None:
                payload[key] = plotly_media(figure)
            else:
                payload[key] = wandb_module.Html(
                    figure.to_html(include_plotlyjs="cdn"),
                    inject=False,
                )

        for key, table_df in bundle.get("tables", {}).items():
            payload[key] = wandb_module.Table(dataframe=table_df)

    return payload
