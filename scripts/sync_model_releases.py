#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASES_PATH = REPO_ROOT / "registry" / "models" / "releases.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "OSS" / "output"
DEFAULT_DATASET_SUMMARY = REPO_ROOT / "data" / "OSS" / "tool_embedding_dataset_summary.jsonl"
DEFAULT_CARDS_DIR = REPO_ROOT / "registry" / "models" / "cards"
DEFAULT_PUBLISH_MANIFEST = REPO_ROOT / "registry" / "models" / "publish_manifest.json"
DEFAULT_HF_ORG = "OpenToolEmbeddings"
DEFAULT_HF_ORG_URL = "https://huggingface.co/OpenToolEmbeddings"

STATUS_ORDER = {"published": 0, "ready": 1, "planned": 2, "deprecated": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local OSS checkpoints into registry release metadata.")
    parser.add_argument("--releases-path", type=Path, default=DEFAULT_RELEASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--cards-dir", type=Path, default=DEFAULT_CARDS_DIR)
    parser.add_argument("--publish-manifest", type=Path, default=DEFAULT_PUBLISH_MANIFEST)
    parser.add_argument("--hf-org", default=DEFAULT_HF_ORG)
    parser.add_argument("--hf-org-url", default=DEFAULT_HF_ORG_URL)
    parser.add_argument(
        "--status",
        choices=("planned", "ready", "published", "deprecated"),
        default="ready",
        help="Status to assign while syncing local artifacts. Use 'published' after a successful upload.",
    )
    parser.add_argument(
        "--published-at",
        default="",
        help="Optional publication timestamp, used only when --status=published.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_generator_name(value: str) -> str:
    lowered = value.lower()
    for old, new in (
        ("qwen/", ""),
        ("qwen3.5-", "qwen3.5-"),
        ("qwen3_5-", "qwen3.5-"),
        (" ", "-"),
        ("/", "-"),
    ):
        lowered = lowered.replace(old, new)
    return lowered


def build_dataset_version(summary: dict[str, Any]) -> str:
    generator = normalize_generator_name(str(summary.get("generator_model", "unknown")).strip())
    tool_count = int(summary.get("tool_count", 0) or 0)
    row_count = int(summary.get("rows_written", 0) or 0)
    return f"oss-{generator}-{tool_count}tools-{row_count}queries-v0.1.0"


def sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_best_epoch(history: list[dict[str, Any]], architecture: str) -> dict[str, Any]:
    key = "val_retrieval_accuracy" if architecture == "normal" else "val_tool_retrieval_accuracy"
    require(history, "metrics history is empty")
    return max(history, key=lambda item: float(item.get(key, float("-inf"))))


def round_metric(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def summarize_metrics(history: list[dict[str, Any]], architecture: str, summary: dict[str, Any]) -> dict[str, Any]:
    best_epoch = pick_best_epoch(history, architecture)
    metrics: dict[str, Any] = {
        "epochs": len(history),
        "best_epoch": int(best_epoch.get("best_epoch", best_epoch.get("epoch", 0)) or 0),
        "tool_count": int(summary.get("tool_count", 0) or 0),
        "query_count": int(summary.get("rows_written", 0) or 0),
    }
    if architecture == "normal":
        metrics.update(
            {
                "best_val_retrieval_accuracy": round_metric(float(best_epoch.get("val_retrieval_accuracy", 0.0) or 0.0)),
                "best_val_classification_accuracy": round_metric(
                    float(best_epoch.get("val_classification_accuracy", 0.0) or 0.0)
                ),
                "silhouette_score": round_metric(float(best_epoch.get("validation/tool/silhouette_score", 0.0) or 0.0)),
            }
        )
    else:
        metrics.update(
            {
                "best_val_tool_retrieval_accuracy": round_metric(
                    float(best_epoch.get("val_tool_retrieval_accuracy", 0.0) or 0.0)
                ),
                "best_val_parent_retrieval_accuracy": round_metric(
                    float(best_epoch.get("val_parent_retrieval_accuracy", 0.0) or 0.0)
                ),
                "best_val_tool_classification_accuracy": round_metric(
                    float(best_epoch.get("val_tool_classification_accuracy", 0.0) or 0.0)
                ),
                "best_val_parent_classification_accuracy": round_metric(
                    float(best_epoch.get("val_parent_classification_accuracy", 0.0) or 0.0)
                ),
                "tool_silhouette_score": round_metric(
                    float(best_epoch.get("validation/tool/silhouette_score", 0.0) or 0.0)
                ),
                "parent_silhouette_score": round_metric(
                    float(best_epoch.get("validation/parent/silhouette_score", 0.0) or 0.0)
                ),
            }
        )
    return metrics


def build_release_note(summary: dict[str, Any], status: str) -> str:
    tool_count = int(summary.get("tool_count", 0) or 0)
    row_count = int(summary.get("rows_written", 0) or 0)
    generator = str(summary.get("generator_model", "unknown")).strip() or "unknown"
    suffix = (
        "Published to the OpenToolEmbeddings Hugging Face organization."
        if status == "published"
        else "Local artifacts are packaged and ready for OpenToolEmbeddings publication."
    )
    return f"OSS synthetic snapshot with {tool_count} tools, {row_count} queries, generated via {generator}. {suffix}"


def build_model_card(
    release: dict[str, Any],
    metrics: dict[str, Any],
    summary: dict[str, Any],
    sha256: str,
    repo_id: str,
) -> str:
    metric_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in metrics.items())
    usage = (
        "```python\n"
        "from training import load_checkpoint_bundle\n\n"
        "bundle = load_checkpoint_bundle(\"best.pt\", device=\"cpu\")\n"
        "print(bundle[\"architecture\"], bundle[\"loss_name\"], len(bundle[\"tool_names\"]))\n"
        "```"
    )
    return (
        f"# {release['title']}\n\n"
        f"NTILC tool-embedding checkpoint prepared for `{repo_id}`.\n\n"
        f"## Variant\n\n"
        f"- Architecture: `{release['architecture']}`\n"
        f"- Loss: `{release['loss']}`\n"
        f"- Encoder: `{release['encoder']}`\n"
        f"- Embedding dimension: `{release['embedding_dim']}`\n"
        f"- Dataset version: `{release['dataset_version']}`\n"
        f"- Synthetic generator: `{summary.get('generator_model', 'unknown')}`\n"
        f"- Tool count: `{summary.get('tool_count', 0)}`\n"
        f"- Query count: `{summary.get('rows_written', 0)}`\n"
        f"- Checkpoint SHA256: `{sha256}`\n\n"
        "## Files\n\n"
        "- `best.pt`: best validation checkpoint bundle\n"
        "- `metrics.json`: per-epoch training and validation metrics\n\n"
        "## Usage\n\n"
        "These are NTILC checkpoint bundles, not plain `transformers` repositories. Load them with the local helper below.\n\n"
        f"{usage}\n\n"
        "## Metrics\n\n"
        f"{metric_lines}\n"
    )


def repo_url(org: str, release_id: str) -> str:
    return f"https://huggingface.co/{org}/{release_id}"


def download_url(org: str, release_id: str) -> str:
    return f"https://huggingface.co/{org}/{release_id}/resolve/main/best.pt"


def main() -> None:
    args = parse_args()
    releases_path = args.releases_path.resolve()
    output_root = args.output_root.resolve()
    cards_dir = args.cards_dir.resolve()
    publish_manifest_path = args.publish_manifest.resolve()
    dataset_summary = load_json(args.dataset_summary.resolve())
    releases_payload = load_yaml(releases_path) or {}
    releases = releases_payload.get("releases")
    require(isinstance(releases, list) and releases, f"No releases found in {releases_path}")

    dataset_version = build_dataset_version(dataset_summary)
    published_at = args.published_at.strip()
    if args.status == "published" and not published_at:
        published_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    publish_manifest_releases: list[dict[str, Any]] = []
    updated_releases: list[dict[str, Any]] = []
    for release in releases:
        architecture = str(release.get("architecture", "")).strip()
        loss = str(release.get("loss", "")).strip()
        release_id = str(release.get("id", "")).strip()
        require(release_id, "Release is missing id")

        output_dir = output_root / architecture / loss
        checkpoint_path = output_dir / "best.pt"
        metrics_path = output_dir / "metrics.json"
        require(checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}")
        require(metrics_path.exists(), f"Missing metrics file: {metrics_path}")

        metrics_history = load_json(metrics_path)
        require(isinstance(metrics_history, list), f"Invalid metrics file: {metrics_path}")
        summary_metrics = summarize_metrics(metrics_history, architecture=architecture, summary=dataset_summary)
        checkpoint_sha256 = sha256_for_file(checkpoint_path)
        card_path = cards_dir / f"{release_id}.md"
        exact_repo_url = repo_url(args.hf_org, release_id)

        updated_release = dict(release)
        updated_release["dataset_version"] = dataset_version
        updated_release["status"] = args.status
        updated_release["published_at"] = published_at if args.status == "published" else ""
        updated_release["download_url"] = download_url(args.hf_org, release_id) if args.status == "published" else ""
        updated_release["repository_url"] = exact_repo_url if args.status == "published" else args.hf_org_url
        updated_release["sha256"] = checkpoint_sha256
        updated_release["notes"] = build_release_note(dataset_summary, status=args.status)
        updated_release["metrics"] = summary_metrics

        card_body = build_model_card(
            release=updated_release,
            metrics=summary_metrics,
            summary=dataset_summary,
            sha256=checkpoint_sha256,
            repo_id=f"{args.hf_org}/{release_id}",
        )
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card_body, encoding="utf-8")

        publish_manifest_releases.append(
            {
                "id": release_id,
                "title": updated_release["title"],
                "repo_id": f"{args.hf_org}/{release_id}",
                "repository_url": exact_repo_url,
                "download_url": download_url(args.hf_org, release_id),
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "card_path": str(card_path),
                "sha256": checkpoint_sha256,
                "architecture": architecture,
                "loss": loss,
            }
        )
        updated_releases.append(updated_release)

    updated_releases.sort(
        key=lambda item: (
            STATUS_ORDER.get(str(item.get("status", "")).strip(), 99),
            str(item.get("architecture", "")),
            str(item.get("loss", "")),
            str(item.get("id", "")),
        )
    )

    releases_payload["version"] = int(releases_payload.get("version", 1) or 1)
    releases_payload["releases"] = updated_releases
    write_yaml(releases_path, releases_payload)

    publish_manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "organization": args.hf_org,
        "organization_url": args.hf_org_url,
        "dataset_version": dataset_version,
        "releases": publish_manifest_releases,
    }
    write_json(publish_manifest_path, publish_manifest)

    print(
        f"Synced {len(updated_releases)} releases, wrote model cards to {cards_dir}, "
        f"and updated {publish_manifest_path}"
    )


if __name__ == "__main__":
    main()
