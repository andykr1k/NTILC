#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "registry"
TOOLS_DIR = REGISTRY_DIR / "tools"
MODELS_PATH = REGISTRY_DIR / "models" / "releases.yaml"
CATEGORIES_PATH = REGISTRY_DIR / "categories.yaml"
GENERATED_DIR = REGISTRY_DIR / "generated"

ALLOWED_INTERFACE_TYPES = {"cli", "python", "javascript", "http", "other"}
ALLOWED_ARCHITECTURES = {"normal", "hierarchical"}
ALLOWED_LOSSES = {"prototype_ce", "contrastive", "circle"}
ALLOWED_MODEL_STATUSES = {"planned", "ready", "published", "deprecated"}
MODEL_STATUS_ORDER = {"published": 0, "ready": 1, "planned": 2, "deprecated": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile registry manifests into generated JSON assets.")
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=REGISTRY_DIR,
        help="Path to the registry directory.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_categories(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    payload = load_yaml(path) or {}
    categories = payload.get("categories")
    require(isinstance(categories, list) and categories, f"No categories found in {path}")

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(categories):
        require(isinstance(raw, dict), f"Category #{index + 1} in {path} must be an object")
        category_id = str(raw.get("id", "")).strip()
        name = str(raw.get("name", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        require(category_id, f"Category #{index + 1} in {path} is missing id")
        require(name, f"Category {category_id} is missing name")
        require(category_id not in seen_ids, f"Duplicate category id: {category_id}")
        seen_ids.add(category_id)
        normalized.append(
            {
                "id": category_id,
                "name": name,
                "summary": summary,
            }
        )
    return normalized, seen_ids


def load_examples(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"Missing examples file: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            require(isinstance(payload, dict), f"{path}:{line_number} must be a JSON object")
            query = str(payload.get("query", "")).strip()
            require(query, f"{path}:{line_number} is missing query")
            rows.append(
                {
                    "query": query,
                    "split": str(payload.get("split", "train")).strip() or "train",
                    "language": str(payload.get("language", "en")).strip() or "en",
                    "notes": str(payload.get("notes", "")).strip(),
                }
            )
    require(len(rows) >= 3, f"{path} must include at least 3 example queries")
    return rows


def validate_parameters(tool_id: str, parameters: Any) -> dict[str, Any]:
    require(isinstance(parameters, dict), f"{tool_id}: parameters must be an object")
    require(parameters.get("type") == "object", f"{tool_id}: parameters.type must be 'object'")
    properties = parameters.get("properties", {})
    required_fields = parameters.get("required", [])
    require(isinstance(properties, dict), f"{tool_id}: parameters.properties must be an object")
    require(isinstance(required_fields, list), f"{tool_id}: parameters.required must be a list")
    for field_name in required_fields:
        require(field_name in properties, f"{tool_id}: required field '{field_name}' missing from properties")
    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
    }


def load_tool(tool_dir: Path, category_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = tool_dir / "tool.yaml"
    examples_path = tool_dir / "examples.jsonl"
    require(manifest_path.exists(), f"Missing tool manifest: {manifest_path}")
    manifest = load_yaml(manifest_path) or {}
    require(isinstance(manifest, dict), f"{manifest_path} must contain an object")

    tool_id = str(manifest.get("id", "")).strip()
    require(tool_id, f"{manifest_path} is missing id")

    interface_type = str(manifest.get("interface_type", "")).strip()
    require(interface_type in ALLOWED_INTERFACE_TYPES, f"{tool_id}: unsupported interface type '{interface_type}'")

    parent_id = str(manifest.get("parent_id") or manifest.get("parent_category") or "").strip()
    require(parent_id in category_ids, f"{tool_id}: unknown parent_id '{parent_id}'")

    maintainers = manifest.get("maintainers", [])
    tags = manifest.get("tags", [])
    require(isinstance(maintainers, list) and maintainers, f"{tool_id}: maintainers must be a non-empty list")
    require(isinstance(tags, list) and tags, f"{tool_id}: tags must be a non-empty list")

    parameters = validate_parameters(tool_id, manifest.get("parameters", {}))
    examples = load_examples(examples_path)

    tests_path = tool_dir / "tests.json"
    tool_record = {
        "id": tool_id,
        "display_name": str(manifest.get("display_name", "")).strip(),
        "description": str(manifest.get("description", "")).strip(),
        "interface_type": interface_type,
        "source_repo": str(manifest.get("source_repo", "")).strip(),
        "homepage": str(manifest.get("homepage", "")).strip(),
        "license": str(manifest.get("license", "")).strip(),
        "maintainers": [str(item).strip() for item in maintainers if str(item).strip()],
        "parent_id": parent_id,
        "parent_category": parent_id,
        "tags": [str(item).strip() for item in tags if str(item).strip()],
        "parameters": parameters,
        "example_count": len(examples),
        "has_tests": tests_path.exists(),
        "registry_path": str(tool_dir.relative_to(REPO_ROOT)),
    }

    for field_name in ("display_name", "description", "source_repo", "license"):
        require(tool_record[field_name], f"{tool_id}: missing {field_name}")

    dataset_rows: list[dict[str, Any]] = []
    for example in examples:
        dataset_rows.append(
            {
                "tool": tool_id,
                "query": example["query"],
                "parent_id": parent_id,
                "parent_category": parent_id,
                "interface_type": interface_type,
                "split": example["split"],
                "language": example["language"],
                "license": tool_record["license"],
                "source_repo": tool_record["source_repo"],
                "registry_path": tool_record["registry_path"],
            }
        )
    return tool_record, dataset_rows


def load_model_releases(path: Path) -> list[dict[str, Any]]:
    payload = load_yaml(path) or {}
    releases = payload.get("releases")
    require(isinstance(releases, list), f"No releases found in {path}")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(releases):
        require(isinstance(raw, dict), f"Release #{index + 1} in {path} must be an object")
        release_id = str(raw.get("id", "")).strip()
        architecture = str(raw.get("architecture", "")).strip()
        loss = str(raw.get("loss", "")).strip()
        status = str(raw.get("status", "")).strip()
        require(release_id, f"Release #{index + 1} in {path} is missing id")
        require(release_id not in seen_ids, f"Duplicate release id: {release_id}")
        require(architecture in ALLOWED_ARCHITECTURES, f"{release_id}: invalid architecture '{architecture}'")
        require(loss in ALLOWED_LOSSES, f"{release_id}: invalid loss '{loss}'")
        require(status in ALLOWED_MODEL_STATUSES, f"{release_id}: invalid status '{status}'")
        metrics = raw.get("metrics", {})
        require(isinstance(metrics, dict), f"{release_id}: metrics must be an object")
        download_url = str(raw.get("download_url", "") or "").strip()
        repository_url = str(raw.get("repository_url", "") or "").strip()
        if status == "published":
            require(download_url, f"{release_id}: published releases require download_url")
        seen_ids.add(release_id)
        normalized.append(
            {
                "id": release_id,
                "title": str(raw.get("title", "")).strip(),
                "architecture": architecture,
                "loss": loss,
                "encoder": str(raw.get("encoder", "")).strip(),
                "embedding_dim": int(raw.get("embedding_dim", 0) or 0),
                "dataset_version": str(raw.get("dataset_version", "")).strip(),
                "status": status,
                "published_at": str(raw.get("published_at", "") or "").strip(),
                "download_url": download_url,
                "repository_url": repository_url,
                "sha256": str(raw.get("sha256", "") or "").strip(),
                "notes": str(raw.get("notes", "") or "").strip(),
                "metrics": metrics,
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            MODEL_STATUS_ORDER.get(item["status"], 99),
            item["architecture"],
            item["loss"],
            item["id"],
        ),
    )


def main() -> None:
    args = parse_args()
    registry_dir = args.registry_dir.resolve()
    categories, category_ids = load_categories(registry_dir / "categories.yaml")
    tools_dir = registry_dir / "tools"
    generated_dir = registry_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    tool_records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    seen_tool_ids: set[str] = set()

    for tool_dir in sorted(path for path in tools_dir.iterdir() if path.is_dir()):
        tool_record, tool_examples = load_tool(tool_dir, category_ids)
        require(tool_record["id"] not in seen_tool_ids, f"Duplicate tool id: {tool_record['id']}")
        seen_tool_ids.add(tool_record["id"])
        tool_records.append(tool_record)
        dataset_rows.extend(tool_examples)

    model_releases = load_model_releases(registry_dir / "models" / "releases.yaml")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    category_counts = Counter(item["parent_id"] for item in tool_records)
    registry_manifest = {
        "version": 1,
        "generated_at": generated_at,
        "tool_count": len(tool_records),
        "example_count": len(dataset_rows),
        "category_count": len(categories),
        "model_release_count": len(model_releases),
        "published_model_count": sum(1 for item in model_releases if item["status"] == "published"),
        "categories": [
            {
                "id": category["id"],
                "name": category["name"],
                "tool_count": category_counts.get(category["id"], 0),
            }
            for category in categories
        ],
        "artifacts": {
            "tools": "registry/generated/tools.json",
            "models": "registry/generated/models.json",
            "hierarchy": "registry/generated/hierarchy.json",
            "dataset": "registry/generated/tool_embedding_dataset.jsonl",
        },
    }

    tools_payload = {
        "version": 1,
        "generated_at": generated_at,
        "categories": categories,
        "tools": sorted(tool_records, key=lambda item: item["id"]),
    }
    models_payload = {
        "version": 1,
        "generated_at": generated_at,
        "releases": model_releases,
    }
    hierarchy_payload = {item["id"]: item["parent_id"] for item in tool_records}

    write_json(generated_dir / "tools.json", tools_payload)
    write_json(generated_dir / "models.json", models_payload)
    write_json(generated_dir / "hierarchy.json", hierarchy_payload)
    write_json(generated_dir / "registry_manifest.json", registry_manifest)
    write_jsonl(generated_dir / "tool_embedding_dataset.jsonl", dataset_rows)

    print(
        f"Built registry with {len(tool_records)} tools, "
        f"{len(dataset_rows)} example queries, and {len(model_releases)} model releases."
    )


if __name__ == "__main__":
    main()
