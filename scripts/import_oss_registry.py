#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLS_PATH = REPO_ROOT / "data" / "OSS" / "tools.json"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "OSS" / "tool_embedding_dataset.jsonl"
DEFAULT_HIERARCHY_PATH = REPO_ROOT / "data" / "OSS" / "tool_embedding_dataset_hierarchy.json"
DEFAULT_REGISTRY_DIR = REPO_ROOT / "registry"
DEFAULT_SOURCE_REPO = "https://github.com/andykr1k/NTILC"
DEFAULT_MAINTAINERS = ["andykr1k"]
DEFAULT_LICENSE = "MIT"


CATEGORY_SUMMARIES = {
    "Information Retrieval": "Search the web, fetch remote content, and query retrieval-oriented indexes.",
    "Computation": "Execute code, run notebook or SQL workloads, and perform local calculations.",
    "File System": "Read, write, inspect, search, and manipulate local files and directories.",
    "Memory": "Persist, retrieve, update, and inspect lightweight key-value memory entries.",
    "External Integrations": "Call external APIs and services through generic HTTP request workflows.",
    "Structured Data": "Parse documents and convert content between structured data formats.",
}


INTERFACE_TYPE_BY_TOOL = {
    "web_search": "http",
    "fetch_url": "http",
    "search_knowledge_base": "python",
    "run_python": "python",
    "run_javascript": "javascript",
    "run_sql": "python",
    "run_notebook_cell": "python",
    "calculate": "python",
    "read_file": "python",
    "write_file": "python",
    "edit_file": "python",
    "list_directory": "python",
    "move_file": "python",
    "copy_file": "python",
    "delete_file": "python",
    "fetch_file_metadata": "python",
    "grep": "cli",
    "run_bash_command": "cli",
    "store_memory": "python",
    "retrieve_memory": "python",
    "update_memory": "python",
    "delete_memory": "python",
    "list_memories": "python",
    "send_http_request": "http",
    "parse_document": "python",
    "convert_format": "python",
}


TAGS_BY_CATEGORY = {
    "Information Retrieval": ["retrieval", "search", "web"],
    "Computation": ["execution", "analysis", "runtime"],
    "File System": ["files", "workspace", "local-io"],
    "Memory": ["state", "storage", "kv"],
    "External Integrations": ["http", "api", "network"],
    "Structured Data": ["documents", "formats", "parsing"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the OSS tool dataset into registry manifests.")
    parser.add_argument("--tools-path", type=Path, default=DEFAULT_TOOLS_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--hierarchy-path", type=Path, default=DEFAULT_HIERARCHY_PATH)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--homepage", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--license", default=DEFAULT_LICENSE)
    parser.add_argument("--maintainer", action="append", dest="maintainers")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            require(isinstance(payload, dict), f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


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


def humanize_tool_name(name: str) -> str:
    special_cases = {
        "sql": "SQL",
        "url": "URL",
        "http": "HTTP",
        "javascript": "JavaScript",
    }
    words = name.split("_")
    return " ".join(special_cases.get(word, word.capitalize()) for word in words)


def slugify_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_parameters(parameters: Any) -> dict[str, Any]:
    require(isinstance(parameters, dict), "tool parameters must be an object")
    require(parameters.get("type") == "object", "tool parameters.type must be 'object'")
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    require(isinstance(properties, dict), "tool parameters.properties must be an object")
    require(isinstance(required, list), "tool parameters.required must be a list")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_tool_manifest(
    tool: dict[str, Any],
    parent_id: str,
    source_repo: str,
    homepage: str,
    license_name: str,
    maintainers: list[str],
) -> dict[str, Any]:
    tool_id = str(tool["name"]).strip()
    category_name = str(tool["category"]).strip().replace("_", " ").title()
    tags = list(dict.fromkeys([*TAGS_BY_CATEGORY.get(category_name, []), category_name.lower().replace(" ", "-"), tool_id]))
    return {
        "id": tool_id,
        "display_name": humanize_tool_name(tool_id),
        "description": str(tool["description"]).strip(),
        "interface_type": INTERFACE_TYPE_BY_TOOL.get(tool_id, "other"),
        "source_repo": source_repo,
        "homepage": homepage,
        "license": license_name,
        "maintainers": maintainers,
        "parent_id": parent_id,
        "tags": tags,
        "parameters": normalize_parameters(tool["parameters"]),
    }


def main() -> None:
    args = parse_args()
    registry_dir = args.registry_dir.resolve()
    tools_path = args.tools_path.resolve()
    dataset_path = args.dataset_path.resolve()
    hierarchy_path = args.hierarchy_path.resolve()
    maintainers = args.maintainers or list(DEFAULT_MAINTAINERS)

    tools_payload = load_json(tools_path)
    hierarchy_payload = load_json(hierarchy_path)
    dataset_rows = load_jsonl(dataset_path)

    require(isinstance(tools_payload, dict) and isinstance(tools_payload.get("tools"), list), f"Invalid tools payload: {tools_path}")
    require(isinstance(hierarchy_payload, dict), f"Invalid hierarchy payload: {hierarchy_path}")

    tools: list[dict[str, Any]] = tools_payload["tools"]
    examples_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        tool_id = str(row.get("tool", "")).strip()
        require(tool_id, f"Dataset row missing tool id: {row}")
        examples_by_tool[tool_id].append(
            {
                "query": str(row.get("query", "")).strip(),
                "split": "train",
                "language": "en",
                "notes": f"source={str(row.get('source', '')).strip() or 'unknown'}; generator={str(row.get('generator_model', '')).strip() or 'unknown'}",
            }
        )

    categories_in_use: dict[str, str] = {}
    normalized_hierarchy = {str(tool).strip(): str(parent).strip() for tool, parent in hierarchy_payload.items()}
    require(normalized_hierarchy, f"No hierarchy entries found in {hierarchy_path}")

    tools_dir = registry_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    for tool in tools_dir.iterdir():
        if tool.is_dir():
            for child in tool.iterdir():
                if child.is_file():
                    child.unlink()
            tool.rmdir()

    imported_tools = 0
    imported_examples = 0
    for tool in tools:
        tool_id = str(tool.get("name", "")).strip()
        require(tool_id, "Tool is missing name")
        require(tool_id in normalized_hierarchy, f"Missing hierarchy mapping for {tool_id}")
        require(tool_id in examples_by_tool, f"Missing dataset examples for {tool_id}")

        parent_name = normalized_hierarchy[tool_id]
        parent_id = slugify_category(parent_name)
        categories_in_use[parent_id] = parent_name
        manifest = build_tool_manifest(
            tool=tool,
            parent_id=parent_id,
            source_repo=args.source_repo,
            homepage=args.homepage,
            license_name=args.license,
            maintainers=maintainers,
        )

        tool_dir = tools_dir / tool_id
        tool_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(tool_dir / "tool.yaml", manifest)
        write_jsonl(tool_dir / "examples.jsonl", examples_by_tool[tool_id])

        imported_tools += 1
        imported_examples += len(examples_by_tool[tool_id])

    categories_payload = {
        "version": 1,
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "summary": CATEGORY_SUMMARIES.get(
                    category_name,
                    f"Tools grouped under the {category_name} family.",
                ),
            }
            for category_id, category_name in sorted(categories_in_use.items(), key=lambda item: item[1])
        ],
    }
    write_yaml(registry_dir / "categories.yaml", categories_payload)

    import_manifest = {
        "version": 1,
        "tools_imported": imported_tools,
        "examples_imported": imported_examples,
        "source": {
            "tools_path": str(tools_path),
            "dataset_path": str(dataset_path),
            "hierarchy_path": str(hierarchy_path),
        },
        "maintainers": maintainers,
        "source_repo": args.source_repo,
        "homepage": args.homepage,
        "license": args.license,
    }
    write_json(registry_dir / "generated" / "import_manifest.json", import_manifest)

    print(
        f"Imported {imported_tools} tools and {imported_examples} example queries into {tools_dir}"
    )


if __name__ == "__main__":
    main()
