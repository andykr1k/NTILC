from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def load_dataset_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        return load_jsonl_rows(path)

    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("rows", "examples", "dataset"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ValueError(
                f"Expected {path} to contain a top-level list or one of: rows, examples, dataset."
            )
    else:
        raise ValueError(f"Unsupported dataset payload in {path}: expected JSON object or list.")

    normalized_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1} must be a JSON object")
        normalized_rows.append(row)
    return normalized_rows


def clean_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        tool = str(row.get("tool", "")).strip()
        query = str(row.get("query", row.get("text", ""))).strip()
        if tool and query:
            normalized = dict(row)
            normalized["tool"] = tool
            normalized["query"] = query
            cleaned.append(normalized)
    return cleaned


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def count_rows_by_tool(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["tool"]) for row in rows).items()))


def split_rows_by_tool(
    rows: Sequence[Dict[str, Any]],
    *,
    examples_per_tool: int,
    test_per_tool: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if examples_per_tool <= 0:
        raise ValueError("examples_per_tool must be positive.")
    if test_per_tool <= 0:
        raise ValueError("test_per_tool must be positive.")
    if test_per_tool >= examples_per_tool:
        raise ValueError("test_per_tool must be smaller than examples_per_tool.")

    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["tool"])].append(dict(row))

    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    selected_counts: Dict[str, int] = {}

    for tool_name in sorted(grouped):
        tool_rows = list(grouped[tool_name])
        if len(tool_rows) < examples_per_tool:
            raise ValueError(
                f"Tool {tool_name!r} has only {len(tool_rows)} rows, but {examples_per_tool} are required."
            )

        rng.shuffle(tool_rows)
        selected_rows = tool_rows[:examples_per_tool]
        selected_counts[tool_name] = len(selected_rows)

        for row in selected_rows[:test_per_tool]:
            updated = dict(row)
            updated["split"] = "test"
            test_rows.append(updated)

        for row in selected_rows[test_per_tool:]:
            updated = dict(row)
            updated["split"] = "train"
            train_rows.append(updated)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    summary = {
        "tool_count": len(grouped),
        "examples_per_tool_selected": examples_per_tool,
        "test_per_tool": test_per_tool,
        "train_per_tool": examples_per_tool - test_per_tool,
        "rows_selected": len(train_rows) + len(test_rows),
        "train_rows_written": len(train_rows),
        "test_rows_written": len(test_rows),
        "selected_counts_per_tool": selected_counts,
        "train_counts_per_tool": count_rows_by_tool(train_rows),
        "test_counts_per_tool": count_rows_by_tool(test_rows),
        "seed": seed,
    }
    return train_rows, test_rows, summary
