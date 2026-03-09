"""Pure-Python helpers for tool-calling evaluation."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FULL_CLEAN_SPLIT = "full_clean"
TRAIN_OVERLAP_SPLIT = "train_overlap"
UNSEEN_ONLY_SPLIT = "unseen_only"
SUPPORTED_SPLITS = {FULL_CLEAN_SPLIT, TRAIN_OVERLAP_SPLIT, UNSEEN_ONLY_SPLIT}


@dataclass(frozen=True)
class ToolMetadata:
    tool_name: str
    one_line: str
    invocation: str
    canonical_prefix_tokens: Tuple[str, ...]
    alias_prefixes: Tuple[Tuple[str, ...], ...]

    @property
    def canonical_prefix(self) -> str:
        return " ".join(self.canonical_prefix_tokens).strip()


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data).__name__}.")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def normalize_command_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue

        if "examples" in row:
            tool = str(row.get("tool", "")).strip()
            source_url = str(row.get("source_url", "") or "")
            for example in row.get("examples", []):
                if not isinstance(example, Mapping):
                    continue
                query = str(example.get("nl_query", example.get("query", ""))).strip()
                command = str(example.get("command", "")).strip()
                if tool and query and command:
                    normalized.append(
                        {
                            "tool": tool,
                            "source_url": source_url,
                            "nl_query": query,
                            "command": command,
                        }
                    )
            continue

        tool = str(row.get("tool", "")).strip()
        query = str(row.get("nl_query", row.get("query", ""))).strip()
        command = str(row.get("command", row.get("ground_truth", ""))).strip()
        if tool and query and command:
            normalized.append(
                {
                    "tool": tool,
                    "source_url": str(row.get("source_url", "") or ""),
                    "nl_query": query,
                    "command": command,
                }
            )
    return normalized


def collapse_raw_tool_records(raw_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    collapsed: Dict[str, Dict[str, str]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue

        current = collapsed.get(
            name,
            {
                "name": name,
                "one_line": "",
                "invocation": "",
                "source_url": "",
            },
        )
        one_line = str(row.get("one_line", "") or "").strip()
        invocation = str(row.get("invocation", "") or "").strip()
        source_url = str(row.get("source_url", "") or "").strip()

        if len(one_line) > len(current["one_line"]):
            current["one_line"] = one_line
        if len(invocation) > len(current["invocation"]):
            current["invocation"] = invocation
        if source_url and not current["source_url"]:
            current["source_url"] = source_url

        collapsed[name] = current

    return [collapsed[name] for name in sorted(collapsed.keys())]


def build_tool_metadata(raw_rows: Iterable[Mapping[str, Any]]) -> Dict[str, ToolMetadata]:
    metadata: Dict[str, ToolMetadata] = {}
    for row in collapse_raw_tool_records(raw_rows):
        tool_name = row["name"]
        canonical_prefix_tokens = _extract_invocation_prefix_tokens(
            row.get("invocation", ""),
            fallback_tool_name=tool_name,
        )
        if not canonical_prefix_tokens:
            fallback_token = _normalize_executable_token(tool_name)
            canonical_prefix_tokens = (fallback_token,) if fallback_token else tuple()

        alias_prefixes = {tuple(canonical_prefix_tokens)}
        for alias in _split_tool_name_aliases(tool_name):
            normalized = _normalize_executable_token(alias)
            if normalized:
                alias_prefixes.add((normalized,))

        metadata[tool_name] = ToolMetadata(
            tool_name=tool_name,
            one_line=str(row.get("one_line", "") or "").strip(),
            invocation=str(row.get("invocation", "") or "").strip(),
            canonical_prefix_tokens=tuple(canonical_prefix_tokens),
            alias_prefixes=tuple(sorted(alias_prefixes, key=lambda item: (-len(item), item))),
        )
    return metadata


def row_identity(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("tool", "")).strip(),
        str(row.get("nl_query", row.get("query", ""))).strip(),
        str(row.get("command", row.get("ground_truth", ""))).strip(),
    )


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def tokenize_command(command: str) -> List[str]:
    text = normalize_whitespace(command)
    if not text:
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def is_complex_shell_command(command: str) -> bool:
    text = str(command).strip()
    if not text:
        return False
    if "$(" in text or "`" in text:
        return True

    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = text.split()

    complex_tokens = {"|", "||", "&&", ";", "<", ">", ">>", "<<", "<>", "|&"}
    return any(token in complex_tokens for token in tokens)


def canonical_prefix_for_tool(
    tool_name: str,
    tool_metadata: Mapping[str, ToolMetadata],
) -> str:
    metadata = tool_metadata.get(str(tool_name).strip())
    if metadata is not None and metadata.canonical_prefix:
        return metadata.canonical_prefix
    fallback = _normalize_executable_token(tool_name)
    return fallback


def canonicalize_command(
    command: str,
    tool_name: str,
    tool_metadata: Mapping[str, ToolMetadata],
) -> str:
    tokens = tokenize_command(command)
    if not tokens:
        return ""

    metadata = tool_metadata.get(str(tool_name).strip())
    if metadata is None or not metadata.canonical_prefix_tokens:
        return " ".join(tokens)

    compare_tokens = tuple(_normalize_executable_token(token) for token in tokens)
    for alias_prefix in metadata.alias_prefixes:
        if _starts_with(compare_tokens, alias_prefix):
            tail = tokens[len(alias_prefix) :]
            return " ".join((*metadata.canonical_prefix_tokens, *tail)).strip()

    return " ".join(tokens).strip()


def prepare_eval_rows(
    clean_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    tool_metadata: Mapping[str, ToolMetadata],
) -> List[Dict[str, Any]]:
    clean_normalized = normalize_command_rows(clean_rows)
    train_normalized = normalize_command_rows(train_rows)
    train_keys = {row_identity(row) for row in train_normalized}

    prepared: List[Dict[str, Any]] = []
    for index, row in enumerate(clean_normalized):
        split_name = TRAIN_OVERLAP_SPLIT if row_identity(row) in train_keys else UNSEEN_ONLY_SPLIT
        prepared.append(
            {
                "example_id": index,
                "tool": row["tool"],
                "source_url": row.get("source_url", ""),
                "nl_query": row["nl_query"],
                "command": row["command"],
                "raw_command_normalized": normalize_whitespace(row["command"]),
                "canonical_command": canonicalize_command(
                    row["command"],
                    tool_name=row["tool"],
                    tool_metadata=tool_metadata,
                ),
                "canonical_tool_prefix": canonical_prefix_for_tool(row["tool"], tool_metadata),
                "is_complex_shell": is_complex_shell_command(row["command"]),
                "split": split_name,
            }
        )
    return prepared


def select_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    include_complex: bool,
    num_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    split_name = str(split).strip()
    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split: {split_name}")

    selected: List[Dict[str, Any]] = []
    for row in rows:
        if split_name != FULL_CLEAN_SPLIT and row.get("split") != split_name:
            continue
        if not include_complex and bool(row.get("is_complex_shell")):
            continue
        selected.append(dict(row))

    if num_samples is None or num_samples <= 0 or len(selected) <= num_samples:
        return selected

    import random

    rng = random.Random(seed)
    sampled = list(selected)
    rng.shuffle(sampled)
    return sampled[:num_samples]


def dataset_partition_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        FULL_CLEAN_SPLIT: len(rows),
        TRAIN_OVERLAP_SPLIT: sum(1 for row in rows if row.get("split") == TRAIN_OVERLAP_SPLIT),
        UNSEEN_ONLY_SPLIT: sum(1 for row in rows if row.get("split") == UNSEEN_ONLY_SPLIT),
        "complex_shell": sum(1 for row in rows if bool(row.get("is_complex_shell"))),
        "non_complex": sum(1 for row in rows if not bool(row.get("is_complex_shell"))),
    }


def evaluate_command_prediction(
    predicted_command: str,
    gold_row: Mapping[str, Any],
    tool_metadata: Mapping[str, ToolMetadata],
) -> Dict[str, Any]:
    raw_command = normalize_whitespace(predicted_command)
    canonical_command = canonicalize_command(
        predicted_command,
        tool_name=str(gold_row.get("tool", "")),
        tool_metadata=tool_metadata,
    )
    expected_prefix_tokens = tuple(
        tokenize_command(str(gold_row.get("canonical_tool_prefix", "")))
    )
    predicted_prefix_tokens = tuple(tokenize_command(canonical_command))
    prefix_match = bool(expected_prefix_tokens) and _starts_with(
        predicted_prefix_tokens,
        expected_prefix_tokens,
    )

    return {
        "predicted_command": str(predicted_command).strip(),
        "predicted_command_raw_normalized": raw_command,
        "predicted_command_canonical": canonical_command,
        "raw_exact_match": raw_command == str(gold_row.get("raw_command_normalized", "")),
        "canonical_exact_match": canonical_command == str(gold_row.get("canonical_command", "")),
        "command_tool_match": prefix_match,
    }


def aggregate_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_examples": len(predictions),
        "systems": {},
    }

    for system_name in ("ntilc", "baseline"):
        system_rows = [
            row.get(system_name)
            for row in predictions
            if isinstance(row, Mapping) and isinstance(row.get(system_name), Mapping)
        ]
        metrics: Dict[str, Any] = {
            "count": len(system_rows),
            "latency_seconds": _summarize_latency(
                [
                    float(item["latency_seconds"])
                    for item in system_rows
                    if item.get("latency_seconds") is not None
                ]
            ),
        }

        metrics.update(_boolean_rate(system_rows, "raw_exact_match"))
        metrics.update(_boolean_rate(system_rows, "canonical_exact_match"))
        metrics.update(_boolean_rate(system_rows, "command_tool_match"))

        if system_name == "baseline":
            metrics.update(_boolean_rate(system_rows, "strict_json_parse", output_name="strict_json_parse_rate"))
        if system_name == "ntilc":
            metrics.update(_boolean_rate(system_rows, "structured_output", output_name="structured_output_rate"))
            metrics.update(_boolean_rate(system_rows, "retrieval_top1_label_match"))
            metrics.update(_boolean_rate(system_rows, "retrieval_hit_at_k"))

        summary["systems"][system_name] = metrics

    return summary


def _boolean_rate(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    output_name: Optional[str] = None,
) -> Dict[str, float]:
    values = [bool(row[key]) for row in rows if key in row and row[key] is not None]
    if not values:
        return {}
    return {output_name or key: sum(values) / len(values)}


def _summarize_latency(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "total": 0.0}

    ordered = sorted(float(value) for value in values)
    total = float(sum(ordered))
    mean = total / len(ordered)
    p50 = _nearest_rank_percentile(ordered, 0.50)
    p95 = _nearest_rank_percentile(ordered, 0.95)
    return {
        "mean": mean,
        "p50": p50,
        "p95": p95,
        "total": total,
    }


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    rank = max(1, int(round(percentile * len(values) + 0.0000001)))
    index = min(len(values) - 1, rank - 1)
    return float(values[index])


def _split_tool_name_aliases(name: str) -> List[str]:
    aliases = []
    for part in re.split(r"[,/]|(?:\s+or\s+)", str(name).strip()):
        part = part.strip()
        if part:
            aliases.append(part)
    return aliases


def _normalize_executable_token(token: str) -> str:
    text = str(token).strip().strip("[](){}")
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def _extract_invocation_prefix_tokens(
    invocation: str,
    fallback_tool_name: str,
) -> Tuple[str, ...]:
    first_line = str(invocation).strip().splitlines()
    if not first_line:
        fallback = _normalize_executable_token(fallback_tool_name)
        return (fallback,) if fallback else tuple()

    cleaned = first_line[0]
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"<[^>]*>", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        fallback = _normalize_executable_token(fallback_tool_name)
        return (fallback,) if fallback else tuple()

    prefix_tokens: List[str] = []
    executable = _normalize_executable_token(tokens[0])
    if executable:
        prefix_tokens.append(executable)

    for token in tokens[1:]:
        if token.startswith("-"):
            break
        if _looks_like_placeholder(token):
            break
        if any(char.islower() for char in token):
            prefix_tokens.append(_normalize_executable_token(token))
            continue
        break

    return tuple(token for token in prefix_tokens if token)


def _looks_like_placeholder(token: str) -> bool:
    text = str(token).strip()
    if not text:
        return True
    if text.startswith("-"):
        return True
    if any(char in text for char in "[]{}<>"):
        return True
    if "..." in text:
        return True
    letters_only = re.sub(r"[^A-Za-z]", "", text)
    if letters_only and letters_only.isupper():
        return True
    return False


def _starts_with(tokens: Sequence[str], prefix: Sequence[str]) -> bool:
    if len(prefix) > len(tokens):
        return False
    return tuple(tokens[: len(prefix)]) == tuple(prefix)
