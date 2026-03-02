#!/usr/bin/env python3
"""
Clean and flatten NL-command pairs generated from man pages.

This script preserves the original notebook cleaning behavior:
- Keep records that have a non-empty `examples` list.

It also adds optional strict filtering for training-oriented outputs:
- Validate first command token against aliases/invocation from raw tool metadata.
- Optionally validate command flags against known flags from metadata.
- Optionally deduplicate by (tool, command).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


@dataclass
class ToolRules:
    allowed_first_tokens: Set[str]
    allowed_flags: Set[str]
    allowed_short_flag_chars: Set[str]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data).__name__}")
    return data


def normalize_examples(examples: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(examples, list):
        return out
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        nl = str(ex.get("nl_query", "")).strip()
        cmd = str(ex.get("command", "")).strip()
        if not nl or not cmd:
            continue
        out.append({"nl_query": nl, "command": cmd})
    return out


def clean_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        tool = str(rec.get("tool", "")).strip()
        source_url = rec.get("source_url")
        examples = normalize_examples(rec.get("examples"))
        if not tool or not examples:
            continue
        cleaned.append(
            {
                "tool": tool,
                "source_url": source_url,
                "examples": examples,
            }
        )
    return cleaned


def flatten_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for rec in records:
        tool = rec["tool"]
        source_url = rec.get("source_url")
        for ex in rec["examples"]:
            rows.append(
                {
                    "tool": tool,
                    "source_url": source_url,
                    "nl_query": ex["nl_query"],
                    "command": ex["command"],
                }
            )
    return rows


def _extract_first_tokens_from_name(name: str) -> Set[str]:
    tokens: Set[str] = set()
    # Examples: "tset, reset", "nfs4_setfacl, nfs4_editfacl"
    for part in re.split(r"[,/]|\s+or\s+", name):
        part = part.strip()
        if part:
            tokens.add(part)
    return tokens


def _extract_first_tokens_from_invocation(invocation: str) -> Set[str]:
    tokens: Set[str] = set()
    for line in invocation.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split()[0].strip("[]()")
        if first and not first.startswith("-"):
            tokens.add(first)
    return tokens


def build_tool_rules(raw_tools: List[Dict[str, Any]]) -> Dict[str, ToolRules]:
    by_url: Dict[str, ToolRules] = {}
    for rec in raw_tools:
        source_url = rec.get("source_url")
        if not source_url:
            continue

        name = str(rec.get("name", "")).strip()
        invocation = str(rec.get("invocation", "")).strip()
        options = rec.get("options", [])

        allowed_first_tokens: Set[str] = set()
        if name:
            allowed_first_tokens.update(_extract_first_tokens_from_name(name))
        if invocation:
            allowed_first_tokens.update(_extract_first_tokens_from_invocation(invocation))

        allowed_flags: Set[str] = set()
        short_chars: Set[str] = set()
        if isinstance(options, list):
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                flags = opt.get("flags", [])
                if not isinstance(flags, list):
                    continue
                for raw_flag in flags:
                    f = str(raw_flag).strip()
                    if not f:
                        continue
                    for part in re.split(r"\s*,\s*", f):
                        part = part.strip()
                        if not part.startswith("-"):
                            continue
                        base = part.split("=")[0]
                        allowed_flags.add(base)
                        if re.fullmatch(r"-[A-Za-z0-9?]", base):
                            short_chars.add(base[1])

        by_url[source_url] = ToolRules(
            allowed_first_tokens=allowed_first_tokens,
            allowed_flags=allowed_flags,
            allowed_short_flag_chars=short_chars,
        )
    return by_url


def command_first_token(command: str) -> str:
    tokens = command.split()
    return tokens[0] if tokens else ""


def is_first_token_valid(row: Dict[str, str], rules_by_url: Dict[str, ToolRules]) -> bool:
    source_url = row.get("source_url")
    if not source_url or source_url not in rules_by_url:
        return True
    rules = rules_by_url[source_url]
    if not rules.allowed_first_tokens:
        return True
    return command_first_token(row["command"]) in rules.allowed_first_tokens


def _iter_flag_tokens(command: str) -> Iterable[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for tok in tokens[1:]:
        if tok.startswith("-"):
            yield tok


def is_flag_set_valid(row: Dict[str, str], rules_by_url: Dict[str, ToolRules]) -> bool:
    source_url = row.get("source_url")
    if not source_url or source_url not in rules_by_url:
        return True
    rules = rules_by_url[source_url]
    if not rules.allowed_flags:
        return True

    for tok in _iter_flag_tokens(row["command"]):
        base = tok.split("=")[0]
        if base == "--":
            continue
        if base in rules.allowed_flags:
            continue
        # Allow bundled short opts like -xzf if each char is known.
        if re.fullmatch(r"-[A-Za-z0-9?]{2,}", base):
            if all(ch in rules.allowed_short_flag_chars for ch in base[1:]):
                continue
        return False
    return True


def dedupe_tool_command(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in rows:
        key = (row["tool"], row["command"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and flatten NL-command pairs.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/man/nl_command_pairs.json"),
        help="Input JSON/JSONL from generation stage.",
    )
    parser.add_argument(
        "--raw-tools-json",
        type=Path,
        default=Path("data/man/raw_ai.json"),
        help="Raw tool metadata JSON used for strict validation.",
    )

    parser.add_argument(
        "--output-clean-json",
        type=Path,
        default=Path("data/man/nl_command_pairs_cleaned_v2.json"),
        help="Output cleaned nested JSON.",
    )
    parser.add_argument(
        "--output-clean-jsonl",
        type=Path,
        default=Path("data/man/nl_command_pairs_cleaned_v2.jsonl"),
        help="Output cleaned nested JSONL.",
    )
    parser.add_argument(
        "--output-flat-json",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_clean_v2.json"),
        help="Output flattened clean JSON.",
    )
    parser.add_argument(
        "--output-flat-jsonl",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_clean_v2.jsonl"),
        help="Output flattened clean JSONL.",
    )
    parser.add_argument(
        "--output-flat-train-json",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_train_v2.json"),
        help="Output flattened strict/dedup JSON for training.",
    )
    parser.add_argument(
        "--output-flat-train-jsonl",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_train_v2.jsonl"),
        help="Output flattened strict/dedup JSONL for training.",
    )

    parser.add_argument(
        "--strict-first-token",
        action="store_true",
        help="Enable strict first-token validation for training output.",
    )
    parser.add_argument(
        "--strict-flags",
        action="store_true",
        help="Enable strict flag validation for training output.",
    )
    parser.add_argument(
        "--dedupe-tool-command",
        action="store_true",
        help="Dedupe training output by (tool, command).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_records = load_json_or_jsonl(args.input_json)
    cleaned_records = clean_records(source_records)
    flat_clean = flatten_records(cleaned_records)

    print(f"Loaded records: {len(source_records)}")
    print(f"Cleaned records (examples present): {len(cleaned_records)}")
    print(f"Flat clean examples: {len(flat_clean)}")

    write_json(args.output_clean_json, cleaned_records)
    write_jsonl(args.output_clean_jsonl, cleaned_records)
    write_json(args.output_flat_json, flat_clean)
    write_jsonl(args.output_flat_jsonl, flat_clean)

    print(f"Saved cleaned JSON: {args.output_clean_json}")
    print(f"Saved cleaned JSONL: {args.output_clean_jsonl}")
    print(f"Saved flat clean JSON: {args.output_flat_json}")
    print(f"Saved flat clean JSONL: {args.output_flat_jsonl}")

    # Build training-oriented output (optional strict filters).
    training_rows = list(flat_clean)

    rules_by_url: Dict[str, ToolRules] = {}
    if args.strict_first_token or args.strict_flags:
        raw_tools = load_json_or_jsonl(args.raw_tools_json)
        rules_by_url = build_tool_rules(raw_tools)
        print(f"Loaded raw tool metadata records: {len(raw_tools)}")

    if args.strict_first_token:
        before = len(training_rows)
        training_rows = [r for r in training_rows if is_first_token_valid(r, rules_by_url)]
        print(f"After first-token validation: {len(training_rows)} (removed {before - len(training_rows)})")

    if args.strict_flags:
        before = len(training_rows)
        training_rows = [r for r in training_rows if is_flag_set_valid(r, rules_by_url)]
        print(f"After flag validation: {len(training_rows)} (removed {before - len(training_rows)})")

    if args.dedupe_tool_command:
        before = len(training_rows)
        training_rows = dedupe_tool_command(training_rows)
        print(f"After (tool, command) dedupe: {len(training_rows)} (removed {before - len(training_rows)})")

    write_json(args.output_flat_train_json, training_rows)
    write_jsonl(args.output_flat_train_jsonl, training_rows)
    print(f"Saved train JSON: {args.output_flat_train_json}")
    print(f"Saved train JSONL: {args.output_flat_train_jsonl}")


if __name__ == "__main__":
    main()
