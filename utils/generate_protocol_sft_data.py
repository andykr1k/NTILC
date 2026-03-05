#!/usr/bin/env python3
"""Generate protocol-format SFT datasets for planner and dispatch training."""

from __future__ import annotations

import argparse
import json
import random
import re
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}, got {type(data).__name__}")
    return data


def normalize_flat_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tool = str(row.get("tool", "")).strip()
        query = str(row.get("nl_query", row.get("query", ""))).strip()
        command = str(row.get("command", "")).strip()
        if not tool or not query or not command:
            continue
        out.append({"tool": tool, "query": query, "command": command})
    return out


def split_command_tail(tool: str, command: str) -> str:
    command = str(command).strip()
    if not command:
        return ""
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        if parts[0] == tool:
            return "<NO_ARGS>"
        return command
    if parts[0] == tool:
        return parts[1].strip()
    return command


def normalize_action_text(query: str) -> str:
    text = " ".join(str(query).strip().split())
    text = re.sub(r"^(please\s+)", "", text, flags=re.IGNORECASE)
    text = text.strip(" .;:!?")
    return text


def build_plan_target(actions: Sequence[str]) -> str:
    lines = ["<plan>"]
    for action in actions:
        action_text = xml_escape(str(action).strip())
        if action_text:
            lines.append(f"  <action>{action_text}</action>")
    lines.append("</plan>")
    return "\n".join(lines)


def build_dispatch_target(command_value: str, query_value: str) -> str:
    command_text = xml_escape(str(command_value).strip())
    query_text = xml_escape(str(query_value).strip())
    return (
        "<dispatch>"
        f"<arg><key>command</key><value>{command_text}</value></arg>"
        f"<arg><key>query</key><value>{query_text}</value></arg>"
        "</dispatch>"
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_dispatch_rows(rows: Sequence[Dict[str, str]], mode: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        query = row["query"]
        tool = row["tool"]
        command = row["command"]
        command_payload = command if mode == "full" else split_command_tail(tool, command)
        output.append(
            {
                "task": "dispatch",
                "mode": mode,
                "request": query,
                "action": normalize_action_text(query),
                "tool": tool,
                "prior_steps": [],
                "target": build_dispatch_target(command_payload, query),
            }
        )
    return output


def _compose_request(actions: Sequence[str]) -> str:
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} and then {actions[1]}"
    return ", then ".join(actions[:-1]) + f", and then {actions[-1]}"


def build_planner_rows(
    rows: Sequence[Dict[str, str]],
    seed: int,
    max_rows: int,
    multi_step_ratio: float,
    max_actions_per_plan: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    atomic_actions = [normalize_action_text(row["query"]) for row in rows if normalize_action_text(row["query"])]
    if not atomic_actions:
        return []

    rng.shuffle(atomic_actions)
    target_total = min(max_rows, len(atomic_actions)) if max_rows > 0 else len(atomic_actions)
    if target_total <= 0:
        return []

    multi_count = int(target_total * max(0.0, min(1.0, multi_step_ratio)))
    multi_count = min(multi_count, target_total)
    single_count = target_total - multi_count

    output: List[Dict[str, Any]] = []

    for action in atomic_actions[:single_count]:
        output.append(
            {
                "task": "plan",
                "request": action,
                "target": build_plan_target([action]),
                "num_actions": 1,
            }
        )

    if multi_count > 0:
        pool = list(atomic_actions)
        for _ in range(multi_count):
            k = rng.randint(2, max(2, int(max_actions_per_plan)))
            k = min(k, len(pool))
            sampled = rng.sample(pool, k=k)
            output.append(
                {
                    "task": "plan",
                    "request": _compose_request(sampled),
                    "target": build_plan_target(sampled),
                    "num_actions": len(sampled),
                }
            )

    rng.shuffle(output)
    return output[:target_total]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate protocol-format SFT datasets.")
    parser.add_argument(
        "--input-data",
        type=Path,
        default=Path("data/man/nl_command_pairs_flat_train_v2.jsonl"),
        help="Flat NL-command training data (.json/.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/protocol"),
        help="Output directory for generated protocol datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--planner-max-rows",
        type=int,
        default=120000,
        help="Maximum planner rows to generate (0 means all single-step rows).",
    )
    parser.add_argument(
        "--planner-multi-step-ratio",
        type=float,
        default=0.35,
        help="Fraction of planner rows that should be synthetic multi-step tasks.",
    )
    parser.add_argument(
        "--planner-max-actions-per-plan",
        type=int,
        default=3,
        help="Maximum number of actions in synthetic multi-step plan rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = normalize_flat_rows(load_rows(args.input_data))
    if not rows:
        raise ValueError(f"No valid rows found in {args.input_data}")

    dispatch_full = build_dispatch_rows(rows, mode="full")
    dispatch_tail = build_dispatch_rows(rows, mode="tail")
    planner = build_planner_rows(
        rows=rows,
        seed=args.seed,
        max_rows=args.planner_max_rows,
        multi_step_ratio=args.planner_multi_step_ratio,
        max_actions_per_plan=args.planner_max_actions_per_plan,
    )

    output_dir = args.output_dir
    write_jsonl(output_dir / "dispatch_full_protocol.jsonl", dispatch_full)
    write_jsonl(output_dir / "dispatch_tail_protocol.jsonl", dispatch_tail)
    write_jsonl(output_dir / "planner_protocol.jsonl", planner)

    print(f"Input rows: {len(rows)}")
    print(f"Dispatch full rows: {len(dispatch_full)} -> {output_dir / 'dispatch_full_protocol.jsonl'}")
    print(f"Dispatch tail rows: {len(dispatch_tail)} -> {output_dir / 'dispatch_tail_protocol.jsonl'}")
    print(f"Planner rows: {len(planner)} -> {output_dir / 'planner_protocol.jsonl'}")


if __name__ == "__main__":
    main()
