#!/usr/bin/env python3
"""
Inference and lightweight evaluation for NL-command LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise ImportError("Missing `peft`. Install with: pip install peft") from exc


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


def normalize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "examples" in row:
            tool = str(row.get("tool", "")).strip()
            for ex in row.get("examples", []):
                if not isinstance(ex, dict):
                    continue
                query = str(ex.get("nl_query", ex.get("query", ""))).strip()
                command = str(ex.get("command", "")).strip()
                if tool and query and command:
                    out.append({"tool": tool, "nl_query": query, "command": command})
            continue

        tool = str(row.get("tool", "")).strip()
        query = str(row.get("nl_query", row.get("query", ""))).strip()
        command = str(row.get("command", "")).strip()
        if tool and query and command:
            out.append({"tool": tool, "nl_query": query, "command": command})
    return out


def split_command_tail(tool: str, command: str) -> str:
    command = command.strip()
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


def build_prompt(query: str, tool: str, mode: str) -> str:
    if mode == "tail":
        return (
            "You map a user request to shell command arguments.\n"
            "Given the selected tool and request, output only the command tail (arguments and values).\n"
            "Do not repeat the tool name.\n\n"
            f"Tool: {tool}\n"
            f"User request: {query}\n"
            "Command tail:"
        )
    return (
        "You map a user request to exactly one Linux shell command.\n"
        "Output only the command and nothing else.\n\n"
        f"User request: {query}\n"
        "Command:"
    )


def normalize_text(x: str) -> str:
    return " ".join(x.strip().split())


def build_full_from_tail(tool: str, tail: str) -> str:
    tail = tail.strip()
    if not tail or tail == "<NO_ARGS>":
        return tool
    return f"{tool} {tail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LoRA command model.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "tail"], default="full")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--query", type=str, default=None, help="Single-query test.")
    parser.add_argument("--tool", type=str, default=None, help="Tool name required for tail mode single-query test.")

    parser.add_argument("--eval-data", type=Path, default=None, help="Optional eval dataset (.json/.jsonl).")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of eval samples.")
    parser.add_argument("--print-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()

    device = next(model.parameters()).device

    def generate(query: str, tool: str) -> str:
        prompt = build_prompt(query=query, tool=tool, mode=args.mode)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-6),
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text.splitlines()[0].strip()

    if args.query is not None:
        if args.mode == "tail" and not args.tool:
            raise ValueError("--tool is required for --mode tail when using --query.")
        tool = args.tool or ""
        pred = generate(query=args.query, tool=tool)
        if args.mode == "tail":
            full = build_full_from_tail(tool=tool, tail=pred)
            print(f"pred_tail: {pred}")
            print(f"pred_full: {full}")
        else:
            print(f"pred_command: {pred}")
        return

    if args.eval_data is None:
        raise ValueError("Provide either --query or --eval-data.")

    rows = normalize_rows(load_rows(args.eval_data))
    if not rows:
        raise ValueError(f"No valid rows in {args.eval_data}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: min(args.num_samples, len(rows))]

    exact_full = 0
    exact_target = 0
    examples: List[Tuple[str, str, str, str, str]] = []

    for row in rows:
        query = row["nl_query"]
        tool = row["tool"]
        gold_command = row["command"]

        pred_target = generate(query=query, tool=tool)
        if args.mode == "tail":
            gold_target = split_command_tail(tool=tool, command=gold_command)
            pred_command = build_full_from_tail(tool=tool, tail=pred_target)
        else:
            gold_target = gold_command
            pred_command = pred_target

        if normalize_text(pred_command) == normalize_text(gold_command):
            exact_full += 1
        if normalize_text(pred_target) == normalize_text(gold_target):
            exact_target += 1

        if len(examples) < args.print_examples:
            examples.append((query, tool, gold_command, pred_target, pred_command))

    n = len(rows)
    print(f"Evaluated samples: {n}")
    print(f"Mode: {args.mode}")
    print(f"Exact target match: {exact_target}/{n} = {exact_target / n:.4f}")
    print(f"Exact full command match: {exact_full}/{n} = {exact_full / n:.4f}")
    print("")

    for i, (query, tool, gold, pred_target, pred_full) in enumerate(examples, start=1):
        print(f"[{i}]")
        print(f"query: {query}")
        print(f"tool: {tool}")
        print(f"gold: {gold}")
        print(f"pred_target: {pred_target}")
        print(f"pred_full: {pred_full}")
        print("")


if __name__ == "__main__":
    main()
