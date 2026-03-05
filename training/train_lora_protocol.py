#!/usr/bin/env python3
"""
PEFT LoRA training for protocol-format NTILC planner + dispatch generation.

Expected row schema (.json/.jsonl):
- task: "plan" | "dispatch"
- request: original user request
- target: protocol output text (<plan>... or <dispatch>...)
Optional for dispatch:
- tool, action, mode, prior_steps
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from orchestrator.generation.prompting import build_dispatch_prompt, build_plan_prompt
from orchestrator.protocol import register_protocol_tokens


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


def _normalize_prior_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_rows(
    rows: Iterable[Dict[str, Any]],
    allowed_tasks: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = {str(t).strip().lower() for t in allowed_tasks if str(t).strip()}
    if not allowed:
        allowed = {"plan", "dispatch"}

    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        task = str(row.get("task", "")).strip().lower()
        if not task:
            task = "dispatch" if row.get("tool") else "plan"
        if task not in allowed:
            continue

        request = str(row.get("request", row.get("query", ""))).strip()
        target = str(row.get("target", "")).strip()
        if not request or not target:
            continue

        normalized: Dict[str, Any] = {
            "task": task,
            "request": request,
            "target": target,
        }

        if task == "plan":
            num_actions = row.get("num_actions", 8)
            try:
                normalized["num_actions"] = max(1, int(num_actions))
            except (TypeError, ValueError):
                normalized["num_actions"] = 8
            out.append(normalized)
            continue

        tool = str(row.get("tool", "")).strip()
        if not tool:
            continue

        normalized["tool"] = tool
        normalized["action"] = str(row.get("action", request)).strip() or request
        normalized["mode"] = str(row.get("mode", "full")).strip() or "full"
        normalized["prior_steps"] = _normalize_prior_steps(row.get("prior_steps", []))
        out.append(normalized)

    return out


def split_train_val(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    if n_val >= len(rows):
        n_val = max(0, len(rows) - 1)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def _build_prompt(row: Mapping[str, Any]) -> str:
    task = str(row["task"])
    if task == "plan":
        return build_plan_prompt(
            request=str(row["request"]),
            max_actions=int(row.get("num_actions", 8)),
        )

    return build_dispatch_prompt(
        query=str(row["request"]),
        tool=str(row["tool"]),
        mode=str(row.get("mode", "full")),
        current_action=str(row.get("action", row["request"])),
        prior_step_summaries=row.get("prior_steps", []),
    )


@dataclass
class TokenizedSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class ProtocolSFTDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
    ):
        self.samples: List[TokenizedSample] = []
        eos = tokenizer.eos_token or ""

        for row in rows:
            prompt = _build_prompt(row)
            target = str(row["target"]).strip()
            if not target:
                continue

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(target + eos, add_special_tokens=False)
            if not target_ids:
                continue

            if len(target_ids) > max_seq_len:
                continue

            max_prompt_len = max_seq_len - len(target_ids)
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            attention_mask = [1] * len(input_ids)

            self.samples.append(
                TokenizedSample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        s = self.samples[idx]
        return {
            "input_ids": s.input_ids,
            "attention_mask": s.attention_mask,
            "labels": s.labels,
        }


def make_collator(pad_token_id: int):
    def collate_fn(features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PEFT LoRA for protocol-style planner/dispatch generation.")
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        default=[
            Path("data/protocol/planner_protocol.jsonl"),
            Path("data/protocol/dispatch_full_protocol.jsonl"),
        ],
        help="One or more train data files (.json/.jsonl).",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        nargs="*",
        default=None,
        help="Optional validation data files (.json/.jsonl).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="plan,dispatch",
        help="Comma-separated task filter (plan,dispatch).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Base model for LoRA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/lora_protocol"),
        help="Output directory for adapter and tokenizer.",
    )
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument(
        "--modules-to-save",
        type=str,
        default="embed_tokens,lm_head",
        help="Comma-separated non-LoRA modules to save/train (important for newly added protocol tokens).",
    )

    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit loading (requires bitsandbytes).",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Trainer report_to value, e.g. none, wandb.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs (e.g., '0,1').",
    )
    return parser.parse_args()


def _load_many(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(load_rows(path))
    return rows


def main() -> None:
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    is_main_process = rank == 0

    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task_filter = [t.strip().lower() for t in str(args.tasks).split(",") if t.strip()]

    raw_train = _load_many(args.train_data)
    train_rows = normalize_rows(raw_train, allowed_tasks=task_filter)
    if not train_rows:
        raise ValueError(f"No valid training rows found in train-data={args.train_data}")

    if args.val_data:
        raw_val = _load_many(list(args.val_data))
        val_rows = normalize_rows(raw_val, allowed_tasks=task_filter)
    else:
        train_rows, val_rows = split_train_val(
            rows=train_rows,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    if not train_rows:
        raise ValueError("No training rows remain after train/val split.")

    if is_main_process:
        print(f"Train rows: {len(train_rows)}")
        print(f"Val rows:   {len(val_rows)}")
        print(f"Tasks:      {','.join(task_filter)}")
        print(f"Distributed: {distributed} (world_size={world_size})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    added_protocol_tokens = register_protocol_tokens(tokenizer)

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if distributed and args.use_4bit:
        raise ValueError(
            "Distributed LoRA with --use-4bit is not supported in this script. "
            "Run without --use-4bit for multi-GPU training."
        )

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit requested but BitsAndBytesConfig is unavailable. Install bitsandbytes."
            ) from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available() and not distributed:
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if added_protocol_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        if is_main_process:
            print(f"Added protocol tokens: {added_protocol_tokens}")

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    modules_to_save = [m.strip() for m in args.modules_to_save.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save if modules_to_save else None,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = ProtocolSFTDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    val_dataset = ProtocolSFTDataset(
        rows=val_rows,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    if is_main_process:
        print(f"Tokenized train samples: {len(train_dataset)}")
        print(f"Tokenized val samples:   {len(val_dataset)}")

    collator = make_collator(pad_token_id=tokenizer.pad_token_id)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    has_eval_set = len(val_dataset) > 0
    ta_signature = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_signature.parameters.keys())
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "bf16": bf16_ok,
        "fp16": (torch.cuda.is_available() and not bf16_ok),
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": args.seed,
        "save_on_each_node": False,
    }

    if distributed:
        training_kwargs["ddp_find_unused_parameters"] = False
        training_kwargs["local_rank"] = local_rank

    if has_eval_set:
        if "evaluation_strategy" in ta_params:
            training_kwargs["evaluation_strategy"] = "steps"
        if "eval_strategy" in ta_params:
            training_kwargs["eval_strategy"] = "steps"
        if "load_best_model_at_end" in ta_params:
            training_kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in ta_params:
            training_kwargs["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in ta_params:
            training_kwargs["greater_is_better"] = False

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    if is_main_process:
        print(f"Saved protocol LoRA adapter + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
