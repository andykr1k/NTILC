#!/usr/bin/env python3
"""Minimal single-file NTILC test inference loop with trace metrics."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional
from xml.sax.saxutils import escape

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from models.software_layer import DispatchResult
from orchestrator.blocks import (
    build_dispatch_block as build_orchestrator_dispatch_block,
    build_plan_block as build_orchestrator_plan_block,
    build_response_block as build_orchestrator_response_block,
)
from orchestrator.generation.model import QwenOrchestratorModel
from orchestrator.planning import (
    PlanAction,
    actions_to_instruction_list,
    enforce_atomic_actions,
    normalize_actions_to_natural_language,
    parse_plan_block,
    salvage_plan_actions,
)
from orchestrator.protocol import register_protocol_tokens

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


DEFAULT_RETRIEVAL_CKPT = Path("checkpoints/cluster_retrieval/best_model.pt")
DEFAULT_LORA_ADAPTER = Path("checkpoints/lora_nl_command_full")
DEFAULT_RAW_TOOLS_JSON = Path("data/man/raw_ai.json")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def xml_block(tag: str, text: str) -> str:
    payload = str(text).strip()
    return f"<{tag}><len:{len(payload)}>{escape(payload)}</len></{tag}>"


def round_ms(value: float) -> float:
    return round(float(value), 3)


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
            },
        )
        one_line = str(row.get("one_line", "") or "").strip()
        invocation = str(row.get("invocation", "") or "").strip()

        if len(one_line) > len(current["one_line"]):
            current["one_line"] = one_line
        if len(invocation) > len(current["invocation"]):
            current["invocation"] = invocation

        collapsed[name] = current

    return [collapsed[name] for name in sorted(collapsed.keys())]


def build_registry_lines(raw_rows: Iterable[Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    for row in collapse_raw_tool_records(raw_rows):
        name = str(row.get("name", "")).strip()
        one_line = " ".join(str(row.get("one_line", "")).split())
        invocation = " ".join(str(row.get("invocation", "")).split())
        if not name:
            continue

        line = f"- {name}"
        if one_line:
            line += f": {one_line}"
        if invocation:
            line += f" | usage: {invocation}"
        lines.append(line)
    return lines


def build_static_registry_prompt(raw_rows: Iterable[Mapping[str, Any]]) -> str:
    registry_lines = build_registry_lines(raw_rows)
    if not registry_lines:
        raise ValueError("Tool registry is empty.")
    return (
        "You are a Linux tool-calling model.\n"
        "Select exactly one tool from the registry and produce exactly one full shell command.\n"
        'Return strict JSON only with exactly these keys: {"tool":"<registry tool id>","command":"<full command>"}.\n'
        "Do not add markdown, explanations, or trailing text.\n\n"
        "Registry:\n"
        f"{chr(10).join(registry_lines)}\n"
    )


def build_prompt_baseline_prompt(static_prompt: str, query: str) -> str:
    return (
        f"{str(static_prompt).rstrip()}\n\n"
        f"User request: {str(query).strip()}\n"
        "JSON:"
    )


def parse_strict_json_tool_call(text: str) -> Dict[str, Any]:
    stripped = str(text).strip()
    if not stripped:
        return {
            "parse_ok": False,
            "tool": "",
            "command": "",
            "payload": None,
            "error": "empty_output",
        }

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return {
            "parse_ok": False,
            "tool": "",
            "command": "",
            "payload": None,
            "error": f"json_decode_error:{exc.msg}",
        }

    if not isinstance(payload, dict):
        return {
            "parse_ok": False,
            "tool": "",
            "command": "",
            "payload": payload,
            "error": "payload_not_object",
        }

    tool = str(payload.get("tool", "")).strip()
    command = str(payload.get("command", "")).strip()
    if set(payload.keys()) != {"tool", "command"}:
        return {
            "parse_ok": False,
            "tool": tool,
            "command": command,
            "payload": payload,
            "error": "unexpected_keys",
        }
    if not tool or not command:
        return {
            "parse_ok": False,
            "tool": tool,
            "command": command,
            "payload": payload,
            "error": "missing_tool_or_command",
        }

    return {
        "parse_ok": True,
        "tool": tool,
        "command": command,
        "payload": payload,
        "error": "",
    }


def split_atomic_actions(text: str) -> List[str]:
    request = " ".join(str(text).split())
    if not request:
        return []

    separators = [
        r"\band then\b",
        r"\bthen\b",
        r"\bafter that\b",
        r"\bafterwards\b",
        r";",
        r"\n+",
    ]
    pieces = [request]
    for pattern in separators:
        next_pieces: List[str] = []
        for piece in pieces:
            split_parts = re.split(pattern, piece, flags=re.IGNORECASE)
            next_pieces.extend(split_parts)
        pieces = next_pieces

    actions: List[str] = []
    for piece in pieces:
        chunk = " ".join(piece.split()).strip(" ,")
        if not chunk:
            continue
        if " and " in chunk.lower():
            subparts = re.split(r"\band\b", chunk, flags=re.IGNORECASE)
            cleaned = [" ".join(part.split()).strip(" ,") for part in subparts]
            actions.extend([part for part in cleaned if part])
        else:
            actions.append(chunk)

    return actions or [request]


class QueryEncoder(nn.Module):
    def __init__(
        self,
        base_model: str,
        output_dim: int,
        dropout: float,
        dtype_name: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        dtype = resolve_dtype(dtype_name, device)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder = AutoModel.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        config = self.encoder.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and getattr(config, "text_config", None) is not None:
            hidden_size = getattr(config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden size for {base_model}")

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        ).to(dtype)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        ).to(dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        weights = self.attention_pool(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (hidden * weights).sum(dim=1)
        return F.normalize(self.projection(pooled), p=2, dim=1)


class TestInference:
    def __init__(
        self,
        query_encoder_path: Path = DEFAULT_RETRIEVAL_CKPT,
        lora_adapter_path: Optional[Path] = DEFAULT_LORA_ADAPTER,
        raw_tools_path: Path = DEFAULT_RAW_TOOLS_JSON,
        qwen_model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        baseline_device: Optional[str] = None,
        max_seq_len: int = 512,
        max_new_tokens: int = 96,
        baseline_max_new_tokens: int = 128,
        max_plan_actions: int = 8,
        plan_max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tool_timeout_seconds: int = 20,
    ) -> None:
        self.device = resolve_device(device)
        self.baseline_device = resolve_device(baseline_device) if baseline_device else self.device
        self.query_encoder_path = Path(query_encoder_path)
        self.lora_adapter_path = Path(lora_adapter_path) if lora_adapter_path else None
        self.raw_tools_path = Path(raw_tools_path)
        self.max_seq_len = int(max_seq_len)
        self.max_new_tokens = int(max_new_tokens)
        self.baseline_max_new_tokens = int(baseline_max_new_tokens)
        self.max_plan_actions = int(max_plan_actions)
        self.plan_max_new_tokens = int(plan_max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.tool_timeout_seconds = int(tool_timeout_seconds)

        self.query_encoder, self.cluster_centroids, self.tool_names = self.load_retrieval_model()
        self.base_model_name_or_path = self.resolve_base_model_name(qwen_model_name_or_path)
        self.tokenizer, self.command_model = self.load_generation_model(
            base_model=self.base_model_name_or_path,
            adapter_path=self.lora_adapter_path,
            device=self.device,
            register_protocol=True,
        )
        self.qwen_orchestrator_model = self.build_orchestrator_generation_model(
            tokenizer=self.tokenizer,
            model=self.command_model,
        )
        self.baseline_tokenizer, self.baseline_model = self.load_generation_model(
            base_model=self.base_model_name_or_path,
            adapter_path=None,
            device=self.baseline_device,
        )
        self.raw_tool_rows = self.load_raw_tool_rows()
        self.registry_lines = build_registry_lines(self.raw_tool_rows)
        self.baseline_static_prompt = build_static_registry_prompt(self.raw_tool_rows)
        self.baseline_static_prompt_tokens = self.count_tokens(
            self.baseline_static_prompt,
            tokenizer=self.baseline_tokenizer,
        )

    def load_retrieval_model(self) -> tuple[QueryEncoder, torch.Tensor, List[str]]:
        checkpoint = torch.load(self.query_encoder_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        cluster_centroids = checkpoint["cluster_centroids"]
        if not isinstance(cluster_centroids, torch.Tensor):
            cluster_centroids = torch.tensor(cluster_centroids, dtype=torch.float32)

        dtype_name = str(config.get("torch_dtype", "float32"))
        model_dtype = resolve_dtype(dtype_name, self.device)
        encoder = QueryEncoder(
            base_model=str(config.get("encoder_model", "Qwen/Qwen3.5-9B")),
            output_dim=int(config.get("projection_dim", cluster_centroids.shape[1])),
            dropout=float(config.get("dropout", 0.15)),
            dtype_name=dtype_name,
            device=self.device,
        )
        encoder.load_state_dict(checkpoint["query_encoder_state_dict"])
        encoder = encoder.to(self.device).to(model_dtype)
        encoder.eval()

        tool_names = checkpoint.get("tool_names", [])
        if not isinstance(tool_names, list):
            tool_names = []
        if len(tool_names) < int(cluster_centroids.shape[0]):
            tool_names = list(tool_names) + [
                f"cluster_{i}" for i in range(len(tool_names), int(cluster_centroids.shape[0]))
            ]

        centroids = F.normalize(cluster_centroids.float(), p=2, dim=1).to(self.device).to(model_dtype)
        return encoder, centroids, tool_names

    def resolve_base_model_name(self, explicit_base_model: Optional[str]) -> Optional[str]:
        base_model = explicit_base_model
        if base_model is None and self.lora_adapter_path is not None:
            config_path = self.lora_adapter_path / "adapter_config.json"
            if config_path.exists():
                base_model = str(read_json(config_path).get("base_model_name_or_path", "")).strip() or None
        return base_model

    def load_generation_model(
        self,
        base_model: Optional[str],
        adapter_path: Optional[Path],
        device: torch.device,
        register_protocol: bool = False,
    ) -> tuple[Optional[Any], Optional[Any]]:
        if base_model is None:
            return None, None

        tokenizer_source = (
            str(adapter_path)
            if adapter_path and (adapter_path / "tokenizer_config.json").exists()
            else base_model
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        added_protocol_tokens = register_protocol_tokens(tokenizer) if register_protocol else 0

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device.type == "cuda":
            model_kwargs["device_map"] = {"": str(device)}
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model: Any = base
        if adapter_path is not None:
            if PeftModel is None:
                raise ImportError("`peft` is required to load the LoRA adapter.")
            model = PeftModel.from_pretrained(base, str(adapter_path))

        if added_protocol_tokens > 0:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
            elif hasattr(base, "resize_token_embeddings"):
                base.resize_token_embeddings(len(tokenizer))

        if device.type != "cuda":
            model = model.to(device)
        model.eval()
        return tokenizer, model

    def build_orchestrator_generation_model(
        self,
        tokenizer: Optional[Any],
        model: Optional[Any],
    ) -> Optional[QwenOrchestratorModel]:
        if tokenizer is None or model is None:
            return None
        return QwenOrchestratorModel(
            tokenizer=tokenizer,
            model=model,
            mode="full",
        )

    def load_raw_tool_rows(self) -> List[Dict[str, Any]]:
        path = self.raw_tools_path
        if not path.exists():
            raise FileNotFoundError(f"Raw tool registry not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        return [row for row in payload if isinstance(row, dict)]

    def count_tokens(self, text: str, tokenizer: Optional[Any] = None) -> int:
        tokenizer = tokenizer or self.tokenizer or self.query_encoder.tokenizer
        if tokenizer is None or not str(text).strip():
            return 0
        encoded = tokenizer(str(text), add_special_tokens=False)
        return int(len(encoded["input_ids"]))

    def stage_metrics(
        self,
        elapsed_ms: float,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        block_text: str = "",
        extra: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "elapsed_ms": round_ms(elapsed_ms),
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "total_model_tokens": int(prompt_tokens + generated_tokens),
            "block_tokens": self.count_tokens(block_text, tokenizer=tokenizer),
        }
        if extra:
            metrics.update(extra)
        return metrics

    def build_plan_block(self, actions: List[str]) -> str:
        return build_orchestrator_plan_block(actions)

    def split_command_parts(self, command: str) -> List[Dict[str, str]]:
        text = str(command).strip()
        if not text:
            return []
        try:
            parts = shlex.split(text, posix=False)
        except ValueError:
            parts = text.split()
        if len(parts) <= 1:
            return []

        payload: List[Dict[str, str]] = []
        for value in parts[1:]:
            tag = "opt" if str(value).startswith("-") else "arg"
            payload.append({"tag": tag, "value": str(value)})
        return payload

    def build_dispatch_block(self, tool: str, command: str) -> str:
        lines = ["<dispatch>", f"  {xml_block('tool', tool)}"]
        for item in self.split_command_parts(command):
            lines.append(f"  {xml_block(item['tag'], item['value'])}")
        lines.append("</dispatch>")
        return "\n".join(lines)

    def build_response_block(self, tool: str, ok: bool, text: str, retry: bool) -> str:
        return "\n".join(
            [
                "<response>",
                f"  {xml_block('tool', tool)}",
                f"  {xml_block('status', 'ok' if ok else 'fail')}",
                f"  {xml_block('text', text)}",
                f"  {xml_block('retry', str(retry).lower())}",
                "</response>",
            ]
        )

    def generate_with_model(
        self,
        tokenizer: Optional[Any],
        model: Optional[Any],
        prompt: str,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        if tokenizer is None or model is None:
            return {
                "text": "",
                "metrics": self.stage_metrics(0.0, block_text="", tokenizer=tokenizer),
            }

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(next(model.parameters()).device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        start = perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-6),
                top_p=self.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (perf_counter() - start) * 1000.0

        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "metrics": self.stage_metrics(
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                generated_tokens=int(generated_ids.shape[0]),
                block_text=text,
                tokenizer=tokenizer,
            ),
        }

    def generate_text(self, prompt: str) -> Dict[str, Any]:
        return self.generate_with_model(
            tokenizer=self.tokenizer,
            model=self.command_model,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
        )

    def generate_baseline_text(self, prompt: str) -> Dict[str, Any]:
        return self.generate_with_model(
            tokenizer=self.baseline_tokenizer,
            model=self.baseline_model,
            prompt=prompt,
            max_new_tokens=self.baseline_max_new_tokens,
        )

    def generate_plan_block(self, initial_query: str) -> Dict[str, Any]:
        request = str(initial_query).strip()
        prompt = ""
        prompt_tokens = 0
        raw_text = ""
        raw_token_ids: List[int] = []
        parsed_actions: List[PlanAction] = []

        start = perf_counter()
        if self.qwen_orchestrator_model is not None:
            payload = self.qwen_orchestrator_model.generate_plan_actions(
                request=request,
                max_actions=self.max_plan_actions,
                max_new_tokens=self.plan_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            prompt = str(payload.get("prompt", "")).strip()
            prompt_tokens = self.count_tokens(prompt, tokenizer=self.tokenizer)
            raw_text = str(payload.get("raw_text", "")).strip()
            raw_token_ids = list(payload.get("raw_token_ids", []) or [])
            action_texts = [
                str(action).strip()
                for action in payload.get("actions", [])
                if str(action).strip()
            ]
            if action_texts:
                parsed_actions = [
                    PlanAction(id=idx, instruction=instruction)
                    for idx, instruction in enumerate(action_texts, start=1)
                ]
            raw_plan_block = str(payload.get("plan_block", "")).strip()
        else:
            raw_plan_block = self.build_plan_block([request] if request else [])
        elapsed_ms = (perf_counter() - start) * 1000.0

        if not parsed_actions:
            try:
                parsed_actions = parse_plan_block(raw_plan_block)
            except ValueError:
                parsed_actions = salvage_plan_actions(raw_plan_block, fallback_request=request)

        normalized_actions = normalize_actions_to_natural_language(
            parsed_actions,
            fallback_request=request,
        )
        atomic_actions = enforce_atomic_actions(normalized_actions)
        if not atomic_actions:
            atomic_actions = salvage_plan_actions(request, fallback_request=request)
        if not atomic_actions and request:
            atomic_actions = [PlanAction(id=1, instruction=request)]

        actions = actions_to_instruction_list(atomic_actions)
        plan_block = self.build_plan_block(actions)
        return {
            "actions": actions,
            "plan_block": plan_block,
            "prompt": prompt,
            "raw_output": raw_text,
            "metrics": self.stage_metrics(
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(raw_token_ids),
                block_text=plan_block,
                extra={"action_count": len(actions)},
            ),
        }

    def retrieve_clusters(self, action_text: str, top_k: int = 3) -> Dict[str, Any]:
        start = perf_counter()
        encoded = self.query_encoder.tokenizer(
            action_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        query_tokens = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            query_embedding = self.query_encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            scores = torch.matmul(query_embedding, self.cluster_centroids.T)
            top_scores, top_ids = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        elapsed_ms = (perf_counter() - start) * 1000.0

        candidates: List[Dict[str, Any]] = []
        for cluster_id, score in zip(top_ids[0].tolist(), top_scores[0].tolist()):
            candidates.append(
                {
                    "cluster_id": int(cluster_id),
                    "tool_name": self.tool_names[int(cluster_id)],
                    "score": float(score),
                }
            )

        return {
            "candidates": candidates,
            "metrics": self.stage_metrics(
                elapsed_ms=elapsed_ms,
                block_text="",
                extra={
                    "query_tokens": query_tokens,
                    "candidate_count": len(candidates),
                },
            ),
        }

    def retreive_clusters(self, query_embedding: str, top_k: int = 3) -> Dict[str, Any]:
        return self.retrieve_clusters(query_embedding, top_k=top_k)

    def tool_mapping(self, cluster_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return cluster_results

    def normalize_command(self, tool: str, text: str) -> str:
        stripped = str(text).strip()
        first_line = stripped.splitlines()[0].strip() if stripped else ""
        if not first_line:
            return tool
        parts = first_line.split()
        if not parts:
            return tool
        if parts[0] == tool:
            return first_line
        if len(parts) == 1:
            return f"{tool} {parts[0]}".strip()
        return f"{tool} {' '.join(parts[1:])}".strip()

    def generate_dispatch_block(self, plan: Dict[str, Any], tool_result: Dict[str, Any]) -> Dict[str, Any]:
        tool = str(tool_result["tool_name"]).strip()
        prior_step_summaries = list(plan.get("prior_step_summaries", []) or [])
        prompt = ""
        prompt_tokens = 0
        raw_text = ""
        raw_token_ids: List[int] = []

        start = perf_counter()
        if self.qwen_orchestrator_model is not None:
            generated = self.qwen_orchestrator_model.generate_dispatch_arguments(
                query=str(plan["request"]),
                tool=tool,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                current_action=str(plan["action"]),
                prior_step_summaries=prior_step_summaries,
            )
            prompt = str(generated.get("prompt", "")).strip()
            prompt_tokens = self.count_tokens(prompt, tokenizer=self.tokenizer)
            generated_text = str(generated.get("generated_text", "")).strip()
            dispatch_arguments = dict(generated.get("arguments", {}) or {})
            raw_text = str(generated.get("raw_text", "")).strip()
            raw_token_ids = list(generated.get("raw_token_ids", []) or [])
            command = str(dispatch_arguments.get("command", "")).strip()
        else:
            generated_text = tool
            command = tool
            dispatch_arguments = {"command": command, "query": str(plan["request"])}
        elapsed_ms = (perf_counter() - start) * 1000.0

        if not command:
            command = tool
        dispatch_arguments["command"] = command
        dispatch_arguments.setdefault("query", str(plan["request"]))
        dispatch_block = build_orchestrator_dispatch_block(tool=tool, arguments=dispatch_arguments)

        return {
            "tool": tool,
            "cluster_id": int(tool_result["cluster_id"]),
            "generated_text": generated_text,
            "command": command,
            "dispatch_arguments": dispatch_arguments,
            "prompt": prompt,
            "raw_output": raw_text,
            "dispatch_block": dispatch_block,
            "parts": self.split_command_parts(command),
            "metrics": self.stage_metrics(
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(raw_token_ids),
                block_text=dispatch_block,
            ),
        }

    def execute_dispatch_block(self, dispatch_block: Dict[str, Any], execute: bool = False) -> Dict[str, Any]:
        command = str(dispatch_block["command"]).strip()
        start = perf_counter()

        if not command:
            elapsed_ms = (perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "executed": False,
                "command": "",
                "stdout": "",
                "stderr": "empty command",
                "returncode": None,
                "metrics": self.stage_metrics(
                    elapsed_ms=elapsed_ms,
                    block_text="",
                    extra={"stdout_tokens": 0, "stderr_tokens": self.count_tokens("empty command")},
                ),
            }

        if not execute:
            elapsed_ms = (perf_counter() - start) * 1000.0
            return {
                "ok": True,
                "executed": False,
                "command": command,
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "metrics": self.stage_metrics(elapsed_ms=elapsed_ms, block_text=""),
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.tool_timeout_seconds,
            )
            elapsed_ms = (perf_counter() - start) * 1000.0
            return {
                "ok": result.returncode == 0,
                "executed": True,
                "command": command,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": int(result.returncode),
                "metrics": self.stage_metrics(
                    elapsed_ms=elapsed_ms,
                    block_text="",
                    extra={
                        "stdout_tokens": self.count_tokens(result.stdout.strip()),
                        "stderr_tokens": self.count_tokens(result.stderr.strip()),
                    },
                ),
            }
        except subprocess.TimeoutExpired:
            elapsed_ms = (perf_counter() - start) * 1000.0
            stderr = f"Timed out after {self.tool_timeout_seconds}s"
            return {
                "ok": False,
                "executed": True,
                "command": command,
                "stdout": "",
                "stderr": stderr,
                "returncode": None,
                "metrics": self.stage_metrics(
                    elapsed_ms=elapsed_ms,
                    block_text="",
                    extra={"stdout_tokens": 0, "stderr_tokens": self.count_tokens(stderr)},
                ),
            }

    def generate_response_block(
        self,
        tool: str,
        execution_results: Dict[str, Any],
        retry: bool,
    ) -> Dict[str, Any]:
        start = perf_counter()
        if not execution_results["ok"]:
            text = execution_results["stderr"] or "command failed"
        elif not execution_results["executed"]:
            text = f"dry run: {execution_results['command']}"
        else:
            text = execution_results["stdout"] or "command succeeded"
        dispatch_result = DispatchResult(
            ok=bool(execution_results["ok"]),
            tool=str(tool),
            arguments={"command": str(execution_results.get("command", "")).strip()},
            result=(
                {
                    "returncode": execution_results.get("returncode"),
                    "stdout": execution_results.get("stdout", ""),
                    "stderr": execution_results.get("stderr", ""),
                }
                if execution_results["executed"]
                else None
            ),
            errors=[] if execution_results["ok"] else [text],
            executed=bool(execution_results["executed"]),
        )
        response_block = build_orchestrator_response_block(
            tool=str(tool),
            dispatch_result=dispatch_result,
            retry=retry,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        return {
            "ok": bool(execution_results["ok"]),
            "done": bool(execution_results["ok"]),
            "text": text,
            "tool": tool,
            "response_block": response_block,
            "metrics": self.stage_metrics(elapsed_ms=elapsed_ms, block_text=response_block),
        }

    def check_done(self, response_block: Dict[str, Any]) -> bool:
        return bool(response_block["done"])

    def run_prompt_baseline(self, request: str, execute_tools: bool = False) -> Dict[str, Any]:
        total_start = perf_counter()
        prompt = build_prompt_baseline_prompt(self.baseline_static_prompt, request)
        generation = self.generate_baseline_text(prompt)
        parsed = parse_strict_json_tool_call(generation["text"])

        tool = str(parsed["tool"]).strip()
        command = str(parsed["command"]).strip()

        dispatch_start = perf_counter()
        dispatch_block = self.build_dispatch_block(tool=tool, command=command)
        dispatch_elapsed_ms = (perf_counter() - dispatch_start) * 1000.0
        dispatch = {
            "tool": tool,
            "command": command,
            "dispatch_block": dispatch_block,
            "parts": self.split_command_parts(command),
            "parse": parsed,
            "metrics": self.stage_metrics(
                elapsed_ms=dispatch_elapsed_ms,
                block_text=dispatch_block,
                extra={
                    "parse_ok": bool(parsed["parse_ok"]),
                    "parse_error": str(parsed["error"]),
                },
                tokenizer=self.baseline_tokenizer,
            ),
        }

        execution = self.execute_dispatch_block({"command": command}, execute=execute_tools)
        response = self.generate_response_block(
            tool=tool,
            execution_results=execution,
            retry=False,
        )

        total_elapsed_ms = (perf_counter() - total_start) * 1000.0
        success = bool(parsed["parse_ok"]) and bool(execution["ok"])

        return {
            "request": request,
            "prompt": prompt,
            "static_prompt": self.baseline_static_prompt,
            "generation": {
                "raw_output": generation["text"],
                "metrics": generation["metrics"],
            },
            "dispatch": dispatch,
            "execution": execution,
            "response": response,
            "success": success,
            "metrics": {
                "elapsed_ms": round_ms(total_elapsed_ms),
                "prompt_tokens": int(generation["metrics"]["prompt_tokens"]),
                "generated_tokens": int(generation["metrics"]["generated_tokens"]),
                "total_model_tokens": int(generation["metrics"]["total_model_tokens"]),
                "generation_elapsed_ms": float(generation["metrics"]["elapsed_ms"]),
                "dispatch_elapsed_ms": float(dispatch["metrics"]["elapsed_ms"]),
                "execution_elapsed_ms": float(execution["metrics"]["elapsed_ms"]),
                "response_elapsed_ms": float(response["metrics"]["elapsed_ms"]),
                "static_prompt_tokens": int(self.baseline_static_prompt_tokens),
                "registry_tool_count": len(self.registry_lines),
            },
        }

    def build_comparison(self, ntilc: Dict[str, Any], prompt_baseline: Dict[str, Any]) -> Dict[str, Any]:
        def metric_pair(name: str) -> Dict[str, Any]:
            left = float(ntilc["metrics"].get(name, 0.0))
            right = float(prompt_baseline["metrics"].get(name, 0.0))
            return {
                "ntilc": left,
                "prompt_baseline": right,
                "delta": round(left - right, 3),
            }

        ntilc_last_step = ntilc["steps"][-1] if ntilc["steps"] else {}
        ntilc_dispatch = ntilc_last_step.get("dispatch", {})
        baseline_dispatch = prompt_baseline["dispatch"]

        return {
            "success": {
                "ntilc": bool(ntilc["success"]),
                "prompt_baseline": bool(prompt_baseline["success"]),
            },
            "outputs": {
                "tool": {
                    "ntilc": str(ntilc_dispatch.get("tool", "")),
                    "prompt_baseline": str(baseline_dispatch.get("tool", "")),
                },
                "command": {
                    "ntilc": str(ntilc_dispatch.get("command", "")),
                    "prompt_baseline": str(baseline_dispatch.get("command", "")),
                },
            },
            "metrics": {
                "elapsed_ms": metric_pair("elapsed_ms"),
                "prompt_tokens": metric_pair("prompt_tokens"),
                "generated_tokens": metric_pair("generated_tokens"),
                "total_model_tokens": metric_pair("total_model_tokens"),
            },
        }

    def compact_prior_step_summaries(self, steps: List[Dict[str, Any]], tail_k: int = 6) -> List[str]:
        summaries: List[str] = []
        for step in steps[-max(0, int(tail_k)) :]:
            action_id = step.get("action_id", "?")
            action_text = str(step.get("action", "")).strip()
            command = str(step.get("dispatch", {}).get("command", "")).strip()
            status = "ok" if step.get("execution", {}).get("ok") else "fail"
            summaries.append(f"action#{action_id} {action_text} -> {command} [{status}]")
        return summaries

    def run_comparison(
        self,
        request: str,
        execute_tools: bool = False,
        top_k_candidates: int = 3,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        ntilc = self.run(
            request=request,
            execute_tools=execute_tools,
            top_k_candidates=top_k_candidates,
            max_retries=max_retries,
        )
        prompt_baseline = self.run_prompt_baseline(
            request=request,
            execute_tools=execute_tools,
        )
        return {
            "request": request,
            "ntilc": ntilc,
            "prompt_baseline": prompt_baseline,
            "comparison": self.build_comparison(ntilc, prompt_baseline),
        }

    def summarize_metrics(
        self,
        plan_metrics: Dict[str, Any],
        retrieval_metrics: Dict[str, Any],
        steps: List[Dict[str, Any]],
        total_elapsed_ms: float,
    ) -> Dict[str, Any]:
        prompt_tokens = int(plan_metrics["prompt_tokens"]) + sum(
            int(step["dispatch"]["metrics"]["prompt_tokens"]) for step in steps
        )
        generated_tokens = int(plan_metrics["generated_tokens"]) + sum(
            int(step["dispatch"]["metrics"]["generated_tokens"]) for step in steps
        )
        return {
            "elapsed_ms": round_ms(total_elapsed_ms),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_model_tokens": prompt_tokens + generated_tokens,
            "plan_elapsed_ms": float(plan_metrics["elapsed_ms"]),
            "retrieval_elapsed_ms": float(retrieval_metrics["elapsed_ms"]),
            "dispatch_elapsed_ms": round_ms(
                sum(float(step["dispatch"]["metrics"]["elapsed_ms"]) for step in steps)
            ),
            "execution_elapsed_ms": round_ms(
                sum(float(step["execution"]["metrics"]["elapsed_ms"]) for step in steps)
            ),
            "response_elapsed_ms": round_ms(
                sum(float(step["response"]["metrics"]["elapsed_ms"]) for step in steps)
            ),
            "attempt_count": len(steps),
        }

    def run(self, request: str, execute_tools: bool = False, top_k_candidates: int = 3, max_retries: int = 2) -> Dict[str, Any]:
        total_start = perf_counter()

        plan = self.generate_plan_block(request)
        result: Dict[str, Any] = {
            "request": request,
            "plan": plan,
            "retrievals": [],
            "steps": [],
            "success": False,
            "metrics": {},
        }

        retrieval_elapsed_ms = 0.0
        all_actions_succeeded = True

        for action_id, action in enumerate(plan["actions"], start=1):
            retrieval = self.retrieve_clusters(action, top_k=top_k_candidates)
            candidates = self.tool_mapping(retrieval["candidates"])
            retrieval_elapsed_ms += float(retrieval["metrics"]["elapsed_ms"])

            result["retrievals"].append(
                {
                    "action_id": action_id,
                    "action": action,
                    "candidates": candidates,
                    "metrics": retrieval["metrics"],
                }
            )

            action_succeeded = False
            for attempt, candidate in enumerate(candidates[: max_retries + 1], start=1):
                dispatch = self.generate_dispatch_block(
                    plan={
                        "request": request,
                        "action": action,
                        "prior_step_summaries": self.compact_prior_step_summaries(result["steps"]),
                    },
                    tool_result=candidate,
                )
                execution = self.execute_dispatch_block(dispatch, execute=execute_tools)
                retry = attempt < min(len(candidates), max_retries + 1) and not execution["ok"]
                response = self.generate_response_block(
                    tool=str(candidate["tool_name"]),
                    execution_results=execution,
                    retry=retry,
                )

                step_metrics = {
                    "elapsed_ms": round_ms(
                        float(dispatch["metrics"]["elapsed_ms"])
                        + float(execution["metrics"]["elapsed_ms"])
                        + float(response["metrics"]["elapsed_ms"])
                    ),
                    "prompt_tokens": int(dispatch["metrics"]["prompt_tokens"]),
                    "generated_tokens": int(dispatch["metrics"]["generated_tokens"]),
                    "total_model_tokens": int(dispatch["metrics"]["total_model_tokens"]),
                }

                result["steps"].append(
                    {
                        "action_id": action_id,
                        "action": action,
                        "attempt": attempt,
                        "candidate": candidate,
                        "dispatch": dispatch,
                        "execution": execution,
                        "response": response,
                        "metrics": step_metrics,
                    }
                )

                if self.check_done(response):
                    action_succeeded = True
                    break

            if not action_succeeded:
                all_actions_succeeded = False
                break

        result["success"] = all_actions_succeeded and len(plan["actions"]) > 0
        result["retrieval"] = result["retrievals"][0] if result["retrievals"] else {
            "action_id": 1,
            "action": "",
            "candidates": [],
            "metrics": self.stage_metrics(0.0),
        }

        total_elapsed_ms = (perf_counter() - total_start) * 1000.0
        result["metrics"] = self.summarize_metrics(
            plan_metrics=plan["metrics"],
            retrieval_metrics={"elapsed_ms": retrieval_elapsed_ms},
            steps=result["steps"],
            total_elapsed_ms=total_elapsed_ms,
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal NTILC test inference loop.")
    parser.add_argument("--request", type=str, required=True)
    parser.add_argument("--query-encoder-path", type=Path, default=DEFAULT_RETRIEVAL_CKPT)
    parser.add_argument("--lora-adapter-path", type=Path, default=DEFAULT_LORA_ADAPTER)
    parser.add_argument("--raw-tools-path", type=Path, default=DEFAULT_RAW_TOOLS_JSON)
    parser.add_argument("--qwen-model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--baseline-device", type=str, default=None)
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--execute-tools", action="store_true")
    parser.add_argument("--baseline-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-plan-actions", type=int, default=8)
    parser.add_argument("--plan-max-new-tokens", type=int, default=256)
    parser.add_argument("--compare-prompt-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = TestInference(
        query_encoder_path=args.query_encoder_path,
        lora_adapter_path=args.lora_adapter_path,
        raw_tools_path=args.raw_tools_path,
        qwen_model_name_or_path=args.qwen_model,
        device=args.device,
        baseline_device=args.baseline_device,
        baseline_max_new_tokens=args.baseline_max_new_tokens,
        max_plan_actions=args.max_plan_actions,
        plan_max_new_tokens=args.plan_max_new_tokens,
    )
    if args.compare_prompt_baseline:
        result = inference.run_comparison(
            request=args.request,
            execute_tools=args.execute_tools,
            top_k_candidates=args.top_k_candidates,
            max_retries=args.max_retries,
        )
    else:
        result = inference.run(
            request=args.request,
            execute_tools=args.execute_tools,
            top_k_candidates=args.top_k_candidates,
            max_retries=args.max_retries,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
