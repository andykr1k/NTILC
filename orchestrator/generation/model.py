"""Qwen + optional LoRA generation runtime for planning and token-protocol dispatch."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orchestrator.blocks import build_plan_block
from orchestrator.protocol import (
    extract_dispatch_arguments_from_ids,
    extract_plan_actions_from_ids,
    register_protocol_tokens,
    resolve_protocol_token_ids,
)

from .prompting import (
    build_dispatch_model_input,
    build_plan_model_input,
    extract_plan_block,
    normalize_command_for_tool,
    safe_first_line,
)

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency at runtime.
    PeftModel = None


def _resolve_torch_dtype(dtype_name: Optional[str]) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name and dtype_name in mapping:
        return mapping[dtype_name]
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class QwenOrchestratorModel:
    """
    Planner + argument generator (base Qwen, optionally with a LoRA adapter).
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        mode: str = "full",
        max_seq_len: int = 512,
        enforce_selected_tool: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.mode = mode
        self.max_seq_len = int(max_seq_len)
        self.enforce_selected_tool = enforce_selected_tool
        self.protocol_token_ids = resolve_protocol_token_ids(tokenizer, strict=False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        qwen_model_name_or_path: str,
        lora_adapter_path: Optional[str] = None,
        mode: str = "full",
        max_seq_len: int = 512,
        enforce_selected_tool: bool = True,
        device: Optional[str] = None,
    ) -> "QwenOrchestratorModel":
        if mode not in {"full", "tail"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected `full` or `tail`.")

        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        added_protocol_tokens = register_protocol_tokens(tokenizer)

        target_device = torch.device(device) if device is not None else None
        if target_device is not None and target_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is not available: {device}")

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if target_device is not None:
            if target_device.type == "cuda":
                model_kwargs["device_map"] = {"": str(target_device)}
                model_kwargs["torch_dtype"] = _resolve_torch_dtype(None)
            else:
                model_kwargs["torch_dtype"] = torch.float32
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(None)
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            **model_kwargs,
        )

        model: Any
        if lora_adapter_path:
            if PeftModel is None:
                raise ImportError(
                    "LoRA adapter path provided but `peft` is not installed. Install `peft` first."
                )
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            model = base_model

        if target_device is not None and target_device.type != "cuda":
            model = model.to(target_device)

        if added_protocol_tokens > 0:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
            elif hasattr(base_model, "resize_token_embeddings"):
                base_model.resize_token_embeddings(len(tokenizer))

        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            mode=mode,
            max_seq_len=max_seq_len,
            enforce_selected_tool=enforce_selected_tool,
        )

    def _generate_token_ids(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_id: Optional[int] = None,
        include_eos_token: bool = True,
    ) -> List[int]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.device)

        eos_ids: List[int] = []
        if include_eos_token and getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        if stop_token_id is not None and int(stop_token_id) >= 0:
            token_id = int(stop_token_id)
            if token_id not in eos_ids:
                eos_ids.append(token_id)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids if eos_ids else None,
            )

        generated = output_ids[0, encoded["input_ids"].shape[1] :]
        return [int(tok) for tok in generated.tolist()]

    def _decode_ids(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False)).strip()

    def _is_immediate_eos(self, token_ids: Sequence[int]) -> bool:
        ids = [int(tok) for tok in token_ids]
        if not ids:
            return False

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and all(tok == int(eos_token_id) for tok in ids):
            return True

        eos_token = str(getattr(self.tokenizer, "eos_token", "") or "").strip()
        if eos_token:
            decoded = self._decode_ids(ids)
            return bool(decoded) and decoded == eos_token
        return False

    def generate_plan_actions(
        self,
        request: str,
        max_actions: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        prompt = build_plan_model_input(
            tokenizer=self.tokenizer,
            request=request,
            max_actions=max_actions,
        )
        token_ids = self._generate_token_ids(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_id=self.protocol_token_ids.plan_end,
        )

        actions = extract_plan_actions_from_ids(
            tokenizer=self.tokenizer,
            generated_token_ids=token_ids,
            token_ids=self.protocol_token_ids,
        )
        if not actions and self._is_immediate_eos(token_ids):
            token_ids = self._generate_token_ids(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_id=self.protocol_token_ids.plan_end,
                include_eos_token=False,
            )
            actions = extract_plan_actions_from_ids(
                tokenizer=self.tokenizer,
                generated_token_ids=token_ids,
                token_ids=self.protocol_token_ids,
            )
        if actions:
            plan_block = build_plan_block(actions)
            raw_text = self._decode_ids(token_ids)
            return {
                "actions": actions,
                "plan_block": plan_block,
                "prompt": prompt,
                "raw_text": raw_text,
                "raw_token_ids": token_ids,
            }

        raw_text = self._decode_ids(token_ids)
        return {
            "actions": [],
            "plan_block": extract_plan_block(raw_text),
            "prompt": prompt,
            "raw_text": raw_text,
            "raw_token_ids": token_ids,
        }

    def generate_plan(
        self,
        request: str,
        max_actions: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        payload = self.generate_plan_actions(
            request=request,
            max_actions=max_actions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return str(payload.get("plan_block", "")).strip()

    def generate_dispatch_arguments(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action: Optional[str] = None,
        prior_step_summaries: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        prompt = build_dispatch_model_input(
            tokenizer=self.tokenizer,
            query=query,
            tool=tool,
            mode=self.mode,
            current_action=current_action,
            prior_step_summaries=prior_step_summaries,
        )
        token_ids = self._generate_token_ids(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_id=self.protocol_token_ids.dispatch_end,
        )

        arguments = extract_dispatch_arguments_from_ids(
            tokenizer=self.tokenizer,
            generated_token_ids=token_ids,
            token_ids=self.protocol_token_ids,
        )
        if not arguments and self._is_immediate_eos(token_ids):
            token_ids = self._generate_token_ids(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_id=self.protocol_token_ids.dispatch_end,
                include_eos_token=False,
            )
            arguments = extract_dispatch_arguments_from_ids(
                tokenizer=self.tokenizer,
                generated_token_ids=token_ids,
                token_ids=self.protocol_token_ids,
            )

        generated_command_payload = str(arguments.get("command", "")).strip()
        if generated_command_payload:
            normalized_command = normalize_command_for_tool(
                tool=tool,
                generated_text=generated_command_payload,
                mode=self.mode,
                enforce_selected_tool=self.enforce_selected_tool,
            )
            arguments["command"] = normalized_command
        else:
            arguments["command"] = str(tool).strip()

        arguments.setdefault("query", str(query))

        raw_text = self._decode_ids(token_ids)
        return {
            "arguments": arguments,
            "generated_text": generated_command_payload,
            "command": str(arguments.get("command", "")).strip(),
            "prompt": prompt,
            "raw_text": raw_text,
            "raw_token_ids": token_ids,
        }

    def generate(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action: Optional[str] = None,
        prior_step_summaries: Optional[Sequence[str]] = None,
    ) -> str:
        payload = self.generate_dispatch_arguments(
            query=query,
            tool=tool,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            current_action=current_action,
            prior_step_summaries=prior_step_summaries,
        )
        return safe_first_line(str(payload.get("generated_text", "")))

    def generate_command(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
        current_action: Optional[str] = None,
        prior_step_summaries: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        payload = self.generate_dispatch_arguments(
            query=query,
            tool=tool,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            current_action=current_action,
            prior_step_summaries=prior_step_summaries,
        )
        return {
            "generated_text": str(payload.get("generated_text", "")).strip(),
            "command": str(payload.get("command", "")).strip(),
        }
