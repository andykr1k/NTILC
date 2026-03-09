"""Prompt-only baseline for single-shot tool calling."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from evaluation.metrics import collapse_raw_tool_records


BASELINE_SYSTEM_PROMPT = (
    "You are a Linux tool-calling model.\n"
    "Select exactly one tool from the registry and produce exactly one full shell command.\n"
    "Return strict JSON only with exactly these keys: "
    '{"tool":"<registry tool id>","command":"<full command>"}.\n'
    "Do not add markdown, explanations, or trailing text."
)


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
    registry_block = "\n".join(registry_lines)
    return (
        f"{BASELINE_SYSTEM_PROMPT}\n\n"
        "Registry:\n"
        f"{registry_block}\n"
    )


def build_prompt(static_prompt: str, query: str) -> str:
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

    allowed_keys = {"tool", "command"}
    if set(payload.keys()) != allowed_keys:
        return {
            "parse_ok": False,
            "tool": "",
            "command": "",
            "payload": payload,
            "error": "unexpected_keys",
        }

    tool = str(payload.get("tool", "")).strip()
    command = str(payload.get("command", "")).strip()
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


class PromptBaselineModel:
    """Single-shot full-registry prompt baseline."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        static_prompt: str,
        static_prompt_tokens: int,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.static_prompt = static_prompt
        self.static_prompt_tokens = int(static_prompt_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        raw_rows: Iterable[Mapping[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device: Optional[str] = None,
    ) -> "PromptBaselineModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only in ML env
            raise ImportError(
                "Prompt baseline requires `torch` and `transformers` in the active Python environment."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        static_prompt = build_static_registry_prompt(raw_rows)
        static_prompt_tokens = len(tokenizer.encode(static_prompt, add_special_tokens=False))
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and model_max_length > 0 and static_prompt_tokens >= model_max_length:
            raise ValueError(
                "Baseline registry prompt exceeds tokenizer context length: "
                f"{static_prompt_tokens} >= {model_max_length}."
            )

        target_device = torch.device(device) if device is not None else None
        if target_device is not None and target_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is not available: {device}")

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if target_device is not None:
            if target_device.type == "cuda":
                model_kwargs["device_map"] = {"": str(target_device)}
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if target_device is not None and target_device.type != "cuda":
            model = model.to(target_device)
        model.eval()

        return cls(
            tokenizer=tokenizer,
            model=model,
            static_prompt=static_prompt,
            static_prompt_tokens=static_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

    def generate(self, query: str) -> Dict[str, Any]:
        prompt = build_prompt(self.static_prompt, query)
        encoded_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and model_max_length > 0 and len(encoded_ids) >= model_max_length:
            raise ValueError(
                "Full baseline prompt exceeds tokenizer context length: "
                f"{len(encoded_ids)} >= {model_max_length}."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
        with _no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-6),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = parse_strict_json_tool_call(raw_text)
        return {
            "raw_text": raw_text,
            "strict_json_parse": bool(parsed["parse_ok"]),
            "predicted_tool": str(parsed["tool"]),
            "predicted_command": str(parsed["command"]),
            "parse_error": str(parsed["error"]),
            "payload": parsed["payload"],
        }


def _no_grad():  # pragma: no cover - tiny wrapper around optional torch import
    import torch

    return torch.no_grad()
