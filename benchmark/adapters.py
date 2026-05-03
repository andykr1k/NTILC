from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

try:
    import torch
except ImportError:
    torch = None

try:
    import torch.nn.functional as F
except ImportError:
    F = None

try:
    from pydantic import BaseModel, Field, create_model
except ImportError:
    BaseModel = None
    Field = None
    create_model = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Sequence[Any], **_: Any) -> Sequence[Any]:
        return iterable

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import outlines
except ImportError:
    outlines = None

from benchmark.common import (
    DEFAULT_RANKING_LIMIT,
    EmbeddingVariantSpec,
    build_selection_messages,
    estimate_cost_usd,
    normalize_ranked_tools,
    parse_selection_response,
    reciprocal_rank,
    round_float,
    sample_tools_for_query,
    slugify,
    summarize_result_rows,
)

RATE_LIMIT_RETRY_SECONDS = 20
RATE_LIMIT_MAX_RETRIES = 5


def load_checkpoint_bundle_lazy(*args: Any, **kwargs: Any) -> Any:
    from training import load_checkpoint_bundle

    return load_checkpoint_bundle(*args, **kwargs)


def require_torch() -> Any:
    if torch is None or F is None:
        raise ImportError("torch is required for embedding and local Hugging Face selection.")
    return torch


def require_transformers() -> tuple[Any, Any]:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers is required for local Hugging Face structured generation.")
    return AutoModelForCausalLM, AutoTokenizer


def resolve_runtime_device(device_name: str) -> str:
    torch_module = require_torch()
    if device_name == "auto":
        return "auto" if torch_module.cuda.is_available() else "cpu"
    return device_name


def resolve_torch_dtype(dtype_name: str, device_name: str) -> torch.dtype | None:
    torch_module = require_torch()
    mapping = {
        "auto": None,
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    dtype = mapping.get(dtype_name, None)
    if dtype is not None:
        return dtype
    if device_name.startswith("cpu"):
        return torch_module.float32
    return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32


def first_parameter_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def ensure_transformers_output_recorder_compat() -> None:
    try:
        from transformers.utils import generic, output_capturing
    except Exception:
        return
    if hasattr(generic, "OutputRecorder"):
        return
    if hasattr(output_capturing, "OutputRecorder"):
        generic.OutputRecorder = output_capturing.OutputRecorder


def require_outlines() -> Any:
    if outlines is None:
        raise ImportError("outlines is required for local Hugging Face structured generation.")
    return outlines


def require_pydantic() -> tuple[Any, Any, Any]:
    if BaseModel is None or Field is None or create_model is None:
        raise ImportError("pydantic is required for local Hugging Face structured generation.")
    return BaseModel, Field, create_model


def inference_mode(func: Any) -> Any:
    if torch is None:
        return func
    return torch.inference_mode()(func)


def build_prompt_text(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}:\n{content}")
    lines.append("Assistant:\n")
    return "\n\n".join(lines)


def compute_query_token_count(tokenizer: Any, text: str, max_length: int) -> int:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        return int(encoded["input_ids"].shape[-1])
    return int(attention_mask.sum().item())


class SelectionGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


def build_selection_output_model(
    valid_tool_names: Sequence[str],
    *,
    ranking_limit: int,
) -> type[BaseModel]:
    _, pydantic_field, pydantic_create_model = require_pydantic()
    normalized_tool_names = [str(tool_name).strip() for tool_name in valid_tool_names if str(tool_name).strip()]
    if not normalized_tool_names:
        raise ValueError("Selection output model requires at least one valid tool name.")

    literal_type = Literal.__getitem__(tuple(normalized_tool_names))
    max_ranked_tools = min(max(1, ranking_limit), len(normalized_tool_names))
    digest = hashlib.sha1("|".join(normalized_tool_names).encode("utf-8")).hexdigest()[:12]
    return pydantic_create_model(
        f"ToolSelection_{digest}",
        selected_tool=(
            literal_type,
            pydantic_field(..., description="The single best tool for the request."),
        ),
        ranked_tools=(
            List[literal_type],
            pydantic_field(
                ...,
                min_length=1,
                max_length=max_ranked_tools,
                description="Unique tool names ranked best to worst.",
            ),
        ),
        reason=(
            str,
            pydantic_field(default="", description="One short sentence explaining the choice."),
        ),
    )


def build_error_result(
    *,
    adapter_id: str,
    provider: str,
    mode: str,
    model_name: str,
    row: dict[str, Any],
    error_message: str,
    latency_ms: float | None = None,
    raw_response: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "adapter_id": adapter_id,
        "provider": provider,
        "mode": mode,
        "model_name": model_name,
        "example_id": str(row.get("id", "")),
        "query": str(row.get("query", "")),
        "expected_tool": str(row.get("tool", "")),
        "status": "error",
        "selected_tool": None,
        "ranked_tools": [],
        "correct_top1": None,
        "top_3_hit": None,
        "top_5_hit": None,
        "reciprocal_rank": None,
        "latency_ms": round_float(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": None,
        "error_message": error_message,
        "raw_response": raw_response,
    }


def finalize_selection_result(
    *,
    adapter_id: str,
    provider: str,
    mode: str,
    model_name: str,
    row: dict[str, Any],
    selected_tool: str,
    ranked_tools: Sequence[str],
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    raw_response: str | None,
    reason: str | None,
    cost_usd: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_tool = str(row.get("tool", "")).strip()
    ranked_tool_list = [str(tool).strip() for tool in ranked_tools if str(tool).strip()]
    reciprocal_rank_value = reciprocal_rank(ranked_tool_list, expected_tool)
    payload = {
        "adapter_id": adapter_id,
        "provider": provider,
        "mode": mode,
        "model_name": model_name,
        "example_id": str(row.get("id", "")),
        "query": str(row.get("query", "")),
        "expected_tool": expected_tool,
        "status": "ok",
        "selected_tool": selected_tool,
        "ranked_tools": ranked_tool_list,
        "correct_top1": selected_tool == expected_tool,
        "top_3_hit": expected_tool in ranked_tool_list[:3],
        "top_5_hit": expected_tool in ranked_tool_list[:5],
        "reciprocal_rank": round_float(reciprocal_rank_value),
        "latency_ms": round_float(latency_ms),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "reason": reason or "",
        "raw_response": raw_response,
    }
    if extra:
        payload.update(extra)
    return payload


def build_model_summary(
    *,
    adapter_id: str,
    provider: str,
    mode: str,
    model_name: str,
    results: Sequence[dict[str, Any]],
    status: str = "ok",
    error_message: str | None = None,
    results_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_status = status
    effective_error_message = error_message
    metrics = summarize_result_rows(results) if status == "ok" else None
    if (
        effective_status == "ok"
        and metrics is not None
        and metrics.get("total_examples", 0) > 0
        and metrics.get("successful_examples", 0) == 0
    ):
        effective_status = "error"
        if effective_error_message is None:
            first_error = next(
                (str(row.get("error_message", "")).strip() for row in results if str(row.get("error_message", "")).strip()),
                "",
            )
            effective_error_message = first_error or "All benchmark examples failed."
    payload = {
        "adapter_id": adapter_id,
        "provider": provider,
        "mode": mode,
        "model_name": model_name,
        "status": effective_status,
        "error_message": effective_error_message,
        "metrics": metrics,
        "results_path": str(results_path.resolve()) if results_path is not None else "",
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def build_api_selection_tool(
    valid_tool_names: Sequence[str],
    *,
    ranking_limit: int,
) -> dict[str, Any]:
    normalized_tool_names = [str(tool_name).strip() for tool_name in valid_tool_names if str(tool_name).strip()]
    if not normalized_tool_names:
        raise ValueError("Selection API tool requires at least one valid tool name.")
    max_ranked_tools = min(max(1, ranking_limit), len(normalized_tool_names))
    return {
        "name": "select_tool",
        "description": "Select the single best tool for the user request.",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_tool": {
                    "type": "string",
                    "enum": normalized_tool_names,
                    "description": "The single best tool for the request.",
                },
                "ranked_tools": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": normalized_tool_names,
                    },
                    "minItems": 1,
                    "maxItems": max_ranked_tools,
                    "description": "Unique tool names ranked best to worst.",
                },
                "reason": {
                    "type": "string",
                    "description": "One short sentence explaining the choice.",
                },
            },
            "required": ["selected_tool", "ranked_tools"],
        },
    }


def build_loose_api_selection_tool(*, ranking_limit: int) -> dict[str, Any]:
    return {
        "name": "select_tool",
        "description": "Select the single best tool for the user request.",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_tool": {
                    "type": "string",
                    "description": "The single best tool for the request. Must be one of the listed tool names.",
                },
                "ranked_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": max(1, ranking_limit),
                    "description": "Unique listed tool names ranked best to worst.",
                },
                "reason": {
                    "type": "string",
                    "description": "One short sentence explaining the choice.",
                },
            },
            "required": ["selected_tool", "ranked_tools"],
        },
    }


def serialize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=True)


def extract_openai_tool_arguments(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise SelectionGenerationError("OpenAI response did not include any choices.")
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        raw_response = str(message.get("content", "")).strip() or None
        raise SelectionGenerationError(
            "OpenAI response did not include a tool call.",
            raw_response=raw_response,
        )

    first_call = tool_calls[0]
    function_payload = first_call.get("function", {}) if isinstance(first_call, dict) else {}
    function_name = str(function_payload.get("name", "")).strip()
    if function_name and function_name != "select_tool":
        raise SelectionGenerationError(f"OpenAI called unexpected function: {function_name}")

    arguments = function_payload.get("arguments")
    if arguments is None:
        raise SelectionGenerationError("OpenAI tool call did not include arguments.")
    return serialize_tool_arguments(arguments)


def extract_anthropic_tool_arguments(response: dict[str, Any]) -> str:
    content_blocks = response.get("content", [])
    raw_text_parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip()
        if block_type == "tool_use":
            tool_name = str(block.get("name", "")).strip()
            if tool_name and tool_name != "select_tool":
                raise SelectionGenerationError(f"Anthropic called unexpected tool: {tool_name}")
            return serialize_tool_arguments(block.get("input", {}))
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                raw_text_parts.append(text)

    raw_response = "\n".join(raw_text_parts).strip() or None
    raise SelectionGenerationError(
        "Anthropic response did not include a tool call.",
        raw_response=raw_response,
    )


def extract_gemini_tool_arguments(response: dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    raw_text_parts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall") or part.get("function_call")
            if isinstance(function_call, dict):
                function_name = str(function_call.get("name", "")).strip()
                if function_name and function_name != "select_tool":
                    raise SelectionGenerationError(f"Gemini called unexpected function: {function_name}")
                return serialize_tool_arguments(function_call.get("args", {}))

            text = str(part.get("text", "")).strip()
            if text:
                raw_text_parts.append(text)

    raw_response = "\n".join(raw_text_parts).strip() or None
    raise SelectionGenerationError(
        "Gemini response did not include a function call.",
        raw_response=raw_response,
    )


def extract_gemini_text_response(response: dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    text_parts: list[str] = []
    finish_messages: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_message = str(candidate.get("finishMessage", "")).strip()
        if finish_message:
            finish_messages.append(finish_message)
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if isinstance(part, dict):
                text = str(part.get("text", "")).strip()
                if text:
                    text_parts.append(text)

    raw_response = "\n".join(text_parts).strip()
    if raw_response:
        return raw_response

    detail = "; ".join(finish_messages).strip()
    message = "Gemini response did not include text content."
    if detail:
        message = f"{message} {detail}"
    raise SelectionGenerationError(message)


def redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(str(url))
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = urllib.parse.urlencode(
        [
            (key, "<redacted>" if key.lower() in {"key", "api_key", "access_token"} else value)
            for key, value in query_pairs
        ]
    )
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            redacted_query,
            parsed.fragment,
        )
    )


def is_rate_limit_response(status_code: int | None, body: str) -> bool:
    normalized_body = str(body).lower()
    return (
        status_code == 429
        or "rate_limit_error" in normalized_body
        or "rate limit" in normalized_body
        or "rate-limit" in normalized_body
    )


def http_post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    raw_bytes: bytes | None = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw_bytes = response.read()
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if is_rate_limit_response(exc.code, body) and attempt < RATE_LIMIT_MAX_RETRIES:
                time.sleep(RATE_LIMIT_RETRY_SECONDS)
                continue
            raise RuntimeError(f"HTTP {exc.code} from {redact_url(url)}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request to {redact_url(url)} failed: {exc}") from exc

    if raw_bytes is None:
        raise RuntimeError(f"Request to {redact_url(url)} did not return a response.")

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {redact_url(url)}") from exc


class EmbeddingSelectorEngine:
    def __init__(
        self,
        variant: EmbeddingVariantSpec,
        *,
        device: str,
        ranking_limit: int,
    ) -> None:
        require_torch()
        self.variant = variant
        self.device = resolve_runtime_device(device)
        self.ranking_limit = ranking_limit
        self.bundle = load_checkpoint_bundle_lazy(variant.checkpoint_path, device=self.device)
        self.model = self.bundle["model"]
        self.tokenizer = self.bundle["tokenizer"]
        self.max_length = int(self.bundle["max_length"])
        self.tool_names = [str(tool_name) for tool_name in self.bundle["tool_names"]]
        self.tool_index_by_name = {
            tool_name: index
            for index, tool_name in enumerate(self.tool_names)
        }
        self.centroids = F.normalize(self.bundle["centroids"].to(self.device), dim=-1)

    def select(
        self,
        query: str,
        *,
        candidate_tool_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        encoded = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        input_tokens = int(attention_mask.sum().item())

        start_time = time.perf_counter()
        embedding = self.model.encode(input_ids=input_ids, attention_mask=attention_mask)
        embedding = F.normalize(embedding, dim=-1)
        scores = (embedding @ self.centroids.T).squeeze(0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if candidate_tool_names is None:
            ranked_indices = torch.argsort(scores, descending=True).tolist()
        else:
            candidate_indices = [
                self.tool_index_by_name[tool_name]
                for tool_name in candidate_tool_names
                if tool_name in self.tool_index_by_name
            ]
            if not candidate_indices:
                raise ValueError("No candidate tools are present in the embedding checkpoint.")
            candidate_scores = scores[
                torch.tensor(candidate_indices, device=scores.device, dtype=torch.long)
            ]
            ranked_candidate_offsets = torch.argsort(candidate_scores, descending=True).tolist()
            ranked_indices = [candidate_indices[offset] for offset in ranked_candidate_offsets]

        ranked_tools = [self.tool_names[index] for index in ranked_indices[: self.ranking_limit]]
        ranked_scores = [round_float(float(scores[index].item())) for index in ranked_indices[: self.ranking_limit]]
        return {
            "selected_tool": ranked_tools[0],
            "ranked_tools": ranked_tools,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "total_tokens": input_tokens,
            "score_candidates": [
                {"tool": tool_name, "score": score}
                for tool_name, score in zip(ranked_tools, ranked_scores, strict=True)
            ],
        }


class LocalHFSelectionEngine:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        max_new_tokens: int,
        ranking_limit: int,
        local_files_only: bool,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.requested_device_name = device
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.ranking_limit = ranking_limit
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None
        self.input_device = None
        self.structured_model = None
        self.generator_cache: dict[str, tuple[type[BaseModel], Any]] = {}
        self.auto_fallback_used = False

    def can_retry_with_auto_device_map(self) -> bool:
        return self.device_name != "auto" and not self.auto_fallback_used

    def activate_auto_device_fallback(self) -> None:
        self.close()
        self.device_name = "auto"
        self.auto_fallback_used = True

    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None and self.input_device is not None:
            return

        try:
            model_loader, tokenizer_loader = require_transformers()
            outlines_module = require_outlines()
            ensure_transformers_output_recorder_compat()
            resolved_device = resolve_runtime_device(self.device_name)
            torch_dtype = resolve_torch_dtype(self.dtype_name, resolved_device)
            self.tokenizer = tokenizer_loader.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": self.local_files_only,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if self.device_name == "auto":
                model_kwargs["device_map"] = "auto"

            self.model = model_loader.from_pretrained(self.model_name, **model_kwargs)
            if self.device_name != "auto":
                self.model = self.model.to(resolved_device)
                self.input_device = torch.device(resolved_device)
            else:
                self.input_device = first_parameter_device(self.model)
            self.model.eval()
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.structured_model = outlines_module.from_transformers(self.model, self.tokenizer)
        except Exception as exc:
            if is_cuda_oom_error(exc) and self.can_retry_with_auto_device_map():
                self.activate_auto_device_fallback()
                self.ensure_loaded()
                return
            self.close()
            raise

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.input_device = None
        self.structured_model = None
        self.generator_cache.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_selection_generator(
        self,
        valid_tool_names: Sequence[str],
    ) -> tuple[type[BaseModel], Any]:
        assert self.structured_model is not None
        cache_key = json.dumps(
            {
                "tool_names": [str(tool_name).strip() for tool_name in valid_tool_names],
                "ranking_limit": self.ranking_limit,
            },
            sort_keys=True,
        )
        cached = self.generator_cache.get(cache_key)
        if cached is None:
            outlines_module = require_outlines()
            output_model = build_selection_output_model(
                valid_tool_names,
                ranking_limit=self.ranking_limit,
            )
            cached = (output_model, outlines_module.Generator(self.structured_model, output_model))
            self.generator_cache[cache_key] = cached
        return cached

    @inference_mode
    def select(self, query: str, tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        try:
            self.ensure_loaded()
            assert self.model is not None
            assert self.tokenizer is not None
            assert self.input_device is not None
            assert self.structured_model is not None

            valid_tool_names = [str(tool.get("name", "")).strip() for tool in tools if str(tool.get("name", "")).strip()]
            output_model, selection_generator = self.get_selection_generator(valid_tool_names)

            messages = build_selection_messages(
                query,
                tools,
                ranking_limit=self.ranking_limit,
            )
            prompt_text = build_prompt_text(self.tokenizer, messages)
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.input_device)
            attention_mask = encoded["attention_mask"].to(self.input_device)
            input_tokens = int(attention_mask.sum().item())
            del input_ids  # token counting is enough here; generation uses prompt_text.

            start_time = time.perf_counter()
            raw_response = selection_generator(
                prompt_text,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            output_tokens = compute_query_token_count(
                self.tokenizer,
                raw_response,
                max_length=max(self.max_new_tokens, 1),
            )
            total_tokens = input_tokens + output_tokens
            try:
                parsed_output = output_model.model_validate_json(raw_response)
            except Exception as exc:
                raise SelectionGenerationError(
                    "Model output did not contain a valid JSON object.",
                    raw_response=raw_response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                ) from exc

            normalized_ranked_tools = normalize_ranked_tools(
                list(parsed_output.ranked_tools),
                valid_tool_names=valid_tool_names,
                selected_tool=str(parsed_output.selected_tool),
                ranking_limit=self.ranking_limit,
            )
            if not normalized_ranked_tools:
                normalized_ranked_tools = [str(parsed_output.selected_tool)]
            return {
                "selected_tool": str(parsed_output.selected_tool),
                "ranked_tools": normalized_ranked_tools,
                "reason": str(parsed_output.reason).strip(),
                "raw_response": raw_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            if is_cuda_oom_error(exc) and self.can_retry_with_auto_device_map():
                self.activate_auto_device_fallback()
                return self.select(query, tools)
            raise


class BaseAPISelectionAdapter:
    provider = "api"

    def __init__(
        self,
        model_name: str,
        *,
        ranking_limit: int,
        max_output_tokens: int | None,
        timeout_seconds: int,
        pricing: dict[str, Any] | None,
    ) -> None:
        self.model_name = model_name
        self.ranking_limit = ranking_limit
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.pricing = pricing

    @property
    def adapter_id(self) -> str:
        return f"{self.provider}/{slugify(self.model_name)}"

    def call_api(
        self,
        messages: Sequence[dict[str, str]],
        *,
        valid_tool_names: Sequence[str],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def effective_max_output_tokens(self) -> int | None:
        if self.max_output_tokens is None:
            return None
        if int(self.max_output_tokens) <= 0:
            return None
        return int(self.max_output_tokens)

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"Benchmarking {self.adapter_id}", unit="example"):
            subset = sample_tools_for_query(tools, str(row["tool"]), n=50)
            messages = build_selection_messages(
                str(row["query"]),
                subset,
                ranking_limit=self.ranking_limit,
            )
            start_time = time.perf_counter()
            try:
                payload = self.call_api(
                    messages,
                    valid_tool_names=[str(tool.get("name", "")).strip() for tool in subset],
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                parsed = parse_selection_response(
                    payload["raw_response"],
                    valid_tool_names=[str(tool.get("name", "")).strip() for tool in subset],
                    ranking_limit=self.ranking_limit,
                )
                input_tokens = payload.get("input_tokens")
                output_tokens = payload.get("output_tokens")
                total_tokens = payload.get("total_tokens")
                cost_usd = estimate_cost_usd(
                    self.pricing,
                    model_name=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                results.append(
                    finalize_selection_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode="llm_api",
                        model_name=self.model_name,
                        row=row,
                        selected_tool=parsed["selected_tool"],
                        ranked_tools=parsed["ranked_tools"],
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        raw_response=payload["raw_response"],
                        reason=parsed["reason"],
                        cost_usd=cost_usd,
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                results.append(
                    build_error_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode="llm_api",
                        model_name=self.model_name,
                        row=row,
                        error_message=str(exc),
                        latency_ms=latency_ms,
                        raw_response=getattr(exc, "raw_response", None),
                        input_tokens=getattr(exc, "input_tokens", None),
                        output_tokens=getattr(exc, "output_tokens", None),
                        total_tokens=getattr(exc, "total_tokens", None),
                    )
                )

        return (
            build_model_summary(
                adapter_id=self.adapter_id,
                provider=self.provider,
                mode="llm_api",
                model_name=self.model_name,
                results=results,
            ),
            results,
        )


class OpenAISelectionAdapter(BaseAPISelectionAdapter):
    provider = "openai"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    def call_api(
        self,
        messages: Sequence[dict[str, str]],
        *,
        valid_tool_names: Sequence[str],
    ) -> dict[str, Any]:
        selection_tool = build_api_selection_tool(
            valid_tool_names,
            ranking_limit=self.ranking_limit,
        )
        payload = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": 0.0,
            "tools": [
                {
                    "type": "function",
                    "function": selection_tool,
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "select_tool"},
            },
        }
        max_output_tokens = self.effective_max_output_tokens()
        if max_output_tokens is not None:
            payload["max_completion_tokens"] = max_output_tokens
        response = http_post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        raw_response = extract_openai_tool_arguments(response)
        usage = response.get("usage", {})
        return {
            "raw_response": raw_response,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }


class AnthropicSelectionAdapter(BaseAPISelectionAdapter):
    provider = "anthropic"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

    def call_api(
        self,
        messages: Sequence[dict[str, str]],
        *,
        valid_tool_names: Sequence[str],
    ) -> dict[str, Any]:
        system_content = next(
            (message["content"] for message in messages if message.get("role") == "system"),
            "",
        )
        user_content = next(
            (message["content"] for message in messages if message.get("role") == "user"),
            "",
        )
        selection_tool = build_api_selection_tool(
            valid_tool_names,
            ranking_limit=self.ranking_limit,
        )
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": system_content,
            "messages": [{"role": "user", "content": user_content}],
            "tools": [
                {
                    "name": selection_tool["name"],
                    "description": selection_tool["description"],
                    "input_schema": selection_tool["parameters"],
                }
            ],
            "tool_choice": {
                "type": "tool",
                "name": "select_tool",
            },
        }
        response = http_post_json(
            url=f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        raw_response = extract_anthropic_tool_arguments(response)
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens
        return {
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


class GeminiSelectionAdapter(BaseAPISelectionAdapter):
    provider = "gemini"

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.api_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        self.base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    def call_api(
        self,
        messages: Sequence[dict[str, str]],
        *,
        valid_tool_names: Sequence[str],
    ) -> dict[str, Any]:
        system_content = next(
            (message["content"] for message in messages if message.get("role") == "system"),
            "",
        )
        user_content = next(
            (message["content"] for message in messages if message.get("role") == "user"),
            "",
        )
        model_path = self.model_name
        if not model_path.startswith("models/"):
            model_path = f"models/{model_path}"
        url = f"{self.base_url}/{model_path}:generateContent?{urllib.parse.urlencode({'key': self.api_key})}"
        generation_config: dict[str, Any] = {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }
        max_output_tokens = self.effective_max_output_tokens()
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens

        payload = {
            "systemInstruction": {"parts": [{"text": system_content}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": generation_config,
        }
        response = http_post_json(
            url=url,
            headers={},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        raw_response = extract_gemini_text_response(response)
        usage = response.get("usageMetadata", {})
        return {
            "raw_response": raw_response,
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        }


class EmbeddingSelectionAdapter:
    def __init__(
        self,
        variant: EmbeddingVariantSpec,
        *,
        device: str,
        ranking_limit: int,
    ) -> None:
        self.variant = variant
        self.provider = "embedding"
        self.mode = "embedding"
        self.model_name = variant.variant_id
        self.adapter_id = f"embedding/{slugify(variant.variant_id)}"
        self.engine = EmbeddingSelectorEngine(
            variant,
            device=device,
            ranking_limit=ranking_limit,
        )

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        pricing: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        del pricing
        candidate_tool_names = [
            str(tool.get("name", "")).strip()
            for tool in tools
            if str(tool.get("name", "")).strip()
        ]
        results: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"Benchmarking {self.variant.variant_id}", unit="example"):
            try:
                selection = self.engine.select(
                    str(row["query"]),
                    candidate_tool_names=candidate_tool_names or None,
                )
                results.append(
                    finalize_selection_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode=self.mode,
                        model_name=self.model_name,
                        row=row,
                        selected_tool=selection["selected_tool"],
                        ranked_tools=selection["ranked_tools"],
                        latency_ms=selection["latency_ms"],
                        input_tokens=selection["input_tokens"],
                        output_tokens=selection["output_tokens"],
                        total_tokens=selection["total_tokens"],
                        raw_response=None,
                        reason="Embedding nearest-neighbor selection.",
                        cost_usd=None,
                        extra={"score_candidates": selection["score_candidates"]},
                    )
                )
            except Exception as exc:
                results.append(
                    build_error_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode=self.mode,
                        model_name=self.model_name,
                        row=row,
                        error_message=str(exc),
                    )
                )
        summary = build_model_summary(
            adapter_id=self.adapter_id,
            provider=self.provider,
            mode=self.mode,
            model_name=self.model_name,
            results=results,
            metadata={
                "checkpoint_path": str(self.variant.checkpoint_path.resolve()),
                "architecture": self.variant.architecture,
                "loss_name": self.variant.loss_name,
            },
        )
        return summary, results


class LocalHFSelectionAdapter:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        ranking_limit: int,
        max_new_tokens: int,
        local_files_only: bool,
        pricing: dict[str, Any] | None,
    ) -> None:
        self.provider = "huggingface"
        self.mode = "llm_local"
        self.model_name = model_name
        self.adapter_id = f"hf/{slugify(model_name)}"
        self.engine = LocalHFSelectionEngine(
            model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            ranking_limit=ranking_limit,
            local_files_only=local_files_only,
        )
        self.pricing = pricing

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        try:
            
            for row in tqdm(rows, desc=f"Benchmarking {self.model_name}", unit="example"):
                subset = sample_tools_for_query(tools, str(row["tool"]), n=50)
                start_time = time.perf_counter()
                try:
                    selection = self.engine.select(str(row["query"]), subset)
                    latency_ms = selection["latency_ms"]
                    cost_usd = estimate_cost_usd(
                        self.pricing,
                        model_name=self.model_name,
                        input_tokens=selection["input_tokens"],
                        output_tokens=selection["output_tokens"],
                    )
                    results.append(
                        finalize_selection_result(
                            adapter_id=self.adapter_id,
                            provider=self.provider,
                            mode=self.mode,
                            model_name=self.model_name,
                            row=row,
                            selected_tool=selection["selected_tool"],
                            ranked_tools=selection["ranked_tools"],
                            latency_ms=latency_ms,
                            input_tokens=selection["input_tokens"],
                            output_tokens=selection["output_tokens"],
                            total_tokens=selection["total_tokens"],
                            raw_response=selection["raw_response"],
                            reason=selection["reason"],
                            cost_usd=cost_usd,
                        )
                    )
                except Exception as exc:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    results.append(
                        build_error_result(
                            adapter_id=self.adapter_id,
                            provider=self.provider,
                            mode=self.mode,
                            model_name=self.model_name,
                            row=row,
                            error_message=str(exc),
                            latency_ms=latency_ms,
                            raw_response=getattr(exc, "raw_response", None),
                            input_tokens=getattr(exc, "input_tokens", None),
                            output_tokens=getattr(exc, "output_tokens", None),
                            total_tokens=getattr(exc, "total_tokens", None),
                        )
                    )
        finally:
            self.engine.close()

        return (
            build_model_summary(
                adapter_id=self.adapter_id,
                provider=self.provider,
                mode=self.mode,
                model_name=self.model_name,
                results=results,
            ),
            results,
        )


class HybridEmbeddingRerankAdapter:
    def __init__(
        self,
        variant: EmbeddingVariantSpec,
        reranker_engine: LocalHFSelectionEngine,
        *,
        device: str,
        embedding_top_k: int,
        ranking_limit: int,
        pricing: dict[str, Any] | None,
    ) -> None:
        self.variant = variant
        self.reranker_engine = reranker_engine
        self.provider = "hybrid"
        self.mode = "embedding_rerank"
        self.model_name = f"{variant.variant_id}+{reranker_engine.model_name}"
        self.adapter_id = f"hybrid/{slugify(variant.variant_id)}-rerank-{slugify(reranker_engine.model_name)}"
        self.embedding_top_k = embedding_top_k
        self.ranking_limit = ranking_limit
        self.pricing = pricing
        self.embedding_engine = EmbeddingSelectorEngine(
            variant,
            device=device,
            ranking_limit=embedding_top_k,
        )

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tool_by_name = {
            str(tool.get("name", "")).strip(): tool
            for tool in tools
            if str(tool.get("name", "")).strip()
        }
        results: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"Benchmarking {self.model_name}", unit="example"):
            total_start = time.perf_counter()
            try:
                embedding_selection = self.embedding_engine.select(str(row["query"]))
                candidate_names = embedding_selection["ranked_tools"][: self.embedding_top_k]
                candidate_tools = [tool_by_name[name] for name in candidate_names if name in tool_by_name]
                reranked = self.reranker_engine.select(str(row["query"]), candidate_tools)
                total_latency_ms = (time.perf_counter() - total_start) * 1000.0
                cost_usd = estimate_cost_usd(
                    self.pricing,
                    model_name=self.reranker_engine.model_name,
                    input_tokens=reranked["input_tokens"],
                    output_tokens=reranked["output_tokens"],
                )
                results.append(
                    finalize_selection_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode=self.mode,
                        model_name=self.model_name,
                        row=row,
                        selected_tool=reranked["selected_tool"],
                        ranked_tools=reranked["ranked_tools"],
                        latency_ms=total_latency_ms,
                        input_tokens=(embedding_selection["input_tokens"] + (reranked["input_tokens"] or 0)),
                        output_tokens=reranked["output_tokens"],
                        total_tokens=(embedding_selection["input_tokens"] + (reranked["total_tokens"] or 0)),
                        raw_response=reranked["raw_response"],
                        reason=reranked["reason"],
                        cost_usd=cost_usd,
                        extra={
                            "embedding_candidates": embedding_selection["score_candidates"],
                            "embedding_top_k": self.embedding_top_k,
                        },
                    )
                )
            except Exception as exc:
                total_latency_ms = (time.perf_counter() - total_start) * 1000.0
                results.append(
                    build_error_result(
                        adapter_id=self.adapter_id,
                        provider=self.provider,
                        mode=self.mode,
                        model_name=self.model_name,
                        row=row,
                        error_message=str(exc),
                        latency_ms=total_latency_ms,
                        raw_response=getattr(exc, "raw_response", None),
                        input_tokens=getattr(exc, "input_tokens", None),
                        output_tokens=getattr(exc, "output_tokens", None),
                        total_tokens=getattr(exc, "total_tokens", None),
                    )
                )

        summary = build_model_summary(
            adapter_id=self.adapter_id,
            provider=self.provider,
            mode=self.mode,
            model_name=self.model_name,
            results=results,
            metadata={
                "checkpoint_path": str(self.variant.checkpoint_path.resolve()),
                "architecture": self.variant.architecture,
                "loss_name": self.variant.loss_name,
                "reranker_model": self.reranker_engine.model_name,
                "embedding_top_k": self.embedding_top_k,
            },
        )
        return summary, results


__all__ = [
    "AnthropicSelectionAdapter",
    "EmbeddingSelectionAdapter",
    "GeminiSelectionAdapter",
    "HybridEmbeddingRerankAdapter",
    "LocalHFSelectionAdapter",
    "LocalHFSelectionEngine",
    "OpenAISelectionAdapter",
    "build_model_summary",
]
