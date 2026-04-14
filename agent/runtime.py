from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import outlines
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, create_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from agent.protocol import (
    ASSISTANT_CONTROL_TAGS,
    IncrementalControlBlockParser,
    serialize_json_block,
)
from REPL.tools import TOOL_REGISTRY, dispatch_tool_call
from training import embed_texts, load_checkpoint_bundle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "data/OSS/tools.json"
DEFAULT_EMBED_CHECKPOINT_PATH = PROJECT_ROOT / "data/OSS/output/normal/circle/best.pt"
DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen3.5-27B"
DEFAULT_RESPONSE_PREVIEW_CHARS = 2000
DEFAULT_CONTEXT_CHARS = 5000

SYSTEM_PROMPT = """You are NTILC, an assistant that reasons and responds in markdown, and can invoke external tools through a structured XML control protocol.

---

## Core Behavior

- Respond in markdown by default.
- Use tools only when the task genuinely requires external data, computation, or actions you cannot perform directly.
- Never fabricate results. If a tool call fails or returns nothing useful, say so clearly.

---

## Tool Protocol — Strict Turn Order

Each tool invocation follows exactly this sequence. Emit only one control block per turn, then stop.

### Step 1 — Discover
When you need a capability, emit exactly:
```xml
<search_tools>plain-language description of the capability you need</search_tools>
```
Then stop. Do not write anything after this block.

### Step 2 — Select
The controller will reply with a `<tool_candidates>` block listing available tools.
Choose the best match and emit exactly:
```xml
<select_tool>tool_name</select_tool>
```
Use only names from the candidates list. Then stop.

### Step 3 — Call
The controller will reply with a `<tool_spec>` block describing the tool's arguments.
Read the spec carefully, then emit exactly:
```xml
<dispatch>
  <tool>tool_name</tool>
  <args>
    <arg_name>value</arg_name>
    <!-- one element per required argument -->
  </args>
</dispatch>
```
Use only argument names defined in the `<tool_spec>`. Do not guess or invent schema. Then stop.

### Step 4 — Resume
The controller will reply with a `<response>` block containing the tool result.
Use the result to continue — either answer in markdown, or begin another tool invocation at Step 1 if more data is needed.

---

## Hard Rules

- Never emit `<tool_candidates>`, `<tool_spec>`, `<dispatch>`, or `<response>` — those are controller-only blocks.
- Never emit more than one control block per turn.
- Never invent argument names or schemas before seeing a `<tool_spec>`.
- If no tool candidates match your need, inform the user and answer as best you can without tools.
- If a `<response>` indicates an error, explain what failed and suggest next steps rather than retrying silently.
- Control block tags must be syntactically exact — no attributes, no variations, no extra whitespace inside tags.
"""


Message = dict[str, str]
TextCallback = Callable[[str], None]
EventCallback = Callable[[dict[str, Any]], None]


class ModelAdapter(Protocol):
    def stream_assistant(self, transcript: Sequence[Message]) -> Iterable[str]:
        ...

    def generate_dispatch_arguments(
        self,
        transcript: Sequence[Message],
        user_message: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def count_text_tokens(self, text: str) -> int:
        ...

    def count_assistant_prompt_tokens(self, transcript: Sequence[Message]) -> int:
        ...

    def count_dispatch_prompt_tokens(
        self,
        transcript: Sequence[Message],
        user_message: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> int:
        ...


@dataclass(frozen=True)
class RuntimeConfig:
    tools_path: Path = DEFAULT_TOOLS_PATH
    embed_checkpoint_path: Path = DEFAULT_EMBED_CHECKPOINT_PATH
    qwen_model_name: str = DEFAULT_QWEN_MODEL_NAME
    qwen_device: str = "cuda:6"
    embed_device: str = "cuda:7"
    qwen_dtype: str = "bfloat16"
    local_files_only: bool = False
    top_k: int = 5
    max_tool_steps: int = 999
    max_agent_passes: int = 999
    assistant_max_new_tokens: int = 4096
    dispatch_max_new_tokens: int = 4096
    embedding_batch_size: int = 8
    response_preview_chars: int = DEFAULT_RESPONSE_PREVIEW_CHARS
    transcript_preview_chars: int = DEFAULT_CONTEXT_CHARS


@dataclass(frozen=True)
class ToolCandidate:
    name: str
    description: str
    category: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "score": self.score,
        }


@dataclass
class AgentResources:
    config: RuntimeConfig
    tool_registry: list[dict[str, Any]]
    tool_by_name: dict[str, dict[str, Any]]
    retriever: "ToolRetriever"
    model_adapter: ModelAdapter


@dataclass
class ControllerTurnResult:
    assistant_text: str
    events: list[dict[str, Any]]
    transcript: list[Message]
    stats: dict[str, Any]


@dataclass
class ControllerTurnStats:
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_seconds: float = 0.0
    agent_passes: int = 0
    user_message_tokens: int = 0
    assistant_prompt_tokens: int = 0
    assistant_output_tokens: int = 0
    dispatch_prompt_tokens: int = 0
    dispatch_output_tokens: int = 0
    search_query_tokens: int = 0
    search_count: int = 0
    candidates_returned: int = 0
    dispatch_count: int = 0
    tool_call_count: int = 0
    tool_success_count: int = 0
    tool_error_count: int = 0
    controller_error_count: int = 0
    tools_used: list[str] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    def finish(self) -> None:
        self.completed_at = time.time()
        self.duration_seconds = max(0.0, self.completed_at - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "agent_passes": self.agent_passes,
            "user_message_tokens": self.user_message_tokens,
            "assistant_prompt_tokens": self.assistant_prompt_tokens,
            "assistant_output_tokens": self.assistant_output_tokens,
            "dispatch_prompt_tokens": self.dispatch_prompt_tokens,
            "dispatch_output_tokens": self.dispatch_output_tokens,
            "search_query_tokens": self.search_query_tokens,
            "search_count": self.search_count,
            "candidates_returned": self.candidates_returned,
            "dispatch_count": self.dispatch_count,
            "tool_call_count": self.tool_call_count,
            "tool_success_count": self.tool_success_count,
            "tool_error_count": self.tool_error_count,
            "controller_error_count": self.controller_error_count,
            "tools_used": list(self.tools_used),
            "selected_tools": list(self.selected_tools),
            "search_queries": list(self.search_queries),
            "model_input_tokens": self.assistant_prompt_tokens + self.dispatch_prompt_tokens,
            "model_output_tokens": self.assistant_output_tokens + self.dispatch_output_tokens,
            "total_model_tokens": (
                self.assistant_prompt_tokens
                + self.dispatch_prompt_tokens
                + self.assistant_output_tokens
                + self.dispatch_output_tokens
            ),
        }


def normalize_outline_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def truncate_text(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


def format_controller_observation(block_text: str) -> str:
    return f"Controller observation:\n{block_text}"


def resolve_runtime_devices(qwen_device: str, embed_device: str) -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "cpu", "cpu"

    resolved_qwen = "cuda:6" if qwen_device == "auto" else qwen_device
    if embed_device == "auto":
        if torch.cuda.device_count() > 1:
            return resolved_qwen, "cuda:7"
        return resolved_qwen, resolved_qwen
    return resolved_qwen, embed_device


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    normalized = dtype_name.strip().lower()
    if normalized in {"", "auto"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported qwen dtype: {dtype_name}")
    return mapping[normalized]


def schema_type_to_annotation(spec: dict[str, Any]) -> Any:
    schema_type = spec.get("type", "string")
    enum_values = spec.get("enum")
    if enum_values:
        return str
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]
    return Any


def build_dispatch_output_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[Any, Field]] = {}

    for name, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {"type": "string"}
        annotation = schema_type_to_annotation(spec)
        if name in required:
            fields[name] = (annotation, Field(...))
        else:
            fields[name] = (annotation | None, Field(default=spec.get("default")))

    safe_name = "".join(part.capitalize() for part in tool_name.replace("-", "_").split("_")) or "Tool"
    return create_model(f"{safe_name}Dispatch", **fields)


def normalize_dispatch_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, str):
            normalized[name] = normalize_outline_text(value)
        elif isinstance(value, list):
            normalized[name] = [
                normalize_outline_text(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            normalized[name] = value
    return normalized


def build_dispatcher_prompt(
    transcript: Sequence[Message],
    user_message: str,
    tool_name: str,
    schema: dict[str, Any],
    transcript_preview_chars: int,
) -> str:
    transcript_lines: list[str] = []
    for message in transcript[-12:]:
        role = message["role"].upper()
        content = truncate_text(message["content"], transcript_preview_chars // 3)
        transcript_lines.append(f"{role}: {content}")
    transcript_text = "\n".join(transcript_lines)
    return f"""You are the NTILC dispatcher.
Return only a JSON object containing the arguments for the selected tool.
Use the provided schema exactly.
Do not add extra keys or commentary.

Conversation context:
{transcript_text}

Current user request:
{user_message}

Selected tool:
{tool_name}

Tool schema:
{json.dumps(schema, indent=2, ensure_ascii=True)}
"""


class ToolRetriever:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        max_length: int,
        tool_names: list[str],
        tool_centroids: torch.Tensor,
        tool_by_name: dict[str, dict[str, Any]],
        device: str,
        batch_size: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tool_names = tool_names
        self.tool_centroids = tool_centroids
        self.tool_by_name = tool_by_name
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        tool_registry: list[dict[str, Any]],
        *,
        device: str,
        batch_size: int,
    ) -> "ToolRetriever":
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Embedding checkpoint not found: {checkpoint_path}. "
                "Train or place the 26-tool checkpoint before starting the app."
            )

        bundle = load_checkpoint_bundle(checkpoint_path, device=device)
        tool_by_name = {str(tool["name"]): tool for tool in tool_registry}
        expected_names = list(tool_by_name)
        checkpoint_names = [str(name) for name in bundle["tool_names"]]

        if len(set(expected_names)) != len(expected_names):
            raise RuntimeError("Tool catalog contains duplicate tool names.")
        if len(set(checkpoint_names)) != len(checkpoint_names):
            raise RuntimeError("Embedding checkpoint contains duplicate tool names.")

        if set(checkpoint_names) != set(expected_names):
            missing = sorted(set(expected_names) - set(checkpoint_names))
            extra = sorted(set(checkpoint_names) - set(expected_names))
            raise RuntimeError(
                "Embedding checkpoint tool names do not match data/OSS/tools.json. "
                f"Missing in checkpoint: {missing or '[]'}. Extra in checkpoint: {extra or '[]'}."
            )

        tool_centroids = F.normalize(bundle["centroids"].to(device), dim=-1)
        return cls(
            model=bundle["model"],
            tokenizer=bundle["tokenizer"],
            max_length=int(bundle["max_length"]),
            tool_names=checkpoint_names,
            tool_centroids=tool_centroids,
            tool_by_name=tool_by_name,
            device=device,
            batch_size=batch_size,
        )

    def query(self, query_text: str, top_k: int = 5) -> list[ToolCandidate]:
        normalized_query = normalize_outline_text(query_text)
        if not normalized_query:
            return []

        embeddings = embed_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=[normalized_query],
            device=self.device,
            max_length=self.max_length,
            batch_size=1 if self.batch_size < 1 else min(1, self.batch_size),
            progress_desc="Embedding tool search",
        )
        query_embedding = F.normalize(embeddings.to(self.tool_centroids.device), dim=-1)
        scores = query_embedding @ self.tool_centroids.T
        k = min(top_k, len(self.tool_names))
        top_scores, top_indices = torch.topk(scores[0], k=k)

        candidates: list[ToolCandidate] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
            tool_name = self.tool_names[index]
            tool_spec = self.tool_by_name[tool_name]
            candidates.append(
                ToolCandidate(
                    name=tool_name,
                    description=str(tool_spec.get("description", "")),
                    category=str(tool_spec.get("category", "")),
                    score=round(float(score), 4),
                )
            )
        return candidates


class HFQwenModelAdapter:
    def __init__(
        self,
        *,
        qwen_model: Any,
        qwen_tokenizer: Any,
        structured_qwen: Any,
        assistant_max_new_tokens: int,
        dispatch_max_new_tokens: int,
        transcript_preview_chars: int,
    ) -> None:
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.structured_qwen = structured_qwen
        self.assistant_max_new_tokens = assistant_max_new_tokens
        self.dispatch_max_new_tokens = dispatch_max_new_tokens
        self.transcript_preview_chars = transcript_preview_chars
        self.dispatch_generator_cache: dict[str, tuple[type[BaseModel], Any]] = {}
        self.input_device = next(qwen_model.parameters()).device

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "HFQwenModelAdapter":
        qwen_device, _ = resolve_runtime_devices(config.qwen_device, config.embed_device)
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_model_name,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token or qwen_tokenizer.unk_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": {"": qwen_device},
            "local_files_only": config.local_files_only,
        }
        qwen_dtype = resolve_torch_dtype(config.qwen_dtype)
        if qwen_dtype is not None:
            model_kwargs["dtype"] = qwen_dtype

        qwen_model = AutoModelForCausalLM.from_pretrained(
            config.qwen_model_name,
            **model_kwargs,
        )
        qwen_model.eval()
        qwen_model.generation_config.pad_token_id = qwen_tokenizer.pad_token_id
        structured_qwen = outlines.from_transformers(qwen_model, qwen_tokenizer)
        return cls(
            qwen_model=qwen_model,
            qwen_tokenizer=qwen_tokenizer,
            structured_qwen=structured_qwen,
            assistant_max_new_tokens=config.assistant_max_new_tokens,
            dispatch_max_new_tokens=config.dispatch_max_new_tokens,
            transcript_preview_chars=config.transcript_preview_chars,
        )

    def stream_assistant(self, transcript: Sequence[Message]) -> Iterable[str]:
        prompt = self._render_chat_prompt(transcript)
        encoded = self.qwen_tokenizer(prompt, return_tensors="pt")
        encoded = {name: tensor.to(self.input_device) for name, tensor in encoded.items()}

        streamer = TextIteratorStreamer(
            self.qwen_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": self.assistant_max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.qwen_tokenizer.pad_token_id,
            "eos_token_id": self.qwen_tokenizer.eos_token_id,
        }
        worker_error: list[BaseException] = []

        def run_generation() -> None:
            try:
                self.qwen_model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001
                worker_error.append(exc)

        worker = threading.Thread(
            target=run_generation,
            daemon=True,
        )
        worker.start()
        try:
            for piece in streamer:
                yield piece
        finally:
            worker.join()
        if worker_error:
            raise RuntimeError(f"Assistant generation failed: {worker_error[0]}") from worker_error[0]

    def generate_dispatch_arguments(
        self,
        transcript: Sequence[Message],
        user_message: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = build_dispatcher_prompt(
            transcript=transcript,
            user_message=user_message,
            tool_name=tool_name,
            schema=schema,
            transcript_preview_chars=self.transcript_preview_chars,
        )
        output_model, generator = self._get_dispatch_generator(tool_name, schema)
        raw_dispatch_json = generator(
            prompt,
            max_new_tokens=self.dispatch_max_new_tokens,
            pad_token_id=self.qwen_tokenizer.pad_token_id,
            eos_token_id=self.qwen_tokenizer.eos_token_id,
            do_sample=False,
        )
        dispatch_payload = output_model.model_validate_json(raw_dispatch_json)
        return normalize_dispatch_arguments(dispatch_payload.model_dump(exclude_none=True))

    def count_text_tokens(self, text: str) -> int:
        normalized = str(text or "")
        if not normalized.strip():
            return 0
        encoded = self.qwen_tokenizer(
            normalized,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def count_assistant_prompt_tokens(self, transcript: Sequence[Message]) -> int:
        return self.count_text_tokens(self._render_chat_prompt(transcript))

    def count_dispatch_prompt_tokens(
        self,
        transcript: Sequence[Message],
        user_message: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> int:
        prompt = build_dispatcher_prompt(
            transcript=transcript,
            user_message=user_message,
            tool_name=tool_name,
            schema=schema,
            transcript_preview_chars=self.transcript_preview_chars,
        )
        return self.count_text_tokens(prompt)

    def _render_chat_prompt(self, transcript: Sequence[Message]) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": message["role"], "content": message["content"]} for message in transcript)
        if hasattr(self.qwen_tokenizer, "apply_chat_template"):
            return self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)

    def _get_dispatch_generator(
        self,
        tool_name: str,
        schema: dict[str, Any],
    ) -> tuple[type[BaseModel], Any]:
        cache_key = f"{tool_name}:{json.dumps(schema, sort_keys=True)}"
        cached = self.dispatch_generator_cache.get(cache_key)
        if cached is None:
            output_model = build_dispatch_output_model(tool_name, schema)
            cached = (output_model, outlines.Generator(self.structured_qwen, output_model))
            self.dispatch_generator_cache[cache_key] = cached
        return cached


class AgentController:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        model_adapter: ModelAdapter,
        retriever: ToolRetriever,
        tool_by_name: dict[str, dict[str, Any]],
        executor: Callable[[str, dict[str, Any], dict[str, Any] | None], dict[str, Any]] = dispatch_tool_call,
    ) -> None:
        self.config = config
        self.model_adapter = model_adapter
        self.retriever = retriever
        self.tool_by_name = tool_by_name
        self.executor = executor

    def run_turn(
        self,
        *,
        user_message: str,
        transcript: Sequence[Message],
        on_text: TextCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> ControllerTurnResult:
        working_transcript = [dict(message) for message in transcript]
        working_transcript.append({"role": "user", "content": user_message})

        assistant_text_parts: list[str] = []
        events: list[dict[str, Any]] = []
        latest_candidates: dict[str, ToolCandidate] = {}
        tool_budget_exhausted = False
        stats = ControllerTurnStats(
            user_message_tokens=self.model_adapter.count_text_tokens(user_message),
        )

        for _ in range(self.config.max_agent_passes):
            stats.agent_passes += 1
            stats.assistant_prompt_tokens += self.model_adapter.count_assistant_prompt_tokens(
                working_transcript
            )
            parser = IncrementalControlBlockParser(ASSISTANT_CONTROL_TAGS)
            pass_visible = ""
            for chunk in self.model_adapter.stream_assistant(working_transcript):
                text_chunks, _ = parser.feed(chunk)
                if text_chunks:
                    appended = "".join(text_chunks)
                    pass_visible += appended
                    assistant_text_parts.append(appended)
                    if on_text is not None:
                        on_text(appended)

            snapshot = parser.finalize()
            if len(snapshot.visible_text) > len(pass_visible):
                tail = snapshot.visible_text[len(pass_visible) :]
                if tail:
                    assistant_text_parts.append(tail)
                    if on_text is not None:
                        on_text(tail)

            if snapshot.trimmed_output:
                stats.assistant_output_tokens += self.model_adapter.count_text_tokens(
                    snapshot.trimmed_output
                )

            if snapshot.trimmed_output:
                working_transcript.append({"role": "assistant", "content": snapshot.trimmed_output})

            if snapshot.issue is not None:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool="controller",
                    error=snapshot.issue.message,
                    retryable=True,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            if snapshot.block is None:
                break

            if snapshot.block.tag == "search_tools":
                query = normalize_outline_text(snapshot.block.content)
                stats.search_count += 1
                stats.search_queries.append(query)
                stats.search_query_tokens += self.model_adapter.count_text_tokens(query)
                if tool_budget_exhausted:
                    stats.controller_error_count += 1
                    response_event, response_block = self._build_response_error(
                        tool="controller",
                        error="Tool budget exhausted. Provide a final answer without more tool calls.",
                        retryable=False,
                    )
                    self._emit_event(events, response_event, on_event)
                    working_transcript.append(
                        {"role": "user", "content": format_controller_observation(response_block)}
                    )
                    continue

                candidates = self.retriever.query(query, top_k=self.config.top_k)
                stats.candidates_returned += len(candidates)
                latest_candidates = {candidate.name: candidate for candidate in candidates}
                search_payload = {
                    "query": query,
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
                search_block = serialize_json_block("tool_candidates", search_payload)
                search_event = {
                    "type": "search",
                    "query": query,
                    "candidates": search_payload["candidates"],
                    "block": search_block,
                    "raw_payload": search_payload,
                }
                self._emit_event(events, search_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(search_block)})

                if not candidates:
                    stats.controller_error_count += 1
                    response_event, response_block = self._build_response_error(
                        tool="controller",
                        error="No tools matched the search query.",
                        retryable=True,
                    )
                    self._emit_event(events, response_event, on_event)
                    working_transcript.append(
                        {"role": "user", "content": format_controller_observation(response_block)}
                    )
                continue

            selected_tool_name = normalize_outline_text(snapshot.block.content)
            if tool_budget_exhausted:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool="controller",
                    error="Tool budget exhausted. Provide a final answer without more tool calls.",
                    retryable=False,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            if not latest_candidates:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool="controller",
                    error="Received <select_tool> before any <tool_candidates> block.",
                    retryable=True,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            if selected_tool_name not in latest_candidates:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool="controller",
                    error=(
                        f"Selected tool '{selected_tool_name}' is not present in the latest "
                        "tool candidate list."
                    ),
                    retryable=True,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            tool_spec = self.tool_by_name.get(selected_tool_name)
            if tool_spec is None:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool=selected_tool_name,
                    error=f"Tool specification for '{selected_tool_name}' is missing.",
                    retryable=False,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            tool_spec_payload = {
                "name": selected_tool_name,
                "description": tool_spec.get("description", ""),
                "category": tool_spec.get("category", ""),
                "parameters": tool_spec.get("parameters", {}),
            }
            tool_spec_block = serialize_json_block("tool_spec", tool_spec_payload)
            working_transcript.append({"role": "user", "content": format_controller_observation(tool_spec_block)})

            try:
                stats.dispatch_prompt_tokens += self.model_adapter.count_dispatch_prompt_tokens(
                    transcript=working_transcript,
                    user_message=user_message,
                    tool_name=selected_tool_name,
                    schema=tool_spec_payload["parameters"],
                )
                dispatch_arguments = self.model_adapter.generate_dispatch_arguments(
                    transcript=working_transcript,
                    user_message=user_message,
                    tool_name=selected_tool_name,
                    schema=tool_spec_payload["parameters"],
                )
                stats.dispatch_output_tokens += self.model_adapter.count_text_tokens(
                    json.dumps(dispatch_arguments, ensure_ascii=True, sort_keys=True)
                )
            except Exception as exc:
                stats.controller_error_count += 1
                response_event, response_block = self._build_response_error(
                    tool=selected_tool_name,
                    error=f"Dispatch generation failed: {exc}",
                    retryable=True,
                )
                self._emit_event(events, response_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})
                continue

            dispatch_payload = {"tool": selected_tool_name, "arguments": dispatch_arguments}
            dispatch_block = serialize_json_block("dispatch", dispatch_payload)
            stats.dispatch_count += 1
            stats.selected_tools.append(selected_tool_name)
            dispatch_event = {
                "type": "dispatch",
                "tool": selected_tool_name,
                "arguments": dispatch_arguments,
                "block": dispatch_block,
                "raw_payload": dispatch_payload,
            }
            self._emit_event(events, dispatch_event, on_event)
            working_transcript.append({"role": "user", "content": format_controller_observation(dispatch_block)})

            stats.tool_call_count += 1
            stats.tools_used.append(selected_tool_name)
            tool_response = self.executor(selected_tool_name, dispatch_arguments, tool_spec)
            response_event, response_block = self._build_response_event(tool_response)
            if response_event["status"] == "ok":
                stats.tool_success_count += 1
            else:
                stats.tool_error_count += 1
            self._emit_event(events, response_event, on_event)
            working_transcript.append({"role": "user", "content": format_controller_observation(response_block)})

            if response_event["status"] == "ok":
                latest_candidates = {}
            if sum(1 for event in events if event["type"] == "response" and event["tool"] != "controller") >= self.config.max_tool_steps:
                tool_budget_exhausted = True
                stats.controller_error_count += 1
                limit_event, limit_block = self._build_response_error(
                    tool="controller",
                    error="Reached the configured max tool steps. Provide a final answer without more tools.",
                    retryable=False,
                )
                self._emit_event(events, limit_event, on_event)
                working_transcript.append({"role": "user", "content": format_controller_observation(limit_block)})
        else:
            limit_note = "\n\nI hit the configured pass limit before reaching a final answer."
            assistant_text_parts.append(limit_note)
            if on_text is not None:
                on_text(limit_note)
            working_transcript.append({"role": "assistant", "content": limit_note})

        stats.finish()
        return ControllerTurnResult(
            assistant_text="".join(assistant_text_parts),
            events=events,
            transcript=working_transcript,
            stats=stats.as_dict(),
        )

    def _emit_event(
        self,
        event_store: list[dict[str, Any]],
        event: dict[str, Any],
        callback: EventCallback | None,
    ) -> None:
        event_store.append(event)
        if callback is not None:
            callback(event)

    def _build_response_event(self, tool_response: dict[str, Any]) -> tuple[dict[str, Any], str]:
        output = tool_response.get("output")
        error = tool_response.get("error")
        retryable = bool(tool_response.get("status") != "ok")
        prompt_payload = {
            "tool": tool_response.get("tool"),
            "status": tool_response.get("status"),
            "output": truncate_text(output, self.config.response_preview_chars) if output is not None else None,
            "error": error,
            "retryable": retryable,
        }
        response_block = serialize_json_block("response", prompt_payload)
        response_event = {
            "type": "response",
            "tool": tool_response.get("tool"),
            "status": tool_response.get("status"),
            "output": output,
            "error": error,
            "retryable": retryable,
            "block": response_block,
            "raw_payload": prompt_payload,
        }
        return response_event, response_block

    def _build_response_error(
        self,
        *,
        tool: str,
        error: str,
        retryable: bool,
    ) -> tuple[dict[str, Any], str]:
        payload = {
            "tool": tool,
            "status": "error",
            "output": None,
            "error": error,
            "retryable": retryable,
        }
        response_block = serialize_json_block("response", payload)
        event = {
            "type": "response",
            "tool": tool,
            "status": "error",
            "output": None,
            "error": error,
            "retryable": retryable,
            "block": response_block,
            "raw_payload": payload,
        }
        return event, response_block


def load_tool_catalog(tools_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not tools_path.exists():
        raise RuntimeError(f"Tool catalog not found: {tools_path}")
    try:
        payload = json.loads(tools_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse tool catalog at {tools_path}: {exc}") from exc

    tool_registry = payload.get("tools")
    if not isinstance(tool_registry, list):
        raise RuntimeError("Tool catalog must contain a top-level 'tools' array.")

    tool_by_name: dict[str, dict[str, Any]] = {}
    for tool in tool_registry:
        if not isinstance(tool, dict) or "name" not in tool or "parameters" not in tool:
            raise RuntimeError("Each tool entry must include at least 'name' and 'parameters'.")
        tool_name = str(tool["name"])
        if tool_name in tool_by_name:
            raise RuntimeError(f"Duplicate tool name in catalog: {tool_name}")
        tool_by_name[tool_name] = tool

    missing_in_executor = sorted(set(tool_by_name) - set(TOOL_REGISTRY))
    if missing_in_executor:
        raise RuntimeError(
            "The tool catalog contains tools that REPL/tools.py cannot execute: "
            f"{missing_in_executor}"
        )

    return tool_registry, tool_by_name


def build_agent_resources(config: RuntimeConfig) -> AgentResources:
    tool_registry, tool_by_name = load_tool_catalog(config.tools_path)
    _, embed_device = resolve_runtime_devices(config.qwen_device, config.embed_device)
    retriever = ToolRetriever.from_checkpoint(
        config.embed_checkpoint_path,
        tool_registry,
        device=embed_device,
        batch_size=config.embedding_batch_size,
    )
    model_adapter = HFQwenModelAdapter.from_config(config)
    return AgentResources(
        config=config,
        tool_registry=tool_registry,
        tool_by_name=tool_by_name,
        retriever=retriever,
        model_adapter=model_adapter,
    )
