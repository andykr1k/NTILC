from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List
import outlines
import torch
from pydantic import BaseModel, Field, create_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You create short, realistic user requests for tool-routing datasets.
Each string must be a natural user request.
Do not mention tool names, JSON, schemas, or implementation details.
Do not number the items."""
DATA_DIR = Path("data/OSS")
DEFAULT_TOOLS_PATH = DATA_DIR / "tools.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "tool_embedding_dataset.jsonl"
DEFAULT_SUMMARY_PATH = DATA_DIR / "tool_embedding_dataset_summary.json"
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = {1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple tool-query dataset.")
    parser.add_argument(
        "--tools-path",
        type=str,
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to the tools.json file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output JSONL dataset.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to a small metadata summary JSON file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help=(
            "Path to a resumable generation checkpoint. Defaults to "
            "<output-path>.checkpoint.json."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing checkpoint and start generation from scratch.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="Generator model name. Change this if your exact Qwen checkpoint name differs.",
    )
    parser.add_argument(
        "--examples-per-tool",
        type=int,
        default=20,
        help="How many synthetic queries to create per tool.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=64,
        help="How many queries to ask the model for in one generation step.",
    )
    parser.add_argument(
        "--tool-batch-size",
        type=int,
        default=1,
        help="How many tool prompts to generate in one model batch.",
    )
    parser.add_argument(
        "--max-attempts-per-tool",
        type=int,
        default=8,
        help="Maximum generation retries per tool.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation length budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for generation. Use auto, cuda, auto, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype for model weights.",
    )
    return parser.parse_args()


def load_tools(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"Expected 'tools' to be a list in {path}")
    return tools


def format_parameters(tool: Dict[str, Any]) -> str:
    parameters = tool.get("parameters", {})
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    if not properties:
        return "- no parameters"

    lines: List[str] = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        parts = [f"- {name}: {spec.get('type', 'any')}"]
        if spec.get("enum"):
            parts.append(f"choices={spec['enum']}")
        if "default" in spec:
            parts.append(f"default={spec['default']}")
        if name in required:
            parts.append("required")
        lines.append(", ".join(parts))
    return "\n".join(lines) if lines else "- no parameters"


def build_prompt(tool: Dict[str, Any], count: int) -> str:
    name = tool.get("name", "").strip()
    description = tool.get("description", "").strip()
    parameter_text = format_parameters(tool)
    return f"""Create {count} different user requests for this tool.

Tool name: {name}
Tool description: {description}
Parameters:
{parameter_text}

Requirements:
- The response must contain exactly {count} requests.
- The request should clearly map to this tool.
- Keep the language simple and direct.
- Vary names, locations, dates, numbers, and phrasing.
- Some requests can mention optional parameters when relevant.
- Avoid duplicates.
- Each item in requests should be a standalone natural user request."""


def normalize_query(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def normalize_queries(items: List[Any]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        query = normalize_query(item)
        if query:
            normalized.append(query)
    return unique_preserve_order(normalized)


@dataclass
class ToolGenerationState:
    tool: Dict[str, Any]
    tool_index: int
    tool_name: str
    collected: List[str]
    attempts: int = 0


@dataclass
class CheckpointData:
    tool_states: Dict[str, Dict[str, Any]]
    work_order: List[str]


def build_tool_state_key(tool_index: int, tool_name: str) -> str:
    return f"{tool_index:06d}:{tool_name}"


def resolve_checkpoint_path(output_path: Path, checkpoint_path: str) -> Path:
    requested_path = checkpoint_path.strip()
    if requested_path:
        return Path(requested_path)
    return output_path.with_suffix(f"{output_path.suffix}.checkpoint.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp_path.replace(path)


def load_checkpoint(
    checkpoint_path: Path,
    tools_sha256: str,
    examples_per_tool: int,
    model_name: str,
) -> CheckpointData:
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("version") not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Unsupported checkpoint version in {checkpoint_path}: "
            f"{payload.get('version')!r}"
        )

    metadata = payload.get("metadata", {})
    if metadata.get("tools_sha256") != tools_sha256:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created for a different tools file. "
            "Use --overwrite to start from scratch."
        )
    if metadata.get("examples_per_tool") != examples_per_tool:
        raise ValueError(
            f"Checkpoint {checkpoint_path} requested "
            f"{metadata.get('examples_per_tool')} examples per tool, but this run "
            f"requested {examples_per_tool}. Use --overwrite to start from scratch."
        )
    if metadata.get("generator_model") != model_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created with model "
            f"{metadata.get('generator_model')!r}, but this run uses "
            f"{model_name!r}. Use --overwrite to start from scratch."
        )

    states = payload.get("tool_states", {})
    if not isinstance(states, dict):
        raise ValueError(f"Expected 'tool_states' object in {checkpoint_path}")

    work_order = payload.get("work_order", [])
    if not isinstance(work_order, list):
        work_order = []
    work_order = [key for key in work_order if isinstance(key, str)]
    return CheckpointData(tool_states=states, work_order=work_order)


def save_checkpoint(
    checkpoint_path: Path,
    *,
    tools_path: Path,
    output_path: Path,
    summary_path: Path,
    tools_sha256: str,
    args: argparse.Namespace,
    tool_states: List[ToolGenerationState],
) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "metadata": {
            "tools_path": str(tools_path),
            "tools_sha256": tools_sha256,
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "generator_model": args.model_name,
            "tool_count": len(tool_states),
            "examples_per_tool": args.examples_per_tool,
            "generation_batch_size": args.generation_batch_size,
            "tool_batch_size": args.tool_batch_size,
            "max_attempts_per_tool": args.max_attempts_per_tool,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "work_order": [
            build_tool_state_key(state.tool_index, state.tool_name)
            for state in tool_states
        ],
        "tool_states": {
            build_tool_state_key(state.tool_index, state.tool_name): {
                "collected": normalize_queries(state.collected)[
                    : args.examples_per_tool
                ],
                "attempts": state.attempts,
            }
            for state in tool_states
        },
    }
    write_json_atomic(checkpoint_path, payload)


def restore_tool_state(
    checkpoint_states: Dict[str, Dict[str, Any]],
    used_legacy_checkpoint_keys: set[str],
    tool_index: int,
    tool_name: str,
    examples_per_tool: int,
) -> tuple[List[str], int]:
    checkpoint_key = build_tool_state_key(tool_index, tool_name)
    checkpoint_state = checkpoint_states.get(checkpoint_key)

    if checkpoint_state is None and tool_name not in used_legacy_checkpoint_keys:
        checkpoint_state = checkpoint_states.get(tool_name)
        if checkpoint_state is not None:
            used_legacy_checkpoint_keys.add(tool_name)

    if checkpoint_state is None:
        checkpoint_state = {}

    if not isinstance(checkpoint_state, dict):
        checkpoint_state = {}

    collected = checkpoint_state.get("collected", [])
    if not isinstance(collected, list):
        collected = []
    attempts = checkpoint_state.get("attempts", 0)
    if not isinstance(attempts, int):
        attempts = 0
    return normalize_queries(collected)[:examples_per_tool], max(attempts, 0)


def apply_checkpoint_work_order(
    tool_states: List[ToolGenerationState],
    work_order: List[str],
) -> None:
    if not work_order:
        return

    by_key = {
        build_tool_state_key(state.tool_index, state.tool_name): state
        for state in tool_states
    }
    ordered_states: List[ToolGenerationState] = []
    used_keys = set()
    for key in work_order:
        state = by_key.get(key)
        if state is None:
            continue
        ordered_states.append(state)
        used_keys.add(key)

    ordered_states.extend(
        state
        for state in tool_states
        if build_tool_state_key(state.tool_index, state.tool_name) not in used_keys
    )
    tool_states[:] = ordered_states


def move_tool_states_to_end(
    tool_states: List[ToolGenerationState],
    deferred_states: List[ToolGenerationState],
) -> None:
    if not deferred_states:
        return

    deferred_ids = {id(state) for state in deferred_states}
    original_order = list(tool_states)
    tool_states[:] = [
        state for state in original_order if id(state) not in deferred_ids
    ] + [state for state in original_order if id(state) in deferred_ids]


def build_query_output_model(query_count: int) -> type[BaseModel]:
    return create_model(
        f"GeneratedQueries_{query_count}",
        requests=(List[str], Field(..., min_length=query_count, max_length=query_count)),
    )


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device.startswith("cpu"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_generator(model_name: str, device: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": resolve_dtype(dtype, device),
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()

    structured_model = outlines.from_transformers(model, tokenizer)
    return tokenizer, structured_model


def get_query_generator(
    structured_model,
    generator_cache: Dict[int, Any],
    query_count: int,
) -> tuple[type[BaseModel], Any]:
    output_model, query_generator = generator_cache.get(query_count, (None, None))
    if output_model is None or query_generator is None:
        output_model = build_query_output_model(query_count)
        query_generator = outlines.Generator(structured_model, output_model)
        generator_cache[query_count] = (output_model, query_generator)
    return output_model, query_generator


def build_generation_kwargs(
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False
    return generation_kwargs


def parse_generated_queries(
    output_model: type[BaseModel],
    raw_output: str,
    expected_count: int,
    context: str = "",
) -> List[str]:
    try:
        parsed_output = output_model.model_validate_json(raw_output)
        queries = normalize_queries(parsed_output.requests)
    except Exception as exc:
        context_text = f" for {context}" if context else ""
        first_line = str(exc).splitlines()[0]
        tqdm.write(f"Rejecting model output{context_text}: {first_line}")
        return []

    if len(queries) != expected_count:
        context_text = f" for {context}" if context else ""
        tqdm.write(
            f"Rejecting model output{context_text}: expected "
            f"{expected_count} unique requests, received {len(queries)}"
        )
        return []
    return queries


@torch.inference_mode()
def generate_queries(
    structured_model,
    generator_cache: Dict[int, Any],
    tokenizer: AutoTokenizer,
    prompt: str,
    query_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    context: str = "single prompt",
) -> List[str]:
    output_model, query_generator = get_query_generator(
        structured_model=structured_model,
        generator_cache=generator_cache,
        query_count=query_count,
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    generation_kwargs = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    raw_output = query_generator(full_prompt, **generation_kwargs)
    return parse_generated_queries(
        output_model,
        raw_output,
        expected_count=query_count,
        context=context,
    )


@torch.inference_mode()
def generate_queries_batch(
    structured_model,
    generator_cache: Dict[int, Any],
    tokenizer: AutoTokenizer,
    prompts: List[str],
    query_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    contexts: List[str] | None = None,
) -> List[List[str]]:
    if not prompts:
        return []

    output_model, query_generator = get_query_generator(
        structured_model=structured_model,
        generator_cache=generator_cache,
        query_count=query_count,
    )
    full_prompts = [f"{SYSTEM_PROMPT}\n\n{prompt}" for prompt in prompts]
    generation_kwargs = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    raw_outputs = query_generator.batch(full_prompts, **generation_kwargs)
    if isinstance(raw_outputs, str):
        raw_outputs = [raw_outputs]
    if len(raw_outputs) != len(prompts):
        tqdm.write(
            f"Expected {len(prompts)} generated outputs, received {len(raw_outputs)}"
        )
        return [[] for _ in prompts]
    if contexts is None:
        contexts = [
            f"batch item {index + 1}/{len(raw_outputs)}"
            for index in range(len(raw_outputs))
        ]
    return [
        parse_generated_queries(
            output_model,
            raw_output,
            expected_count=query_count,
            context=contexts[index],
        )
        for index, raw_output in enumerate(raw_outputs)
    ]


def main() -> None:
    args = parse_args()
    if args.tool_batch_size < 1:
        raise ValueError("--tool-batch-size must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tools_path = Path(args.tools_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    checkpoint_path = resolve_checkpoint_path(output_path, args.checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tools = load_tools(tools_path)
    tools_sha256 = sha256_file(tools_path)

    checkpoint_data = CheckpointData(tool_states={}, work_order=[])
    checkpoint_states: Dict[str, Dict[str, Any]] = {}
    resumed_from_checkpoint = False
    if checkpoint_path.exists() and not args.overwrite:
        checkpoint_data = load_checkpoint(
            checkpoint_path=checkpoint_path,
            tools_sha256=tools_sha256,
            examples_per_tool=args.examples_per_tool,
            model_name=args.model_name,
        )
        checkpoint_states = checkpoint_data.tool_states
        resumed_from_checkpoint = True
        print(f"Resuming generation from {checkpoint_path}")
    elif args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    rows: List[Dict[str, Any]] = []
    per_tool_counts: Dict[str, int] = {}
    tool_states: List[ToolGenerationState] = []
    used_legacy_checkpoint_keys: set[str] = set()
    for tool_index, tool in enumerate(tools):
        tool_name = str(tool.get("name", "")).strip()
        if not tool_name:
            continue
        collected, attempts = restore_tool_state(
            checkpoint_states=checkpoint_states,
            used_legacy_checkpoint_keys=used_legacy_checkpoint_keys,
            tool_index=tool_index,
            tool_name=tool_name,
            examples_per_tool=args.examples_per_tool,
        )
        tool_states.append(
            ToolGenerationState(
                tool=tool,
                tool_index=tool_index,
                tool_name=tool_name,
                collected=collected,
                attempts=attempts,
            )
        )

    if used_legacy_checkpoint_keys:
        print(
            "Migrated "
            f"{len(used_legacy_checkpoint_keys)} legacy name-keyed checkpoint "
            "entries to index-keyed checkpoint entries."
        )

    apply_checkpoint_work_order(tool_states, checkpoint_data.work_order)

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        tools_path=tools_path,
        output_path=output_path,
        summary_path=summary_path,
        tools_sha256=tools_sha256,
        args=args,
        tool_states=tool_states,
    )

    has_pending_generation = any(
        len(state.collected) < args.examples_per_tool
        and state.attempts < args.max_attempts_per_tool
        for state in tool_states
    )
    tokenizer = None
    structured_model = None
    if has_pending_generation:
        tokenizer, structured_model = load_generator(
            args.model_name, args.device, args.dtype
        )
    generator_cache: Dict[int, Any] = {}

    query_progress = tqdm(
        total=len(tool_states) * args.examples_per_tool,
        initial=sum(len(state.collected) for state in tool_states),
        desc="Generating queries",
        unit="query",
    )
    try:
        while True:
            active_by_needed: Dict[int, List[ToolGenerationState]] = {}
            for state in tool_states:
                if (
                    len(state.collected) >= args.examples_per_tool
                    or state.attempts >= args.max_attempts_per_tool
                ):
                    continue
                needed = min(
                    args.generation_batch_size,
                    args.examples_per_tool - len(state.collected),
                )
                active_by_needed.setdefault(needed, []).append(state)

            if not active_by_needed:
                break

            for needed, states in active_by_needed.items():
                for start in range(0, len(states), args.tool_batch_size):
                    batch_states = states[start : start + args.tool_batch_size]
                    prompts = [build_prompt(state.tool, needed) for state in batch_states]
                    try:
                        if args.tool_batch_size == 1:
                            query_batches = [
                                generate_queries(
                                    structured_model=structured_model,
                                    generator_cache=generator_cache,
                                    tokenizer=tokenizer,
                                    prompt=prompts[0],
                                    query_count=needed,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    context=(
                                        f"{batch_states[0].tool_name}"
                                        f"#{batch_states[0].tool_index}"
                                    ),
                                )
                            ]
                        else:
                            query_batches = generate_queries_batch(
                                structured_model=structured_model,
                                generator_cache=generator_cache,
                                tokenizer=tokenizer,
                                prompts=prompts,
                                query_count=needed,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                contexts=[
                                    f"{state.tool_name}#{state.tool_index}"
                                    for state in batch_states
                                ],
                            )
                    except Exception as exc:
                        if "out of memory" in str(exc).lower():
                            raise
                        failed_tools = ", ".join(
                            f"{state.tool_name}#{state.tool_index}"
                            for state in batch_states
                        )
                        tqdm.write(
                            "Generation failed for "
                            f"{failed_tools}: {type(exc).__name__}: {exc}"
                        )
                        query_batches = [[] for _ in batch_states]
                    deferred_states: List[ToolGenerationState] = []
                    for state, queries in zip(batch_states, query_batches):
                        previous_count = len(state.collected)
                        if queries:
                            state.collected = unique_preserve_order(
                                state.collected + queries
                            )[: args.examples_per_tool]
                        state.attempts += 1
                        if not queries or len(state.collected) == previous_count:
                            deferred_states.append(state)
                        query_progress.update(len(state.collected) - previous_count)
                    move_tool_states_to_end(tool_states, deferred_states)
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        tools_path=tools_path,
                        output_path=output_path,
                        summary_path=summary_path,
                        tools_sha256=tools_sha256,
                        args=args,
                        tool_states=tool_states,
                    )

                query_progress.set_postfix(
                    active_tools=sum(
                        len(state.collected) < args.examples_per_tool
                        and state.attempts < args.max_attempts_per_tool
                        for state in tool_states
                    ),
                    completed_tools=sum(
                        len(state.collected) >= args.examples_per_tool
                        for state in tool_states
                    ),
                    tool_batch_size=args.tool_batch_size,
                )
    except BaseException:
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            tools_path=tools_path,
            output_path=output_path,
            summary_path=summary_path,
            tools_sha256=tools_sha256,
            args=args,
            tool_states=tool_states,
        )
        raise
    finally:
        query_progress.close()

    tool_name_counts = Counter(state.tool_name for state in tool_states)
    for state in tool_states:
        collected = state.collected[: args.examples_per_tool]
        per_tool_counts[state.tool_name] = (
            per_tool_counts.get(state.tool_name, 0) + len(collected)
        )
        id_prefix = state.tool_name
        if tool_name_counts[state.tool_name] > 1:
            id_prefix = f"{state.tool_name}-{state.tool_index + 1:04d}"

        for index, query in enumerate(collected, start=1):
            rows.append(
                {
                    "id": f"{id_prefix}-{index:04d}",
                    "tool": state.tool_name,
                    "query": query,
                    "text": query,
                    "tool_description": state.tool.get("description", ""),
                    "parameters": state.tool.get("parameters", {}),
                    "source": "qwen_synthetic",
                    "generator_model": args.model_name,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Writing dataset", unit="row"):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    completed_tool_count = sum(
        len(state.collected) >= args.examples_per_tool for state in tool_states
    )
    incomplete_tool_count = len(tool_states) - completed_tool_count
    summary = {
        "tools_path": str(tools_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "generator_model": args.model_name,
        "examples_per_tool_requested": args.examples_per_tool,
        "generation_batch_size": args.generation_batch_size,
        "tool_batch_size": args.tool_batch_size,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "rows_written": len(rows),
        "tool_count": len(tool_states),
        "unique_tool_name_count": len(per_tool_counts),
        "completed_tool_count": completed_tool_count,
        "incomplete_tool_count": incomplete_tool_count,
        "per_tool_counts": per_tool_counts,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if incomplete_tool_count == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()
    elif incomplete_tool_count:
        print(
            f"Kept checkpoint at {checkpoint_path} because "
            f"{incomplete_tool_count} tools are incomplete."
        )

    print(f"\nWrote {len(rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
