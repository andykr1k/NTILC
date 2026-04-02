from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You create short, realistic user requests for tool-routing datasets.
Each string must be a natural user request.
Do not mention tool names, JSON, schemas, or implementation details.
Do not number the items."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple tool-query dataset.")
    parser.add_argument(
        "--tools-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolCall15/tools.json",
        help="Path to the tools.json file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolCall15/tool_embedding_dataset.jsonl",
        help="Path to the output JSONL dataset.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolCall15/tool_embedding_dataset_summary.json",
        help="Path to a small metadata summary JSON file.",
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
        default=12,
        help="How many synthetic queries to create per tool.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=64,
        help="How many queries to ask the model for in one generation step.",
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
        default="cuda:2",
        help="Device for generation. Use auto, cuda, cuda:0, or cpu.",
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
- The request should clearly map to this tool.
- Keep the language simple and direct.
- Vary names, locations, dates, numbers, and phrasing.
- Some requests can mention optional parameters when relevant.
- Avoid duplicates.
- Each user request should start with <request> and end with </request>
- Only output the requests"""


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


def parse_generated_queries(raw_text: str) -> List[str]:
    parts = raw_text.split("</think>")
    post_think_text = parts[-1]
    matches = re.findall(r"<request>(.*?)</request>", post_think_text, flags=re.DOTALL)
    queries = [normalize_query(match) for match in matches if normalize_query(match)]
    return unique_preserve_order(queries)


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
        "torch_dtype": resolve_dtype(dtype, device),
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_queries(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"{SYSTEM_PROMPT}\n\n{prompt}"

    encoded = tokenizer(text, return_tensors="pt")
    if device != "auto":
        encoded = {key: value.to(device) for key, value in encoded.items()}

    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_tokens = encoded["input_ids"].shape[1]
    new_tokens = generated[0][prompt_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tools_path = Path(args.tools_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    tools = load_tools(tools_path)
    tokenizer, model = load_generator(args.model_name, args.device, args.dtype)

    rows: List[Dict[str, Any]] = []
    per_tool_counts: Dict[str, int] = {}

    for tool in tqdm(tools, desc="Generating tools", unit="tool"):
        tool_name = str(tool.get("name", "")).strip()
        if not tool_name:
            continue

        collected: List[str] = []
        attempts = 0
        query_progress = tqdm(
            total=args.examples_per_tool,
            desc=f"{tool_name}",
            unit="query",
            leave=False,
        )

        while len(collected) < args.examples_per_tool and attempts < args.max_attempts_per_tool:
            needed = min(
                args.generation_batch_size,
                args.examples_per_tool - len(collected),
            )
            prompt = build_prompt(tool, needed)
            raw_output = generate_queries(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            queries = parse_generated_queries(raw_output)
            previous_count = len(collected)
            collected = unique_preserve_order(collected + queries)
            query_progress.update(len(collected) - previous_count)
            attempts += 1
            query_progress.set_postfix(
                attempts=attempts,
                parsed=len(queries),
                kept=len(collected),
            )

        collected = collected[: args.examples_per_tool]
        query_progress.n = len(collected)
        query_progress.refresh()
        query_progress.close()
        per_tool_counts[tool_name] = len(collected)

        for index, query in enumerate(collected, start=1):
            rows.append(
                {
                    "id": f"{tool_name}-{index:04d}",
                    "tool": tool_name,
                    "query": query,
                    "text": query,
                    "tool_description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "source": "qwen_synthetic",
                    "generator_model": args.model_name,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Writing dataset", unit="row"):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "tools_path": str(tools_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "generator_model": args.model_name,
        "examples_per_tool_requested": args.examples_per_tool,
        "rows_written": len(rows),
        "tool_count": len(per_tool_counts),
        "per_tool_counts": per_tool_counts,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nWrote {len(rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
