from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

import outlines
import torch
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You create short, realistic tool schemas."


class PropertySchema(BaseModel):
    type: str
    default: Any | None = None


class Parameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, PropertySchema] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tool schemas from a CSV with Outlines.")
    parser.add_argument(
        "--tools-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolVerifier/tools.csv",
        help="Path to the source CSV with Name and Description columns.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/scratch4/home/akrik/NTILC/data/ToolVerifier/tools.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of tools to generate per batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation token budget per schema.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
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
        default="cuda:1",
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


def load_tools(path: Path) -> List[Dict[str, str]]:
    tools: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Name", "Description"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Expected CSV columns {required}, got {reader.fieldnames}")

        for row in reader:
            tools.append(
                {
                    "name": row["Name"].strip(),
                    "description": row["Description"].strip(),
                }
            )

    return tools


def build_prompt(tool: Dict[str, str]) -> str:
    return f"""Create a tool schema.

Tool name: {tool["name"]}
Tool description: {tool["description"]}

Return a realistic schema for this tool with:
- name
- description
- parameters.type = "object"
- parameters.properties
- parameters.required

Requirements:
- Keep it short and practical.
- Use a short snake_case tool name.
- Only add parameters that are clearly useful for this tool.
- Use simple parameter types such as string, integer, number, and boolean.
"""


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

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        hf_model = hf_model.to(device)
    hf_model.eval()

    structured_model = outlines.from_transformers(hf_model, tokenizer)
    schema_generator = outlines.Generator(structured_model, ToolSchema)
    return tokenizer, schema_generator


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


def parse_schema(raw_output: str) -> ToolSchema:
    return ToolSchema.model_validate_json(raw_output)


def generate_schema_with_retry(
    schema_generator,
    prompt: str,
    generation_kwargs: Dict[str, Any],
) -> ToolSchema:
    raw_output = schema_generator(prompt, **generation_kwargs)
    return parse_schema(raw_output)


def generate_schemas(
    schema_generator,
    generation_kwargs: Dict[str, Any],
    tools: List[Dict[str, str]],
    batch_size: int,
) -> List[ToolSchema]:
    schemas: List[ToolSchema] = []

    for start in tqdm(range(0, len(tools), batch_size), desc="Generating schemas"):
        batch_tools = tools[start : start + batch_size]
        batch_prompts = [f"{SYSTEM_PROMPT}\n\n{build_prompt(tool)}" for tool in batch_tools]

        try:
            raw_outputs = schema_generator.batch(batch_prompts, **generation_kwargs)
            if isinstance(raw_outputs, str):
                raw_outputs = [raw_outputs]
        except Exception:
            raw_outputs = []

        if len(raw_outputs) == len(batch_tools):
            for tool, prompt, raw_output in zip(batch_tools, batch_prompts, raw_outputs):
                try:
                    schemas.append(parse_schema(raw_output))
                except Exception:
                    try:
                        schemas.append(
                            generate_schema_with_retry(schema_generator, prompt, generation_kwargs)
                        )
                    except Exception as exc:
                        raise RuntimeError(f"Failed to generate schema for {tool['name']}") from exc
            continue

        for tool, prompt in zip(batch_tools, batch_prompts):
            try:
                schemas.append(generate_schema_with_retry(schema_generator, prompt, generation_kwargs))
            except Exception as exc:
                raise RuntimeError(f"Failed to generate schema for {tool['name']}") from exc

    return schemas


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tools = load_tools(Path(args.tools_path))
    if args.limit is not None:
        tools = tools[: args.limit]

    tokenizer, schema_generator = load_generator(args.model_name, args.device, args.dtype)
    generation_kwargs = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    schemas = generate_schemas(
        schema_generator=schema_generator,
        generation_kwargs=generation_kwargs,
        tools=tools,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tools": [schema.model_dump(mode="json") for schema in schemas],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved {len(schemas)} tool schemas to {output_path}")


if __name__ == "__main__":
    main()
