"""CLI entrypoint for NTILC orchestrator inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator.agent import NTILCOrchestratorAgent


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NTILC full inference runtime.")
    parser.add_argument("--request", type=str, required=True, help="User request.")
    parser.add_argument(
        "--query-encoder-path",
        type=str,
        default="checkpoints/cluster_retrieval/best_model.pt",
        help="Path to cluster retrieval checkpoint.",
    )
    parser.add_argument(
        "--intent-embedder-path",
        type=str,
        default="checkpoints/intent_embedder/best_model.pt",
        help="Kept for API compatibility; not used by retrieval runtime.",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Base generation model path/name.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default="checkpoints/lora_nl_command_full",
        help="Optional LoRA adapter path. Set empty string to disable.",
    )
    parser.add_argument("--lora-mode", choices=["full", "tail"], default="full")
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--execute-tools", action="store_true", help="Execute tools via dispatcher.")
    parser.add_argument(
        "--tool-timeout-seconds",
        type=int,
        default=20,
        help="Per-tool timeout when execute-tools=true.",
    )
    parser.add_argument(
        "--tool-cwd",
        type=str,
        default=None,
        help="Working directory for shell tools.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-plan-actions", type=int, default=8)
    parser.add_argument("--plan-max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write result JSON.")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    lora_adapter_path = str(args.lora_adapter_path).strip() or None
    qwen_model_name_or_path = str(args.qwen_model).strip() or None

    agent = NTILCOrchestratorAgent.from_pretrained(
        intent_embedder_path=args.intent_embedder_path,
        query_encoder_path=args.query_encoder_path,
        qwen_model_name_or_path=qwen_model_name_or_path,
        lora_adapter_path=lora_adapter_path,
        lora_mode=args.lora_mode,
        auto_register_shell_tools=True,
        tool_timeout_seconds=args.tool_timeout_seconds,
        tool_cwd=args.tool_cwd,
        device=args.device,
    )

    run = agent.run(
        request=args.request,
        execute_tools=args.execute_tools,
        top_k_candidates=args.top_k_candidates,
        max_retries=args.max_retries,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_plan_actions=args.max_plan_actions,
        plan_max_new_tokens=args.plan_max_new_tokens,
    )
    payload = run.to_dict()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
