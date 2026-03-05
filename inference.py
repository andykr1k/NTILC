"""
End-to-end inference runtime for NTILC.

This file integrates:
1) Query -> cluster retrieval
2) Cluster ID -> tool mapping via software layer
3) LoRA/LLM command generation
4) Dispatcher validation and optional execution
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass, field
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.cluster_retrieval import ClusterRetrieval
from models.query_encoder import QueryEncoder
from models.software_layer import (
    ClusterToolMapper,
    DispatchResult,
    ToolDispatcher,
)

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency at runtime.
    PeftModel = None


def build_prompt(query: str, tool: str, mode: str) -> str:
    if mode == "tail":
        return (
            "You map a user request to shell command arguments.\n"
            "Given the selected tool and request, output only the command tail (arguments and values).\n"
            "Do not repeat the tool name.\n\n"
            f"Tool: {tool}\n"
            f"User request: {query}\n"
            "Command tail:"
        )
    return (
        "You map a user request to exactly one Linux shell command.\n"
        "Output only the command and nothing else.\n\n"
        f"User request: {query}\n"
        "Command:"
    )


def build_full_from_tail(tool: str, tail: str) -> str:
    tail_text = str(tail).strip()
    if not tail_text or tail_text == "<NO_ARGS>":
        return tool
    return f"{tool} {tail_text}"


def _safe_first_line(text: str) -> str:
    line = str(text).strip().splitlines()
    return line[0].strip() if line else ""


def _normalize_command_for_tool(
    tool: str,
    generated_text: str,
    mode: str,
    enforce_selected_tool: bool = True,
) -> str:
    cleaned = _safe_first_line(generated_text)
    if mode == "tail":
        return build_full_from_tail(tool=tool, tail=cleaned)

    if not cleaned:
        return tool

    if not enforce_selected_tool:
        return cleaned

    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        tokens = cleaned.split()

    if not tokens:
        return tool
    if tokens[0] == tool:
        return cleaned
    if len(tokens) == 1:
        return f"{tool} {tokens[0]}".strip()
    return f"{tool} {' '.join(tokens[1:])}".strip()


def _len_tag(tag: str, value: str) -> str:
    text = str(value)
    return f"<{tag}><len:{len(text)}>{xml_escape(text)}</len></{tag}>"


def _dispatch_block(tool: str, arguments: Mapping[str, Any]) -> str:
    lines = ["<dispatch>", f"  {_len_tag('tool', tool)}"]
    for name, value in arguments.items():
        arg_text = str(value)
        lines.append(
            f"  <arg name=\"{xml_escape(str(name))}\"><len:{len(arg_text)}>{xml_escape(arg_text)}</len></arg>"
        )
    lines.append("</dispatch>")
    return "\n".join(lines)


def _response_block(tool: str, dispatch_result: DispatchResult, retry: bool) -> str:
    if dispatch_result.ok:
        text = "ok"
        if isinstance(dispatch_result.result, Mapping):
            returncode = dispatch_result.result.get("returncode")
            stdout = str(dispatch_result.result.get("stdout", "")).strip()
            if returncode is not None:
                text = f"returncode={returncode}"
            if stdout:
                text = f"{text}; stdout={stdout[:200]}"
    else:
        text = "; ".join(dispatch_result.errors) if dispatch_result.errors else "dispatch failed"

    lines = [
        "<response>",
        f"  {_len_tag('tool', tool)}",
        f"  {_len_tag('status', 'ok' if dispatch_result.ok else 'fail')}",
        f"  {_len_tag('text', text)}",
        f"  {_len_tag('retry', 'true' if retry else 'false')}",
        "</response>",
    ]
    return "\n".join(lines)


def _plan_block(request: str) -> str:
    return "\n".join(
        [
            "<plan>",
            f"  <action><len:{len(request)}>{xml_escape(request)}</len></action>",
            "</plan>",
        ]
    )


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


@dataclass
class RetrievalCandidate:
    cluster_id: int
    score: float
    tool_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": int(self.cluster_id),
            "score": float(self.score),
            "tool_name": self.tool_name,
        }


@dataclass
class OrchestratorStepResult:
    candidate: RetrievalCandidate
    generated_text: str
    command: str
    dispatch_arguments: Dict[str, Any]
    dispatch_result: DispatchResult
    dispatch_block: str
    response_block: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "generated_text": self.generated_text,
            "command": self.command,
            "dispatch_arguments": self.dispatch_arguments,
            "dispatch_result": self.dispatch_result.to_dict(),
            "dispatch_block": self.dispatch_block,
            "response_block": self.response_block,
        }


@dataclass
class OrchestratorRunResult:
    request: str
    plan_block: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    steps: List[OrchestratorStepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return any(step.dispatch_result.ok for step in self.steps)

    @property
    def final_step(self) -> Optional[OrchestratorStepResult]:
        return self.steps[-1] if self.steps else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "plan_block": self.plan_block,
            "success": self.success,
            "candidates": [c.to_dict() for c in self.candidates],
            "steps": [s.to_dict() for s in self.steps],
        }


class ClusterBasedToolSystem:
    """
    Retrieval-only inference: query -> top-k cluster/tool candidates.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        cluster_centroids: torch.Tensor,
        mapper: ClusterToolMapper,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.query_encoder = query_encoder
        self.cluster_centroids = F.normalize(cluster_centroids.float(), p=2, dim=1)
        self.mapper = mapper
        self.max_length = int(max_length)
        self.device = device or next(query_encoder.parameters()).device

        self.cluster_retrieval = ClusterRetrieval(
            embedding_dim=int(self.cluster_centroids.shape[1]),
            num_clusters=int(self.cluster_centroids.shape[0]),
            similarity_type="cosine",
        )

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: Optional[str] = None,
        query_encoder_path: str = "checkpoints/cluster_retrieval/best_model.pt",
        device: Optional[str] = None,
    ) -> "ClusterBasedToolSystem":
        del intent_embedder_path  # Kept for API compatibility with README examples.
        checkpoint = torch.load(query_encoder_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        cluster_centroids = checkpoint["cluster_centroids"]
        if not isinstance(cluster_centroids, torch.Tensor):
            cluster_centroids = torch.tensor(cluster_centroids, dtype=torch.float32)

        num_clusters = int(cluster_centroids.shape[0])
        tool_names = checkpoint.get("tool_names", [])
        if not isinstance(tool_names, list):
            tool_names = []
        if len(tool_names) < num_clusters:
            tool_names = list(tool_names) + [f"cluster_{i}" for i in range(len(tool_names), num_clusters)]
        else:
            tool_names = list(tool_names[:num_clusters])

        mapper = ClusterToolMapper.from_tool_names(tool_names)

        target_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_dtype_name = str(config.get("torch_dtype", "float32"))
        model_dtype = _resolve_torch_dtype(model_dtype_name)
        if target_device.type == "cpu":
            # Keep CPU inference on float32 for broad operator support.
            model_dtype_name = "float32"
            model_dtype = torch.float32
        query_encoder = QueryEncoder(
            base_model=str(config.get("encoder_model", "Qwen/Qwen3.5-9B")),
            output_dim=int(config.get("projection_dim", cluster_centroids.shape[1])),
            dropout=float(config.get("dropout", 0.15)),
            torch_dtype=model_dtype_name if model_dtype_name in {"float16", "bfloat16", "float32"} else "float32",
        )
        query_encoder.load_state_dict(checkpoint["query_encoder_state_dict"])
        query_encoder = query_encoder.to(target_device).to(model_dtype)
        query_encoder.eval()

        return cls(
            query_encoder=query_encoder,
            cluster_centroids=cluster_centroids.to(target_device).to(model_dtype),
            mapper=mapper,
            max_length=int(config.get("max_length", 256)),
            device=target_device,
        )

    def encode_query(self, query: str) -> torch.Tensor:
        encoded = self.query_encoder.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            query_embedding = self.query_encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        return query_embedding

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = -1.0,
    ) -> List[RetrievalCandidate]:
        query_embedding = self.encode_query(query)
        with torch.no_grad():
            retrieval = self.cluster_retrieval(
                query_embeddings=query_embedding,
                cluster_embeddings=self.cluster_centroids.to(query_embedding.dtype),
                top_k=max(1, int(top_k)),
                threshold=float(threshold),
            )

        cluster_ids = retrieval["cluster_ids"][0].detach().cpu().tolist()
        scores = retrieval["similarities"][0].detach().cpu().tolist()
        candidates: List[RetrievalCandidate] = []
        for cluster_id, score in zip(cluster_ids, scores):
            cid = int(cluster_id)
            if cid < 0:
                continue
            try:
                tool_name = self.mapper.cluster_to_tool(cid)
            except KeyError:
                tool_name = f"cluster_{cid}"
            candidates.append(
                RetrievalCandidate(cluster_id=cid, score=float(score), tool_name=tool_name)
            )
        return candidates

    def predict(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = -1.0,
    ) -> Dict[str, Any]:
        candidates = self.retrieve_candidates(query=query, top_k=top_k, threshold=threshold)
        return {
            "query": query,
            "top_k": top_k,
            "candidates": [candidate.to_dict() for candidate in candidates],
        }


class QwenOrchestratorModel:
    """
    Command generator (base Qwen, optionally with a LoRA adapter).
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
    ) -> "QwenOrchestratorModel":
        if mode not in {"full", "tail"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected `full` or `tail`.")

        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(None)
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            **model_kwargs,
        )

        if lora_adapter_path:
            if PeftModel is None:
                raise ImportError(
                    "LoRA adapter path provided but `peft` is not installed. Install `peft` first."
                )
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            model = base_model

        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            mode=mode,
            max_seq_len=max_seq_len,
            enforce_selected_tool=enforce_selected_tool,
        )

    def generate(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        prompt = build_prompt(query=query, tool=tool, mode=self.mode)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _safe_first_line(generated_text)

    def generate_command(
        self,
        query: str,
        tool: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, str]:
        generated_text = self.generate(
            query=query,
            tool=tool,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        command = _normalize_command_for_tool(
            tool=tool,
            generated_text=generated_text,
            mode=self.mode,
            enforce_selected_tool=self.enforce_selected_tool,
        )
        return {"generated_text": generated_text, "command": command}


class NTILCOrchestratorAgent:
    """
    Full NTILC inference controller:
    retrieval -> tool mapping -> command generation -> dispatch.
    """

    def __init__(
        self,
        retrieval_system: ClusterBasedToolSystem,
        dispatcher: ToolDispatcher,
        qwen_model: Optional[QwenOrchestratorModel] = None,
    ):
        self.retrieval_system = retrieval_system
        self.dispatcher = dispatcher
        self.qwen_model = qwen_model

    @classmethod
    def from_pretrained(
        cls,
        intent_embedder_path: Optional[str],
        query_encoder_path: str,
        qwen_model_name_or_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        lora_mode: str = "full",
        auto_register_shell_tools: bool = True,
        tool_timeout_seconds: int = 20,
        tool_cwd: Optional[str] = None,
        fail_on_nonzero_exit: bool = True,
        device: Optional[str] = None,
    ) -> "NTILCOrchestratorAgent":
        retrieval_system = ClusterBasedToolSystem.from_pretrained(
            intent_embedder_path=intent_embedder_path,
            query_encoder_path=query_encoder_path,
            device=device,
        )

        mapper = retrieval_system.mapper
        if auto_register_shell_tools:
            mapper.register_shell_tools_for_all_clusters(
                timeout_seconds=tool_timeout_seconds,
                cwd=tool_cwd,
            )
        dispatcher = ToolDispatcher(mapper=mapper, fail_on_nonzero_exit=fail_on_nonzero_exit)

        qwen_model: Optional[QwenOrchestratorModel] = None
        if qwen_model_name_or_path:
            qwen_model = QwenOrchestratorModel.from_pretrained(
                qwen_model_name_or_path=qwen_model_name_or_path,
                lora_adapter_path=lora_adapter_path,
                mode=lora_mode,
            )

        return cls(
            retrieval_system=retrieval_system,
            dispatcher=dispatcher,
            qwen_model=qwen_model,
        )

    def run(
        self,
        request: str,
        execute_tools: bool = False,
        top_k_candidates: int = 3,
        max_retries: int = 2,
        similarity_threshold: float = -1.0,
        granted_permissions: Optional[Iterable[str]] = None,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> OrchestratorRunResult:
        candidates = self.retrieval_system.retrieve_candidates(
            query=request,
            top_k=top_k_candidates,
            threshold=similarity_threshold,
        )

        run_result = OrchestratorRunResult(
            request=request,
            plan_block=_plan_block(request),
            candidates=candidates,
            steps=[],
        )

        if not candidates:
            empty_dispatch = DispatchResult(
                ok=False,
                tool="",
                arguments={},
                result=None,
                errors=["No candidates retrieved from cluster retrieval."],
                executed=False,
            )
            step = OrchestratorStepResult(
                candidate=RetrievalCandidate(cluster_id=-1, score=0.0, tool_name=""),
                generated_text="",
                command="",
                dispatch_arguments={},
                dispatch_result=empty_dispatch,
                dispatch_block=_dispatch_block("", {}),
                response_block=_response_block("", empty_dispatch, retry=False),
            )
            run_result.steps.append(step)
            return run_result

        max_attempts = min(len(candidates), max(1, int(max_retries) + 1))

        for attempt_idx in range(max_attempts):
            candidate = candidates[attempt_idx]

            if self.qwen_model is not None:
                generated = self.qwen_model.generate_command(
                    query=request,
                    tool=candidate.tool_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated_text = generated["generated_text"]
                command = generated["command"]
            else:
                generated_text = candidate.tool_name
                command = candidate.tool_name

            dispatch_arguments = {
                "command": command,
                "query": request,
            }
            dispatch_result = self.dispatcher.dispatch_cluster(
                cluster_id=candidate.cluster_id,
                arguments=dispatch_arguments,
                execute=execute_tools,
                granted_permissions=granted_permissions,
            )

            has_next_attempt = (attempt_idx + 1) < max_attempts
            step = OrchestratorStepResult(
                candidate=candidate,
                generated_text=generated_text,
                command=command,
                dispatch_arguments=dispatch_arguments,
                dispatch_result=dispatch_result,
                dispatch_block=_dispatch_block(candidate.tool_name, dispatch_arguments),
                response_block=_response_block(candidate.tool_name, dispatch_result, retry=has_next_attempt and not dispatch_result.ok),
            )
            run_result.steps.append(step)

            if dispatch_result.ok:
                break

        return run_result


def _build_cli_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write result JSON.")
    return parser


def main() -> None:
    parser = _build_cli_parser()
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
    )
    payload = run.to_dict()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
